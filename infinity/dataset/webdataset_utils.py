
import numpy as np
import torch
from PIL import Image
import io
import random
import json

from infinity.dataset.dataset_multiview_iterable import transform_wintr
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates


def _training_view_counts(args):
    """Return stackable source/target counts for one training batch."""
    choices_raw = str(getattr(args, "N_views_src_choices", "") or "").strip()
    if choices_raw:
        try:
            source_choices = [
                int(value.strip())
                for value in choices_raw.strip("[]()").split(",")
                if value.strip()
            ]
        except ValueError as exc:
            raise ValueError(
                "N_views_src_choices must be a comma-separated list of positive "
                f"integers, got {choices_raw!r}"
            ) from exc
        if not source_choices or any(count <= 0 for count in source_choices):
            raise ValueError(
                "N_views_src_choices must contain at least one positive integer, "
                f"got {choices_raw!r}"
            )
        if args.N_views_src > 0:
            raise ValueError("Set either N_views_src or N_views_src_choices, not both.")
        if args.N_views_tgt <= 0:
            raise ValueError(
                "N_views_src_choices requires N_views_tgt to be a positive fixed count."
            )
        required_views = max(source_choices) + args.N_views_tgt
        if args.N_views_total < required_views:
            raise ValueError(
                "N_views_total must provide enough candidate views for the largest "
                f"episode: need at least {required_views} for "
                f"N_views_src_choices={choices_raw!r} and "
                f"N_views_tgt={args.N_views_tgt}, got {args.N_views_total}."
            )
        return random.choice(source_choices), args.N_views_tgt

    if args.N_views_src > 0 and args.N_views_tgt > 0:
        required_views = args.N_views_src + args.N_views_tgt
        if args.N_views_total < 0 or 0 < args.N_views_total < required_views:
            raise ValueError(
                "N_views_total must be 0 (derive from source/target counts) or "
                f"at least {required_views}, got {args.N_views_total}."
            )
        return args.N_views_src, args.N_views_tgt

    if args.N_views_total < 2:
        raise ValueError(
            "Set positive N_views_src and N_views_tgt, or set N_views_total "
            f"to at least 2 for a random source/target split, got {args.N_views_total}."
        )
    return random.randint(1, args.N_views_total - 1), None


def process_multiview_rgb(sample, args, is_eval=False, eval_dict=None): 
    raw_json_bytes = sample["json"]
    metadata = json.loads(raw_json_bytes.decode('utf-8'))
    scene_key = sample["__key__"]

    # --- FILTERING LOGIC ---
    if is_eval and eval_dict is not None:
        # If this scene isn't in our JSONL, return None (we will drop it later)
        if scene_key not in eval_dict:
            return None

    frames_meta = metadata.get("frames", {})
    available_views = list(frames_meta.keys())
    if not available_views:
        print(f"Skipping scene {scene_key}: no frames in metadata")
        return None

    def load_specific_views(chosen_views, n_src):
        images, poses, intrs, texts = [], [], [], []
        h_div_w = 1.0
        h_div_w_template = h_div_w_templates[np.argmin(np.abs(h_div_w - h_div_w_templates))]
        tgt_h, tgt_w = dynamic_resolution_h_w[h_div_w_template][args.pn]['pixel']

        blender_to_opencv = torch.diag(torch.tensor([1,-1,-1], dtype=torch.float32))
        for view_id in chosen_views:
            view_meta = frames_meta[view_id]
            img_bytes = sample[f"{view_id}.webp"]
            img = Image.open(io.BytesIO(img_bytes))
            if img.mode == 'RGBA':
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                img = background
            else:
                img = img.convert("RGB")
            
            img_tensor, new_K = transform_wintr(img, view_meta['K'], tgt_h, tgt_w)
            pose = torch.tensor(view_meta['transform_matrix_original'], dtype=torch.float32)

            pose[:3,:3] = pose[:3,:3] @ blender_to_opencv
            
            images.append(img_tensor)
            poses.append(pose)
            intrs.append(new_K)
            texts.append(view_meta.get('text', '')) # Texts uncommented!

        poses = torch.stack(poses) 
        ref_c2w = poses[0] 
        ref_w2c = torch.linalg.inv(ref_c2w)
        poses_centered = torch.matmul(ref_w2c.unsqueeze(0), poses)

        dists = torch.norm(poses_centered[:, :3, 3], dim=-1)
        max_dist = dists.max()
        scale = 1.0 / max_dist if max_dist > 1e-5 else 1.0
        poses_centered[:, :3, 3] *= scale

        return {
            "images": torch.stack(images),
            "poses": poses_centered,  
            "intrs": torch.stack(intrs),
            "n_src": n_src,
            "texts": texts # Texts uncommented!
        }

    # --- EXACT VIEW SELECTION ---
    if is_eval and eval_dict is not None:
        available_views.sort() # Sort alphabetically to match your old HDF5 integer indices
        
        scene_tasks = []
        
        for pair in eval_dict[scene_key]:
            src_indices = pair["src"]
            tgt_indices = pair["tgt"]
            
            try:
                chosen_views = [available_views[i] for i in src_indices + tgt_indices]
                scene_tasks.append(load_specific_views(chosen_views, len(src_indices)))
            except IndexError:
                continue # Skip if this specific pair fails, try the next pair
                
        return scene_tasks if len(scene_tasks) > 0 else None
    else:
        n_src, n_tgt = _training_view_counts(args)
        n_views = args.N_views_total or (n_src + n_tgt)
        # Training mode: Pick random views
        if len(available_views) >= n_views:
            chosen_views = random.sample(available_views, n_views)
        else:
            chosen_views = random.choices(available_views, k=n_views)
    
    images, poses, intrs, texts = [], [], [], []
    
    h_div_w = 1.0
    h_div_w_template = h_div_w_templates[np.argmin(np.abs(h_div_w - h_div_w_templates))]
    tgt_h, tgt_w = dynamic_resolution_h_w[h_div_w_template][args.pn]['pixel']

    blender_to_opencv = torch.diag(torch.tensor([1,-1,-1], dtype=torch.float32))
    for view_id in chosen_views:
        # view_meta = metadata[view_id]
        view_meta = frames_meta[view_id]
        
        # 1. Image Loading & RGBA Fix (White Background)
        try:
            img_bytes = sample[f"{view_id}.webp"]
            img = Image.open(io.BytesIO(img_bytes))
            img.load()  # Force load to catch half-downloaded/truncated files
        except Exception as e:
            print(f"Skipping corrupted scene {scene_key}: {e}") # Optional: uncomment to see how often it happens
            return None # The pipeline will drop this scene and grab the next one!
        
        if img.mode == 'RGBA':
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background
        else:
            img = img.convert("RGB")
        
        img_tensor, new_K = transform_wintr(img, view_meta['K'], tgt_h, tgt_w)
        
        # 2. Extract initial Poses (C2W)
        # pose = torch.eye(4)
        # pose[:3, :3] = torch.tensor(view_meta['R'])
        # pose[:3, 3] = torch.tensor(view_meta['T'])

        #BY DEFAULT: BLENDER C2W
        pose = torch.tensor(view_meta['transform_matrix_original'], dtype=torch.float32)
        # CONVERT TO OPENCV C2W
        pose[:3,:3] = pose[:3,:3] @ blender_to_opencv
        
        images.append(img_tensor)
        poses.append(pose)
        intrs.append(new_K)
        # texts.append(view_meta['text'])

    poses = torch.stack(poses) # [N_views, 4, 4]

    # 3. Apply Camera Pose Normalization (Centered on view 0, scaled to max dist 1.0)
    # We use the FIRST view in the chosen list as the reference origin.
    ref_c2w = poses[0]
    ref_w2c = torch.linalg.inv(ref_c2w)
    
    # Transform all poses: New_Pose = Ref_W2C @ Old_Pose
    poses_centered = torch.matmul(ref_w2c.unsqueeze(0), poses)

    # Determine scale factor (using only source views if we knew the split here, 
    # but scaling based on all chosen views is mathematically safer for the whole scene).
    dists = torch.norm(poses_centered[:, :3, 3], dim=-1)
    max_dist = dists.max()
    
    scale = 1.0 / max_dist if max_dist > 1e-5 else 1.0
    poses_centered[:, :3, 3] *= scale

    # print(f"poses_centered: {poses_centered}")
    # print(f"np.linalg.norm(poses_centered[0][:3,3]): {np.linalg.norm(poses_centered[0][:3,3])}")
    # print(f"np.linalg.norm(poses_centered[1][:3,3]): {np.linalg.norm(poses_centered[1][:3,3])}")

    return {
        "images": torch.stack(images),
        "poses": poses_centered,  # Replaces 'rots' and 'trans' 
        "intrs": torch.stack(intrs),
        "n_src": n_src
        # "texts": texts 
    }


def collate_multiview_rgb(batched_list, args, is_eval=False):
    """Takes a list of normalized scene dictionaries and batches them together"""

    n_src, n_tgt = _training_view_counts(args) if not is_eval else (None, None)

    b_img_src, b_pose_src, b_intr_src, b_txt_src = [], [], [], []
    b_img_tgt, b_pose_tgt, b_intr_tgt, b_txt_tgt = [], [], [], []

    for scene in batched_list:
        if is_eval:
            n_src = scene["n_src"]
        target_end = None if n_tgt is None else n_src + n_tgt
        # n_src = scene["n_src"]
        b_img_src.append(scene["images"][:n_src])
        b_pose_src.append(scene["poses"][:n_src])
        b_intr_src.append(scene["intrs"][:n_src])
        # b_txt_src.append(scene["texts"][:n_src]) 

        b_img_tgt.append(scene["images"][n_src:target_end])
        b_pose_tgt.append(scene["poses"][n_src:target_end])
        b_intr_tgt.append(scene["intrs"][n_src:target_end])
        # b_txt_tgt.append(scene["texts"][n_src:])

    # Stack into [Batch, N_views, ...]
    image_src = torch.stack(b_img_src)
    pose_src = torch.stack(b_pose_src)
    intr_src = torch.stack(b_intr_src)
    
    image_tgt = torch.stack(b_img_tgt)
    pose_tgt = torch.stack(b_pose_tgt)
    intr_tgt = torch.stack(b_intr_tgt)

    return (image_src, b_txt_src, pose_src, intr_src, image_tgt, b_txt_tgt, pose_tgt, intr_tgt)

