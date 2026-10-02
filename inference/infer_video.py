import torch
import os.path as osp
import os
import argparse
import time
from PIL import Image
import math

import imageio
from transformers import AutoTokenizer, T5EncoderModel, T5TokenizerFast



from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates
from infinity.models.infinity3d_src2sos import Infinity3DSrc2Sos
from infinity.dataset.dataset_multiview_iterable import MultiviewIterableDataset
from infinity.models.basic import CrossAttnBlock

from inference.vis_utils import *

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

from natsort import natsorted
from glob import glob
import numpy as np
from PIL import Image as PImage
from torchvision.transforms.functional import to_tensor

from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import RotationSpline


def add_common_arguments(parser):
    parser.add_argument('--cfg', type=str, default='4')
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--pn', type=str, choices=['0.06M', '0.25M', '1M'], default='0.06M')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--cfg_insertion_layer', type=int, default=0)
    parser.add_argument('--vae_type', type=int, default=32)
    parser.add_argument('--vae_path', type=str, default='weights/infinity_vae_d32reg.pth')
    parser.add_argument('--add_lvl_embeding_only_first_block', type=int, default=1, choices=[0,1])
    parser.add_argument('--use_bit_label', type=int, default=1, choices=[0,1])
    
    parser.add_argument('--sampling_per_bits', type=int, default=1, choices=[1,2,4,8,16])
    parser.add_argument('--text_encoder_ckpt', type=str, default='weights/flan_t5')
    parser.add_argument('--text_channels', type=int, default=2048)
    parser.add_argument('--apply_spatial_patchify', type=int, default=0, choices=[0,1])
    parser.add_argument('--h_div_w_template', type=float, default=1.000)
    parser.add_argument('--use_flex_attn', type=int, default=0, choices=[0,1])
    parser.add_argument('--enable_positive_prompt', type=int, default=0, choices=[0,1])
    parser.add_argument('--cache_dir', type=str, default='/dev/shm')
    parser.add_argument('--enable_model_cache', type=int, default=0, choices=[0,1])
    parser.add_argument('--checkpoint_type', type=str, default='torch')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--bf16', type=int, default=1, choices=[0,1])

    # to remove:
    parser.add_argument('--rope2d_each_sa_layer', type=int, default=1, choices=[0,1])
    parser.add_argument('--rope2d_normalized_by_hw', type=int, default=2, choices=[0,1,2])
    parser.add_argument('--use_scale_schedule_embedding', type=int, default=0, choices=[0,1])

    parser.add_argument('--use_prope', type=int, default=1, choices=[0,1])
    parser.add_argument('--cos', type=int, default=0, choices=[0,1])
    
    
def interpolate_trajectory(poses, frames_per_segment=15, closed_loop=True):
    """
    Interpolates a smooth camera trajectory through a set of 4x4 pose matrices.
    Uses Rotation Splines for rotations and Cubic Splines for translations.
    """
    poses_np = poses.cpu().numpy()
    N = poses_np.shape[0]

    if closed_loop:
        # Append the first pose to the end to complete the continuous loop
        poses_np = np.concatenate([poses_np, poses_np[0:1]], axis=0)
        N += 1

    # Extract translations
    translations = poses_np[:, :3, 3]
    
    # Extract rotations (with orthogonalization safeguard for messy matrices)
    rot_matrices = poses_np[:, :3, :3]
    # SVD orthogonalization ensures valid rotation matrices
    u, _, vh = np.linalg.svd(rot_matrices)
    rot_matrices_clean = np.matmul(u, vh)
    rotations = R.from_matrix(rot_matrices_clean)

    key_times = np.arange(N)
    
    # Handle endpoints correctly based on loop status
    if closed_loop:
        total_frames = (N - 1) * frames_per_segment
        smooth_times = np.linspace(0, N - 1, total_frames, endpoint=False)
        bc_type = 'periodic'
    else:
        total_frames = (N - 1) * frames_per_segment + 1
        smooth_times = np.linspace(0, N - 1, total_frames, endpoint=True)
        bc_type = 'not-a-knot'

    # 1. Rotation Spline (Smooth continuous rotations instead of linear SLERP)
    spline_rot = RotationSpline(key_times, rotations)
    interp_rotations = spline_rot(smooth_times).as_matrix()

    # 2. Cubic Spline for Translations
    spline_trans = CubicSpline(key_times, translations, axis=0, bc_type=bc_type)
    interp_translations = spline_trans(smooth_times)

    # Reconstruct 4x4 matrices
    interp_poses = np.zeros((total_frames, 4, 4), dtype=np.float32)
    interp_poses[:, :3, :3] = interp_rotations
    interp_poses[:, :3, 3] = interp_translations
    interp_poses[:, 3, 3] = 1.0

    return torch.from_numpy(interp_poses).to(poses.device)

def transform_wintr(pil_img, intr, tgt_h, tgt_w):
    width, height = pil_img.size
    if width / height <= tgt_w / tgt_h:
        scale = tgt_w / width
        resized_width = tgt_w
        resized_height = int(tgt_w / (width / height))
    else:
        scale = tgt_h / height
        resized_height = tgt_h
        resized_width = int((width / height) * tgt_h)

    pil_img = pil_img.resize((resized_width, resized_height), resample=PImage.LANCZOS)
    # crop the center out
    arr = np.array(pil_img)
    crop_y = (arr.shape[0] - tgt_h) // 2
    crop_x = (arr.shape[1] - tgt_w) // 2

    #TODO:  
    # new_intr = intr.clone()
    intr[0,2] = (scale*intr[0,2] - crop_x) # / tgt_w
    intr[1,2] = (scale*intr[1,2] - crop_y) # / tgt_h

    new_intr = torch.tensor([[1.0, 0.0, 0.5],
                             [0.0, 1.0, 0.5],
                             [0.0, 0.0, 1.0]])

    im = to_tensor(arr[crop_y: crop_y + tgt_h, crop_x: crop_x + tgt_w])
    # print(f'im size {im.shape}')
    return im.add(im).add_(-1), new_intr

# converts blender format to opencv
def blender_to_cv(pose):
    blender_to_opencv = np.diag([1,-1,-1])

    pose_cv = np.eye(4).astype(np.float32)

    R_inv = pose[:3,:3].T
    T_inv = -1 * np.dot(R_inv, pose[:3,3])

    pose_cv[:3,:3] = R_inv @ blender_to_opencv
    pose_cv[:3,3] = T_inv
    return pose_cv


def load_scene(scene_path,
               tgt_h=256,
               tgt_w=256):


    image_files = natsorted(glob(osp.join(scene_path, f"*.{args.img_ext}")))
    if len(image_files) == 0:
        raise ValueError(f"No .{args.img_ext} images found in {scene_path}")
    
    images = []
    poses = []
    intrs = []
    print(f"Found {len(image_files)} frames in {scene_path}")

    print(f"image_files: {image_files}")

    for img_path in image_files:
        # Extract the base ID (e.g., '000' from '.../000.webp')
        basename = osp.basename(img_path)
        file_id = osp.splitext(basename)[0]
        
        # Construct paths for corresponding NPY files
        pose_path = osp.join(scene_path, f"{file_id}.npy")
        intr_path = osp.join(scene_path, f"{file_id}_K.npy")
        
        # --- Load Image ---
        # Open image, convert to RGB, resize if necessary (optional)
        img_i: PImage.Image = PImage.open(img_path)
        img_i = img_i.convert("RGBA")
        background = PImage.new("RGB", img_i.size, (255, 255, 255))
        background.paste(img_i, mask=img_i.split()[3])
        img_i=background

        if os.path.exists(intr_path):
            intr_i = torch.from_numpy(np.load(intr_path))
        else:
            intr_i = torch.eye(3)

        pose_i = np.eye(4).astype(np.float32)
        RT_i = np.load(pose_path)
        pose_i[:3,:3] = RT_i[:3,:3]
        pose_i[:3,3] = RT_i[:3,3]

        pose_i = blender_to_cv(pose_i)
        pose_i = torch.from_numpy(pose_i)
        
        img_B3HW, new_K = transform_wintr(img_i, intr_i, tgt_h, tgt_w)

        images.append(img_B3HW)
        poses.append(pose_i)
        intrs.append(new_K)


    images = torch.stack(images)
    poses = torch.stack(poses)
    intrs = torch.stack(intrs)
    
    return images,poses,intrs


def norm_scene(Ts):
    """
    Ts: A PyTorch tensor of shape (N, 4, 4) representing C2W matrices.
    """
    print(f"Ts.shape: {Ts.shape}")
    
    # Clone to avoid modifying the original tensor in-place
    Ts = Ts.clone()
    
    # 1. Centering (Applied only to translations)
    translations = Ts[:, :3, 3]
    center = translations.mean(dim=0, keepdim=True)
    Ts[:, :3, 3] -= center.squeeze()

    # 2. Rescaling (Applied only to translations)
    CAMERA_SCALE = 2.0
    
    # Calculate norm of just the translation vector of the first frame
    t0_norm = torch.norm(Ts[0, :3, 3])
    
    if t0_norm > 1e-5:
        translation_scaling_factor = CAMERA_SCALE / t0_norm
    else:
        translation_scaling_factor = CAMERA_SCALE
        
    # Scale ONLY the translation column
    Ts[:, :3, 3] *= translation_scaling_factor
    
    # 3. Transform everything so Frame 0 is the Origin / Identity
    ref_c2w = Ts[0] 
    ref_w2c = torch.linalg.inv(ref_c2w)
    
    # Apply W2C of frame 0 to all C2W poses
    c2ws_centered = torch.matmul(ref_w2c.unsqueeze(0), Ts)

    # 4. Scale by max distance
    dists = torch.norm(c2ws_centered[:, :3, 3], dim=-1)
    max_dist = dists.max()
    
    if max_dist > 1e-5:
        scale = 1.0 / max_dist
    else:
        scale = 1.0 

    # 5. Apply Final Scale to Translation
    c2ws_centered[:, :3, 3] *= scale

    return c2ws_centered


def load_visual_tokenizer(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load vae
    if args.vae_type in [14,16,18,20,24,32,64]:
        from infinity.models.bsq_vae.vae import vae_model
        schedule_mode = "dynamic"
        codebook_dim = args.vae_type
        codebook_size = 2**codebook_dim
        if args.apply_spatial_patchify:
            patch_size = 8
            encoder_ch_mult=[1, 2, 4, 4]
            decoder_ch_mult=[1, 2, 4, 4]
        else:
            patch_size = 16
            encoder_ch_mult=[1, 2, 4, 4, 4]
            decoder_ch_mult=[1, 2, 4, 4, 4]
        vae = vae_model(args.vae_path, schedule_mode, codebook_dim, codebook_size, patch_size=patch_size, 
                        encoder_ch_mult=encoder_ch_mult, decoder_ch_mult=decoder_ch_mult, test_mode=True).to(device)
    else:
        raise ValueError(f'vae_type={args.vae_type} not supported')
    return vae

def load_infinity(
    rope2d_each_sa_layer, 
    rope2d_normalized_by_hw, 
    use_scale_schedule_embedding, 
    pn, 
    use_bit_label, 
    add_lvl_embeding_only_first_block, 
    model_path='', 
    scale_schedule=None, 
    vae=None, 
    device='cuda', 
    model_kwargs=None,
    text_channels=2048,
    apply_spatial_patchify=0,
    use_flex_attn=False,
    bf16=False,
    checkpoint_type='torch',
    cos_attn=False,
):
    print(f'[Loading Infinity3DSrc2Sos]')
    text_maxlen = 512
    with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16, cache_enabled=True), torch.no_grad():
        infinity_test: Infinity3DSrc2Sos = Infinity3DSrc2Sos(
            vae_local=vae, 
            text_channels=text_channels, 
            text_maxlen=text_maxlen,
            shared_aln=True, 
            raw_scale_schedule=scale_schedule,
            checkpointing='full-block',
            customized_flash_attn=False,
            fused_norm=True,
            pad_to_multiplier=0,
            use_flex_attn=use_flex_attn,
            add_lvl_embeding_only_first_block=add_lvl_embeding_only_first_block,
            use_bit_label=use_bit_label,
            rope2d_each_sa_layer=rope2d_each_sa_layer,
            rope2d_normalized_by_hw=rope2d_normalized_by_hw,
            pn=pn,
            apply_spatial_patchify=apply_spatial_patchify,
            inference_mode=False,
            train_h_div_w_list=[1.0],
            cos_attn=cos_attn,
            **model_kwargs,
        ).to(device=device)
        print(f'[you selected Infinity with {model_kwargs=}] model size: {sum(p.numel() for p in infinity_test.parameters())/1e9:.2f}B, bf16={bf16}')

        if bf16:
            for block in infinity_test.unregistered_blocks:
                block.bfloat16()

        infinity_test.eval()
        infinity_test.requires_grad_(False)

        infinity_test.cuda()
        torch.cuda.empty_cache()

        

        print(f'[Load Infinity weights]')
        if checkpoint_type == 'torch':
            state_dict = torch.load(model_path, map_location=device)
            
            if 'trainer' in state_dict:
                state_dict = state_dict['trainer']['gpt_fsdp']

            
            if not cos_attn:
                for key in list(state_dict):
                    if key.endswith('.scale_mul_1H11'):
                        del state_dict[key]
            print(infinity_test.load_state_dict(state_dict))
        elif checkpoint_type == 'torch_shard':
            from transformers.modeling_utils import load_sharded_checkpoint
            load_sharded_checkpoint(infinity_test, model_path, strict=False)
        infinity_test.rng = torch.Generator(device=device)
        return infinity_test


def load_transformer(vae, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = args.model_path
    if args.checkpoint_type == 'torch': 
        slim_model_path = model_path
        print(f'load checkpoint from {slim_model_path}')
    elif args.checkpoint_type == 'torch_shard':
        slim_model_path = model_path

    if args.model == "2b":
        depth = 32
    elif args.model == "1b":
        depth = 16

    kwargs_model = dict(depth=depth, 
                        embed_dim=2048, 
                        num_heads=2048//128, 
                        drop_path_rate=0.1, 
                        mlp_ratio=4, 
                        block_chunks=8,
                        use_prope=args.use_prope, 
                        N_views_src=args.N_views_src, 
                        N_views_tgt=args.N_views_tgt)

    infinity = load_infinity(
        rope2d_each_sa_layer=args.rope2d_each_sa_layer, 
        rope2d_normalized_by_hw=args.rope2d_normalized_by_hw,
        use_scale_schedule_embedding=args.use_scale_schedule_embedding,
        pn=args.pn,
        use_bit_label=args.use_bit_label, 
        add_lvl_embeding_only_first_block=args.add_lvl_embeding_only_first_block, 
        model_path=slim_model_path, 
        scale_schedule=None, 
        vae=vae, 
        device=device, 
        model_kwargs=kwargs_model,
        text_channels=args.text_channels,
        apply_spatial_patchify=args.apply_spatial_patchify,
        use_flex_attn=args.use_flex_attn,
        bf16=args.bf16,
        checkpoint_type=args.checkpoint_type,
        cos_attn=bool(args.cos),
    )
    return infinity

import infinity.models.basic as basic_mod
def reset_model_cache(model):
    """Force clears KV cache and ProPE grids, explicitly targeting unregistered blocks."""
    if hasattr(model, 'module'): 
        model = model.module
        
    # Helper to aggressively wipe a single attention module
    def wipe_attn(attn):
        if not attn: return
        
        # 1. Reset KV Caching Boolean
        if hasattr(attn, 'kv_caching'):
            try: attn.kv_caching(False)
            except TypeError: pass
        
        # 2. Deep clear all NAMVIS geometry and KV caches
        for cache_name in ['kv_cache', 'rope2d_freqs_grid', 'P_T_seq_cache', 'prope_cache']:
            if hasattr(attn, cache_name):
                val = getattr(attn, cache_name)
                if isinstance(val, (list, dict)):
                    val.clear()
                else:
                    setattr(attn, cache_name, None)

    # 1. Clear Main Block Chunks
    if hasattr(model, 'block_chunks'):
        for block_chunk in model.block_chunks:
            modules = block_chunk.module.module if hasattr(block_chunk.module, 'module') else block_chunk.module
            for m in modules:
                wipe_attn(getattr(m, 'sa', None))
                wipe_attn(getattr(m, 'attn', None))
                wipe_attn(getattr(m, 'ca', None)) # Hit Cross-Attention too!

    # 2. Clear Unregistered Blocks (Where the cache was hiding!)
    if hasattr(model, 'unregistered_blocks'):
        for m in model.unregistered_blocks:
            wipe_attn(getattr(m, 'sa', None))
            wipe_attn(getattr(m, 'attn', None))
            wipe_attn(getattr(m, 'ca', None))
            
    # 3. Clear Globals in basic.py
    if hasattr(basic_mod, 'GLOBAL_SAVED_CROSS_ATTN'):
        basic_mod.GLOBAL_SAVED_CROSS_ATTN.clear()
        
    for var_name in dir(basic_mod):
        if 'cache' in var_name.lower() or 'global' in var_name.lower():
            var = getattr(basic_mod, var_name)
            if isinstance(var, (dict, list)):
                var.clear()
                    
    # 4. Clear visualization globals just in case
    try:
        from infinity.models.basic import GLOBAL_SAVED_CROSS_ATTN
        GLOBAL_SAVED_CROSS_ATTN.clear()
    except Exception:
        pass

def gen_one_img(
    images_src,
    test_poses_src,
    test_poses_tgt,
    test_intrs_src,
    test_intrs_tgt,
    infinity_test, 
    vae, 
    # text_tokenizer,
    # text_encoder,
    # prompt, 
    cfg_list=[],
    tau_list=[],
    # negative_prompt='',
    scale_schedule=None,
    top_k=900,
    top_p=0.97,
    # cfg_sc=3,
    cfg_exp_k=0.0,
    cfg_insertion_layer=0,
    vae_type=0,
    gumbel=0,
    softmax_merge_topk=-1,
    gt_leak=-1,
    gt_ls_Bl=None,
    g_seed=None,
    sampling_per_bits=1,
    enable_positive_prompt=0,
):
    sstt = time.time()
    if not isinstance(cfg_list, list):
        cfg_list = [cfg_list] * len(scale_schedule)
    if not isinstance(tau_list, list):
        tau_list = [tau_list] * len(scale_schedule)

    H, W = images_src.shape[-2], images_src.shape[-1]

    raw_features_src, hs_src, hs_mid_src = vae.encode_for_raw_features(images_src, scale_schedule=scale_schedule)

    with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16, cache_enabled=True):

        infinity_costs = []
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        _, _, img_list, codes_list = infinity_test.autoregressive_infer_cfg(
                vae_features_src=raw_features_src[None],
                vae=vae,
                scale_schedule=scale_schedule,
                poses=test_poses_tgt, #for prope
                intrs=test_intrs_tgt, #for prope
                poses_src=test_poses_src,
                intrs_src=test_intrs_src,
                N_views=test_poses_tgt.shape[1],
                input_size=(H,W),
                g_seed=g_seed,
                B=1,
                # negative_label_B_or_BLT=None, 
                force_gt_Bhw=None,
                # cfg_sc=cfg_sc, 
                cfg_list=cfg_list, 
                tau_list=tau_list, 
                cfg_insertion_layer=cfg_insertion_layer,
                top_k=top_k,
                top_p=top_p,
                returns_vemb=1, 
                ratio_Bl1=None, 
                gumbel=gumbel, 
                norm_cfg=False,
                cfg_exp_k=cfg_exp_k, 
                vae_type=vae_type, 
                ret_img=True, 
                trunk_scale=1000,
                # softmax_merge_topk=softmax_merge_topk,
                gt_leak=gt_leak,
                gt_ls_Bl=gt_ls_Bl, 
                inference_mode=True,
                sampling_per_bits=sampling_per_bits,
            )
        
    # img = img_list[0]
    return img_list.flip(dims=(3,))

def collect_scene_paths(data_path, img_ext):
    if not osp.isdir(data_path):
        raise ValueError(f"Data directory does not exist: {data_path}")

    scene_paths = []
    for entry in sorted(os.scandir(data_path), key=lambda item: item.name):
        if not entry.is_dir():
            continue
        if osp.isfile(osp.join(entry.path, "transforms.json")) or glob(
            osp.join(entry.path, f"*.{img_ext}")
        ):
            scene_paths.append(entry.path)
    if not scene_paths:
        raise ValueError(f"No scene directories found in {data_path}")
    return scene_paths


# to run:
# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 python3 inference/infer_simple.py --model_path weights/random5_1b_ep500/ar-ckpt-giter084K-ep399-iter208-statedict.pth --img_ext png
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)


    scene_input = parser.add_mutually_exclusive_group(required=True)
    scene_input.add_argument('--scene_path', type=str,
                             help='Process one scene directory')
    scene_input.add_argument('--data_path', type=str,
                             help='Process all scene directories under this path')
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--path_indices', type=int, nargs='+', default=None,
                        help='Camera keyframes used for the video path; defaults to all views')
    parser.add_argument('--frames_per_segment', type=int, default=5)
    parser.add_argument('--fps', type=int, default=24)

    parser.add_argument('--N_views_src', type=int, default=3)
    parser.add_argument('--N_views_tgt', type=int, default=2)

    parser.add_argument('--src_indices', type=int, nargs='+', default=None,
                        help='Exact source view indices; otherwise use src_offset')
    parser.add_argument('--src_offset', type=int, default=2)
    parser.add_argument('--tgt_offset', type=int, default=5)

    parser.add_argument('--img_ext', type=str, default="webp")
    parser.add_argument('--model', type=str, default="1b")

    args = parser.parse_args()
    if args.N_views_src < 1 or args.N_views_tgt < 1:
        parser.error("N_views_src and N_views_tgt must be positive")
    if args.frames_per_segment < 1 or args.fps < 1:
        parser.error("frames_per_segment and fps must be positive")
    if args.src_offset < 0:
        parser.error("src_offset must be nonnegative")
    if args.src_indices is not None:
        if len(args.src_indices) != args.N_views_src:
            parser.error("src_indices must contain exactly N_views_src indices")
        if min(args.src_indices) < 0 or len(set(args.src_indices)) != len(args.src_indices):
            parser.error("src_indices must be distinct nonnegative indices")

    if args.data_path:
        try:
            scene_paths = collect_scene_paths(args.data_path, args.img_ext)
        except ValueError as exc:
            parser.error(str(exc))
    else:
        if not osp.isdir(args.scene_path):
            parser.error(f"Scene directory does not exist: {args.scene_path}")
        scene_paths = [args.scene_path]

    os.makedirs(args.out_dir, exist_ok=True)
    args.cfg = list(map(float, args.cfg.split(',')))
    if len(args.cfg) == 1:
        args.cfg = args.cfg[0]

    # load vae
    vae = load_visual_tokenizer(args)
    # load infinity
    infinity = load_transformer(vae, args)

    # infinity = torch.compile(infinity, mode="reduce-overhead")    
    # infinity = torch.compile(infinity)

    scale_schedule = dynamic_resolution_h_w[args.h_div_w_template][args.pn]['scales']
    scale_schedule = [ (1, h, w) for (_, h, w) in scale_schedule]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Found {len(scene_paths)} scenes to process")
    for scene_index, scene_path in enumerate(scene_paths, start=1):
        scene_out_dir = (
            osp.join(args.out_dir, osp.basename(osp.normpath(scene_path)))
            if args.data_path else args.out_dir
        )
        print(f"Processing scene {scene_index}/{len(scene_paths)}: {scene_path}")
        os.makedirs(scene_out_dir, exist_ok=True)

        if osp.isfile(osp.join(scene_path, "transforms.json")):
            from inference.infer_single import load_scene as load_eval_scene
            images, poses, intrs = load_eval_scene(
                scene_path, use_camera_distance_norm=True
            )
        else:
            images, poses, intrs = load_scene(scene_path)
            poses = norm_scene(poses)

        src_ids = (
            np.array(args.src_indices, dtype=int)
            if args.src_indices is not None
            else np.arange(args.src_offset, args.src_offset + args.N_views_src)
        )
        if np.any(src_ids >= len(poses)):
            raise ValueError(f"{scene_path}: source view indices exceed the scene view count")
        print(f"src_ids: {src_ids}")

        images_src = images[src_ids].to(device)
        poses_src = poses[src_ids].to(device)
        intrs_src = intrs[src_ids].to(device)

        # --- VIDEO GENERATION SETUP ---
        CHUNK_SIZE = args.N_views_tgt
    

        path_ids = args.path_indices if args.path_indices is not None else list(range(len(poses)))
        if len(path_ids) < 2 or any(i < 0 or i >= len(poses) for i in path_ids):
            raise ValueError(f"{scene_path}: path_indices must contain at least two valid scene view indices")
        video_poses = interpolate_trajectory(
            poses[path_ids], frames_per_segment=args.frames_per_segment, closed_loop=False
        ).to(device)
        # video_poses = poses.to(device)
    
        NUM_VIDEO_FRAMES=video_poses.shape[0]

        # We use the same intrinsics as the source image for all target frames
        video_intrs = intrs_src[0].unsqueeze(0).to(device) 
    
        video_frames = []
        video_dir = osp.join(scene_out_dir, "video_frames")
        os.makedirs(video_dir, exist_ok=True)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            with torch.no_grad():
                for frame_idx in range(0, NUM_VIDEO_FRAMES, CHUNK_SIZE):

                    end_idx = min(frame_idx + CHUNK_SIZE, NUM_VIDEO_FRAMES)
                    current_chunk_size = end_idx - frame_idx
                
                    print(f"Rendering frames {frame_idx+1} to {end_idx}/{NUM_VIDEO_FRAMES} concurrently...")
                
                    # Forcefully clear the cache to prevent cross-chunk bleeding
                    reset_model_cache(infinity)
                
                    # Slice the poses for the current chunk and add batch dimension [1, chunk_size, 4, 4]
                    curr_pose_tgt = video_poses[frame_idx : end_idx] #.unsqueeze(0) 
                
                    # Duplicate the intrinsics for the target chunk [1, chunk_size, 3, 3]
                    curr_intrs_tgt = video_intrs.repeat(current_chunk_size, 1, 1).unsqueeze(0)
                    curr_pose_src = poses_src

                    # Generate ONE target view
                    generated_chunk = gen_one_img(
                        images_src,
                        curr_pose_src[None],
                        curr_pose_tgt[None],
                        intrs_src[None],
                        curr_intrs_tgt,
                        infinity,
                        vae,
                        g_seed=args.seed,
                        gt_leak=-1,
                        gt_ls_Bl=None,
                        cfg_list=args.cfg,
                        tau_list=args.tau,
                        scale_schedule=scale_schedule,
                        cfg_insertion_layer=[args.cfg_insertion_layer],
                        vae_type=args.vae_type,
                        sampling_per_bits=args.sampling_per_bits,
                        enable_positive_prompt=args.enable_positive_prompt,
                    )

                    print(f"generated_chunk.shape: {generated_chunk.shape}")

                    for i in range(current_chunk_size):
                        # ---> FIX: Removed the '0,' so it correctly grabs the i-th image <---
                        frame_np = generated_chunk[i].detach().cpu().numpy() 
                        frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)
                    
                        global_frame_idx = frame_idx + i
                        save_path = osp.join(video_dir, f"frame_{global_frame_idx:03d}.png")
                        Image.fromarray(frame_np).save(save_path)
                        video_frames.append(frame_np)

                        # --- Save Pose ---
                        # Extract the i-th pose from the chunk, move to CPU, and save as numpy array
                        pose_np = curr_pose_tgt[i].float().detach().cpu().numpy()
                        pose_save_path = osp.join(video_dir, f"frame_{global_frame_idx:03d}.npy")
                        np.save(pose_save_path, pose_np)

        print(f"Generated {len(video_frames)} frames")

        video_out_path = osp.join(scene_out_dir, "orbit_flythrough.mp4")
        imageio.mimwrite(
            video_out_path, 
            video_frames, 
            format='FFMPEG',      # <--- Forces it to ignore the TIFF writer
            fps=args.fps, 
            codec='libx264', 
            macro_block_size=None, 
            ffmpeg_params=['-pix_fmt', 'yuv420p']
        )
        print(f"Video saved to {video_out_path}")

