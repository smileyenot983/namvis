

from infinity.models.basic import GLOBAL_SAVED_CROSS_ATTN, GLOBAL_SAVED_SELF_ATTN

import cv2
import os.path as osp
import numpy as np

# 001a994fe20b48739e18cf8a4e9d6fa7
POINTS_TO_TEST = [
        (0,128, 155), # Center
        (0,91, 117),  # Top middle
    ]

# Threshold_Porcelain_Teapot_White
POINTS_TO_TEST = [
        (0, 105, 199), # Center
        (0, 99, 64),  # Top middle
    ]

def visualize_single_layer_all_scales_self_attn(generated_images, args, layer_idx):
    """
    Visualizes self-attention for a *specific* layer and saves the RGB query 
    image and the target attention maps as separate files.
    """
    if len(GLOBAL_SAVED_SELF_ATTN) == 0:
        print("No self-attention maps found in global list.")
        return

    scale_schedule = [1, 2, 4, 6, 8, 12, 16]

    # Find all unique sequence lengths present in the saved self-attention maps
    unique_seq_lens = sorted(list(set(t.shape[2] for t in GLOBAL_SAVED_SELF_ATTN)))
    print(f"Detected actual sequence lengths in model: {unique_seq_lens}")
    
    # Map the detected sequence lengths directly to the scales
    if len(unique_seq_lens) < len(scale_schedule):
        print("Warning: Found fewer sequence lengths than expected scales.")
        valid_seq_lens = unique_seq_lens
        valid_scales = scale_schedule[:len(unique_seq_lens)]
    else:
        valid_seq_lens = unique_seq_lens[-len(scale_schedule):]
        valid_scales = scale_schedule

    scale_info = {}
    scale_to_attns = {}
    
    for s, seq_len in zip(valid_scales, valid_seq_lens):
        tokens_this_scale = (s * s) * args.N_views_tgt
        early_offset = seq_len - tokens_this_scale
        
        scale_info[s] = {
            'seq_len': seq_len,
            'early_offset': early_offset,
            'tokens_this_scale': tokens_this_scale
        }
        # Grab all layers for this specific scale's sequence length
        scale_to_attns[s] = [t for t in GLOBAL_SAVED_SELF_ATTN if t.shape[2] == seq_len]

    print(f"Successfully mapped {len(valid_scales)} scales.")

    # Prepare ALL Target Images
    gen_imgs_np = []
    for i in range(args.N_views_tgt):
        img = generated_images[i].cpu().numpy().astype(np.uint8)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        gen_imgs_np.append(img_bgr)
    
    gen_img_0_bgr = gen_imgs_np[0]

    points_to_test = POINTS_TO_TEST

    for (_, target_pixel_x, target_pixel_y) in points_to_test:
        
        # --- 1. SAVE THE SEPARATE MARKED RGB IMAGE ---
        gen_img_marked = gen_img_0_bgr.copy()
        cv2.circle(gen_img_marked, (target_pixel_x, target_pixel_y), 5, (0, 255, 0), -1)
        cv2.circle(gen_img_marked, (target_pixel_x, target_pixel_y), 6, (255, 255, 255), 1)
        
        rgb_out_name = osp.join(args.out_dir, f"query_rgb_x{target_pixel_x}_y{target_pixel_y}.png")
        cv2.imwrite(rgb_out_name, gen_img_marked)
        # ---------------------------------------------
        
        for s in valid_scales:
            attns_for_scale = scale_to_attns[s]
            num_layers = len(attns_for_scale)
            
            # Ensure the requested layer exists for this scale
            if layer_idx >= num_layers:
                print(f"Warning: layer_idx {layer_idx} out of bounds for scale {s} (max {num_layers-1}). Skipping.")
                continue
                
            info = scale_info[s]
            early_offset = info['early_offset']
            tokens_this_scale = info['tokens_this_scale']
            
            # MAP PIXEL TO THIS SPECIFIC SCALE
            token_x = (target_pixel_x * s) // 256
            token_y = (target_pixel_y * s) // 256
            
            # Querying View 0 at the current scale
            spatial_idx = (token_y * s) + token_x
            target_token_idx = early_offset + spatial_idx
            
            # --- 2. EXTRACT ONLY THE SPECIFIED LAYER ---
            layer_attn = attns_for_scale[layer_idx][0] # Shape: [Heads, SeqLen, SeqLen]
            
            # EXTRACT ONLY CURRENT SCALE KEYS (Ignore the history tokens)
            current_scale_attn_all_heads = layer_attn[:, target_token_idx, -tokens_this_scale:]
            
            # TAKE THE MAX ACROSS HEADS
            full_tgt_attn = current_scale_attn_all_heads.max(dim=0).values
            
            blended_tgt_views = []
            for v in range(args.N_views_tgt):
                # Slice exactly the tokens for View 'v' at the current scale
                start_idx = v * (s * s)
                end_idx = start_idx + (s * s)
                
                tgt_attn_1d = full_tgt_attn[start_idx:end_idx]
                
                # Perfect S x S reshape
                tgt_attn_2d = tgt_attn_1d.reshape(s, s).cpu().numpy()
                
                val_min = tgt_attn_2d.min()
                val_max = np.percentile(tgt_attn_2d, 99.5) 
                
                if val_max > val_min:
                    tgt_attn_2d = np.clip(tgt_attn_2d, val_min, val_max)
                    tgt_attn_2d = (tgt_attn_2d - val_min) / (val_max - val_min)
                    
                tgt_attn_2d = np.uint8(255 * tgt_attn_2d)
                
                heatmap_resized = cv2.resize(tgt_attn_2d, (256, 256), interpolation=cv2.INTER_CUBIC)
                heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
                
                blended = cv2.addWeighted(gen_imgs_np[v], 0.5, heatmap_colored, 0.5, 0)
                blended_tgt_views.append(blended)
            
            # Combine the target views for this specific layer side-by-side
            layer_strip = np.hstack(blended_tgt_views)
            
            # --- 3. SAVE THE ATTENTION MAP SEPARATELY ---
            out_name = osp.join(args.out_dir, f"self_attn_scale_{s}x{s}_layer_{layer_idx}_x{target_pixel_x}_y{target_pixel_y}.png")
            cv2.imwrite(out_name, layer_strip)
            
        print(f"Saved RGB and Layer {layer_idx} Self-Attention maps for point ({target_pixel_x}, {target_pixel_y}).")



def safe_tensor_to_cv2(tensor_img):
    """Safely converts any PyTorch image tensor to a valid OpenCV BGR image."""
    img = tensor_img.float().cpu().numpy()
    
    # 1. Fix shape if it is [C, H, W] instead of [H, W, C]
    if img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
        
    # 2. Fix numeric range to [0, 255]
    if img.max() <= 1.0:
        if img.min() < 0:
            img = (img + 1.0) / 2.0  # Convert [-1, 1] to [0, 1]
        img = img * 255.0            # Convert [0, 1] to [0, 255]
        
    # 3. Cast to uint8 and convert to BGR for OpenCV
    img = img.astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def visualize_single_layer_all_scales_cross_attn(images_src, generated_images, args, layer_idx):
    """
    Visualizes cross-attention maps for a *specific* layer.
    Saves the RGB query image and the target attention maps as separate files.
    """
    if len(GLOBAL_SAVED_CROSS_ATTN) == 0:
        print("No cross attention maps found in global list.")
        return

    scale_schedule = [1, 2, 4, 6, 8, 12, 16]
    
    # Group the saved attentions by their target token length
    scale_to_attns = {}
    for s in scale_schedule:
        expected_tokens = (s * s) * args.N_views_tgt
        # USE -2 FOR THE QUERY DIMENSION (Works for 3D or 4D tensors)
        scale_to_attns[s] = [t for t in GLOBAL_SAVED_CROSS_ATTN if t.shape[-2] == expected_tokens]
        
    valid_scales = [s for s in scale_schedule if len(scale_to_attns[s]) > 0]
    
    if len(valid_scales) == 0:
        unique_shapes = set([tuple(t.shape) for t in GLOBAL_SAVED_CROSS_ATTN])
        print(f"CRITICAL DEBUG: Found 0 scales. Available tensor shapes in GLOBAL_SAVED_CROSS_ATTN: {unique_shapes}")
        return
        
    print(f"Detected {len(valid_scales)} scales in the generation process.")

    # --- Prepare Source Images ---
    src_imgs_np = []
    for i in range(args.N_views_src):
        src_imgs_np.append(safe_tensor_to_cv2(images_src[i]))
    
    points_to_test = POINTS_TO_TEST

    # Filter out points if their view_idx exceeds available generated views
    points_to_test = [p for p in points_to_test if p[0] < args.N_views_tgt]

    for (target_view_idx, target_pixel_x, target_pixel_y) in points_to_test:
        
        gen_img_v_bgr = safe_tensor_to_cv2(generated_images[target_view_idx])

        # --- 1. SAVE THE SEPARATE MARKED RGB QUERY IMAGE ---
        gen_img_marked = gen_img_v_bgr.copy()
        cv2.circle(gen_img_marked, (target_pixel_x, target_pixel_y), 5, (0, 255, 0), -1)
        cv2.circle(gen_img_marked, (target_pixel_x, target_pixel_y), 6, (255, 255, 255), 1)
        
        rgb_out_name = osp.join(args.out_dir, f"query_rgb_v{target_view_idx}_x{target_pixel_x}_y{target_pixel_y}.png")
        cv2.imwrite(rgb_out_name, gen_img_marked)
        # ---------------------------------------------------
        
        for s in valid_scales:
            attns_for_scale = scale_to_attns[s]
            num_layers = len(attns_for_scale)
            
            # Ensure the requested layer exists for this scale
            if layer_idx >= num_layers:
                print(f"Warning: layer_idx {layer_idx} out of bounds for scale {s} (max {num_layers-1}). Skipping.")
                continue
            
            # --- Map Pixel to Token Index ---
            # 1. Base spatial token coordinates for this scale
            token_x = (target_pixel_x * s) // 256
            token_y = (target_pixel_y * s) // 256
            
            # 2. Scale-major view offset
            view_offset = target_view_idx * (s * s)
            
            # 3. Final sequence index for the Query
            target_token_idx = view_offset + (token_y * s) + token_x 
            
            # --- 2. EXTRACT ONLY THE SPECIFIED LAYER ---
            # Shape: [Heads, S*S*N_tgt, 256 * N_src]
            layer_attn = attns_for_scale[layer_idx][0] 
            
            # Take max across heads for sharpest geometric activation
            full_source_attn = layer_attn[:, target_token_idx, :].max(dim=0).values.cpu().numpy()
            
            blended_src_views = []
            for v in range(args.N_views_src):
                # Source features are fixed at 16x16 (256 tokens) per view
                start_idx = v * 256
                end_idx = start_idx + 256
                
                src_attn_1d = full_source_attn[start_idx:end_idx]
                src_attn_2d = src_attn_1d.reshape(16, 16)
                
                # Normalize heatmap for visualization
                val_min = src_attn_2d.min()
                val_max = np.percentile(src_attn_2d, 99.5) 
                
                if val_max > val_min:
                    src_attn_2d = np.clip(src_attn_2d, val_min, val_max)
                    src_attn_2d = (src_attn_2d - val_min) / (val_max - val_min)
                    
                src_attn_2d = np.uint8(255 * src_attn_2d)
                
                # Resize and apply colormap
                heatmap_resized = cv2.resize(src_attn_2d, (256, 256), interpolation=cv2.INTER_CUBIC)
                heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
                
                # Blend with corresponding source image
                blended = cv2.addWeighted(src_imgs_np[v], 0.5, heatmap_colored, 0.5, 0)
                blended_src_views.append(blended)
            
            # Combine all source views horizontally for this single layer
            layer_strip = np.hstack(blended_src_views)
            
            # --- 3. SAVE THE ATTENTION MAP SEPARATELY ---
            out_name = osp.join(args.out_dir, f"cross_attn_v{target_view_idx}_scale_{s}x{s}_layer_{layer_idx}_x{target_pixel_x}_y{target_pixel_y}.png")
            cv2.imwrite(out_name, layer_strip)
            
        print(f"Saved RGB and Layer {layer_idx} Cross-Attention maps for View {target_view_idx}, point ({target_pixel_x}, {target_pixel_y}).")
