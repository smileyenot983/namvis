import torch
import torch.nn.functional as F
import lpips
import math

# --- 1. Global Setup (Run once) ---
# Initialize LPIPS globally to avoid reloading weights every iteration
# Ensure it's on the same device as your data (GPU)
loss_fn_alex = lpips.LPIPS(net='alex').eval()
if torch.cuda.is_available():
    loss_fn_alex = loss_fn_alex.cuda()

# --- 2. SSIM Helper Function ---
def calc_ssim_tensor(img1, img2, window_size=11):
    """
    Calculates SSIM for tensors in range [0, 1].
    Assumes shape [B, C, H, W]
    """
    channel = img1.size(1)
    
    # Create window
    def gaussian(window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous().to(img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()

# --- 3. Main Metrics Function ---
def calc_2D_metrics(pred_tensor, gt_tensor):
    """
    Args:
        pred_tensor: torch.Tensor, range [0, 255] (any shape [C,H,W] or [1,C,H,W])
        gt_tensor:   torch.Tensor, range [0, 255]
    Returns:
        loss (MSE), lpips, ssim, psnr
    """
    # 1. Ensure inputs are float and on correct device
    if pred_tensor.dtype != torch.float32:
        pred_tensor = pred_tensor.float()
    if gt_tensor.dtype != torch.float32:
        gt_tensor = gt_tensor.float()

    # 2. Handle Shapes: Ensure [1, C, H, W] for LPIPS/SSIM
    # If input is [C, H, W], add batch dimension
    if pred_tensor.ndim == 3:
        pred_tensor = pred_tensor.unsqueeze(0)
        gt_tensor = gt_tensor.unsqueeze(0)
        
    # Check if we need to permute [B, H, W, C] -> [B, C, H, W]
    # (Heuristic: if last dim is 3 and 2nd dim is not 3, it's likely HWC)
    if pred_tensor.shape[-1] == 3 and pred_tensor.shape[1] != 3:
        pred_tensor = pred_tensor.permute(0, 3, 1, 2)
        gt_tensor = gt_tensor.permute(0, 3, 1, 2)

    # 3. Create Normalized Versions
    # [0, 1] for SSIM and PSNR
    pred_0_1 = pred_tensor / 255.0
    gt_0_1 = gt_tensor / 255.0
    
    # [-1, 1] for LPIPS and MSE (Pixel Loss)
    pred_norm = (pred_tensor / 127.5) - 1.0
    gt_norm = (gt_tensor / 127.5) - 1.0

    with torch.no_grad():
        # --- MSE (Pixel Loss) ---
        # Calculated on [-1, 1] range as per your original code
        loss = F.mse_loss(pred_norm, gt_norm).item()

        # --- LPIPS ---
        # Expects [-1, 1]
        lpips_val = loss_fn_alex(pred_norm, gt_norm).item()

        # --- SSIM ---
        # Expects [0, 1]
        ssim_val = calc_ssim_tensor(pred_0_1, gt_0_1)

        # --- PSNR ---
        # Expects [0, 1] (Max val = 1.0)
        mse_0_1 = F.mse_loss(pred_0_1, gt_0_1)
        if mse_0_1 == 0:
            psnr_val = 100.0 # Perfect match
        else:
            psnr_val = 20 * torch.log10(1.0 / torch.sqrt(mse_0_1)).item()

    return loss, lpips_val, ssim_val, psnr_val