from __future__ import annotations

import torch
import torch.nn.functional as F


def build_origin_direction_raymap(
    poses: torch.Tensor,
    intrs: torch.Tensor,
    height: int,
    width: int,
    frame: str = "modelspace",
    pixel_center: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Build [origin_xyz, direction_xyz] raymaps in modelspace or camera-local frame."""
    if frame not in {"modelspace", "target_local"}:
        raise ValueError(f"Unsupported ray VAE frame: {frame}")
    if poses.ndim == 2:
        poses = poses.unsqueeze(0)
    if intrs.ndim == 2:
        intrs = intrs.unsqueeze(0)

    device = poses.device
    dtype = poses.dtype
    intrs = intrs.to(device=device, dtype=dtype).clone()
    # Some Infinity3D loaders store intrinsics normalized to image width/height.
    # Convert those to pixel units before applying K^-1 to pixel coordinates.
    if torch.nan_to_num(intrs[:, 0, 2]).abs().max() <= 2.0 and torch.nan_to_num(intrs[:, 0, 0]).abs().max() <= 10.0:
        intrs[:, 0, 0] *= float(width)
        intrs[:, 0, 2] *= float(width)
        intrs[:, 1, 1] *= float(height)
        intrs[:, 1, 2] *= float(height)
    n = max(poses.shape[0], intrs.shape[0])
    if poses.shape[0] == 1 and n > 1:
        poses = poses.expand(n, -1, -1)
    if intrs.shape[0] == 1 and n > 1:
        intrs = intrs.expand(n, -1, -1)

    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    if pixel_center:
        x = x + 0.5
        y = y + 0.5
    pix = torch.stack([x, y, torch.ones_like(x)], dim=-1).reshape(-1, 3)
    inv_k = torch.linalg.inv(intrs)
    dirs_cam = torch.matmul(inv_k, pix.t().unsqueeze(0)).transpose(1, 2)
    dirs_cam = F.normalize(dirs_cam, dim=-1, eps=eps)

    if frame == "target_local":
        origins = torch.zeros((n, 3), device=device, dtype=dtype)
        dirs = dirs_cam
    else:
        rot = poses[:, :3, :3]
        origins = poses[:, :3, 3]
        dirs = torch.matmul(rot, dirs_cam.transpose(1, 2)).transpose(1, 2)
        dirs = F.normalize(dirs, dim=-1, eps=eps)

    origins = origins[:, None, :].expand(-1, height * width, -1)
    raymap = torch.cat([origins, dirs], dim=-1)
    return raymap.reshape(n, height, width, 6).permute(0, 3, 1, 2).contiguous()


def modelspace_pointmaps_to_target_local(pointmaps: torch.Tensor, poses: torch.Tensor) -> torch.Tensor:
    if pointmaps.ndim == 4:
        pointmaps = pointmaps.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    if poses.ndim == 3:
        poses = poses.unsqueeze(0)
    b, n, c, h, w = pointmaps.shape
    xyz = pointmaps[:, :, :3].permute(0, 1, 3, 4, 2).reshape(b, n, h * w, 3)
    poses = poses.to(device=pointmaps.device, dtype=pointmaps.dtype)
    rot = poses[:, :, :3, :3]
    trans = poses[:, :, :3, 3]
    xyz_local = torch.matmul(rot.transpose(-1, -2), (xyz - trans[:, :, None]).unsqueeze(-1)).squeeze(-1)
    xyz_local = xyz_local.reshape(b, n, h, w, 3).permute(0, 1, 4, 2, 3)
    out = torch.cat([xyz_local, pointmaps[:, :, 3:]], dim=2) if c > 3 else xyz_local
    return out.squeeze(0) if squeeze else out


def target_local_pointmaps_to_modelspace(pointmaps: torch.Tensor, poses: torch.Tensor) -> torch.Tensor:
    if pointmaps.ndim == 4:
        pointmaps = pointmaps.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    if poses.ndim == 3:
        poses = poses.unsqueeze(0)
    b, n, c, h, w = pointmaps.shape
    xyz = pointmaps[:, :, :3].permute(0, 1, 3, 4, 2).reshape(b, n, h * w, 3)
    poses = poses.to(device=pointmaps.device, dtype=pointmaps.dtype)
    rot = poses[:, :, :3, :3]
    trans = poses[:, :, :3, 3]
    xyz_model = torch.matmul(rot, xyz.unsqueeze(-1)).squeeze(-1) + trans[:, :, None]
    xyz_model = xyz_model.reshape(b, n, h, w, 3).permute(0, 1, 4, 2, 3)
    out = torch.cat([xyz_model, pointmaps[:, :, 3:]], dim=2) if c > 3 else xyz_model
    return out.squeeze(0) if squeeze else out
