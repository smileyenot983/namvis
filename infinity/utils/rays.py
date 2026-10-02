

from typing import Tuple
import torch

def plucker_rays_batched(pose: torch.Tensor,
                         K: torch.Tensor,
                         H: int,
                         W: int,
                         *,
                         pixel_center: bool = True,
                         flip_y: bool = False,
                         eps: float = 1e-9
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Batched computation of Plücker coordinates for pixel rays.

    Args:
        R: (B,3,3) or (3,3) rotation(s) camera -> world.
        t: (B,3) or (3,) camera origin(s) in world coords.
        K: (B,3,3) or (3,3) intrinsic matrix/matrices.
        H: image height (pixels).
        W: image width (pixels).
        pixel_center: if True use pixel centers (u+0.5, v+0.5).
        flip_y: if True flip vertical axis (useful for top-left image origin).
        eps: small value for numeric stability in normalization.

    Returns:
        plucker_coords: (B, H, W, 6) Plücker [moment (3), direction (3)] in world frame.
        uv_world:       (B, H, W, 3) points on image plane z=1 mapped to world (o + R @ (K_inv @ [u,v,1])).
        d_cam:          (B, H, W, 3) normalized ray directions in camera coordinates.
        o_world:        (B, 3) camera origins in world coordinates.
    """

    # print(f"pose: {pose}")
    # print(f"K: {K}")
    # print(f"H: {H} | W: {W}")


    R = pose[:,:3,:3]
    t = pose[:,:3,3]

    # Normalize inputs to have a batch dimension
    if R.ndim == 2:
        R = R.unsqueeze(0)  # (1,3,3)
    if K.ndim == 2:
        K = K.unsqueeze(0)  # (1,3,3)
    if t.ndim == 1:
        t = t.unsqueeze(0)  # (1,3)

    # Determine batch size and broadcast singletons
    B = max(R.shape[0], K.shape[0], t.shape[0])
    def _maybe_broadcast(x, name, target=B):
        if x.shape[0] == target:
            return x
        if x.shape[0] == 1:
            return x.expand(target, *x.shape[1:])
        raise ValueError(f"Batch size mismatch for {name}: got {x.shape[0]}, expected 1 or {target}")

    R = _maybe_broadcast(R, 'R')
    K = _maybe_broadcast(K, 'K')
    t = _maybe_broadcast(t, 't')

    device = K.device
    dtype = K.dtype

    # Inverse intrinsics (batched)
    K_inv = torch.linalg.inv(K)  # (B,3,3)

    # Create image grid (u: width axis, v: height axis) -> shape (H, W)
    u, v = torch.meshgrid(
        torch.arange(W, dtype=dtype, device=device),
        torch.arange(H, dtype=dtype, device=device),
        indexing='xy'
    )
    # meshgrid with indexing='xy' returns u,v shaped (H, W) as used below
    if pixel_center:
        u = u + 0.5
        v = v + 0.5
    if flip_y:
        v = (H - 1) - v

    ones = torch.ones_like(u)
    pix = torch.stack([u, v, ones], dim=-1)        # (H, W, 3)

    # Broadcast pix to batch: (B, H, W, 3, 1) for matmul
    pix_h = pix.unsqueeze(0).unsqueeze(-1)        # (1, H, W, 3, 1)
    K_inv_b = K_inv[:, None, None, ...]           # (B, 1, 1, 3, 3)

    # print(f"pix_h.shape: {pix_h.shape} | K_inv_b.shape: {K_inv_b.shape}")

    # Map pixels to camera coords at z=1: pix_cam = K_inv @ [u, v, 1]^T
    pix_cam = (K_inv_b @ pix_h).squeeze(-1)      # (B, H, W, 3)

    # Ray directions in camera frame (pre-normalized)
    d_cam = pix_cam                                # (B, H, W, 3)
    d_cam_norm = d_cam / (torch.linalg.norm(d_cam, dim=-1, keepdim=True) + eps)

    # Rotate directions to world (no translation for direction vectors)
    R_b = R[:, None, None, ...]                    # (B, 1, 1, 3, 3)
    d_world = (R_b @ d_cam.unsqueeze(-1)).squeeze(-1)  # (B, H, W, 3)
    d_world_norm = d_world / (torch.linalg.norm(d_world, dim=-1, keepdim=True) + eps)

    # Camera origins in world coords
    o_world = t.view(B, 3)                         # (B, 3)

    # uv_world: points on z=1 in world coords: o + R @ pix_cam
    uv_world = (R_b @ pix_cam.unsqueeze(-1)).squeeze(-1) + o_world[:, None, None, :]  # (B, H, W, 3)

    # Plücker moment: m = o x d (use world origin o and world direction d_world_norm)
    o_exp = o_world[:, None, None, :].expand_as(d_world_norm)  # (B, H, W, 3)
    m = torch.cross(o_exp, d_world_norm, dim=-1)               # (B, H, W, 3)

    # Concatenate moment and direction -> (B, H, W, 6)
    plucker = torch.cat([m, d_world_norm], dim=-1)

    return plucker, uv_world, d_cam_norm, o_world


def to_hom(X):
    # get homogeneous coordinates of the input
    X_hom = torch.cat([X, torch.ones_like(X[..., :1])], dim=-1)
    return X_hom


def to_hom_pose(pose):
    # get homogeneous coordinates of the input pose
    if pose.shape[-2:] == (3, 4):
        pose_hom = torch.eye(4, device=pose.device)[None].repeat(pose.shape[0], 1, 1)
        pose_hom[:, :3, :] = pose
        return pose_hom
    return pose

def get_image_grid(img_h, img_w):
    # add 0.5 is VERY important especially when your img_h and img_w
    # is not very large (e.g., 72)!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    y_range = torch.arange(img_h, dtype=torch.float32).add_(0.5)
    x_range = torch.arange(img_w, dtype=torch.float32).add_(0.5)
    Y, X = torch.meshgrid(y_range, x_range, indexing="ij")  # [H,W]
    xy_grid = torch.stack([X, Y], dim=-1).view(-1, 2)  # [HW,2]
    return to_hom(xy_grid)  # [HW,3]


def img2cam(X, cam_intr):
    # print(f"X.shape: {X.shape}")
    # print(f"cam_intr.shape: {cam_intr.shape}")
    return X @ cam_intr.inverse().transpose(-1, -2)


def cam2world(X, pose):
    X_hom = to_hom(X)
    pose_inv = torch.linalg.inv(to_hom_pose(pose))[..., :3, :4]
    return X_hom @ pose_inv.transpose(-1, -2)


def get_center_and_ray(img_h, img_w, pose, intr):  # [HW,2]
    # given the intrinsic/extrinsic matrices, get the camera center and ray directions]
    # assert(opt.camera.model=="perspective")
    B = pose.shape[0]

    # compute center and ray
    grid_img = get_image_grid(img_h, img_w)  # [HW,3]
    # print(f"grid_img.shape: {grid_img.shape}")
    grid_img = grid_img.unsqueeze(0).expand(B,-1,-1)
    # print(f"[1]grid_img.shape: {grid_img.shape}")

    grid_3D_cam = img2cam(grid_img.to(intr.device), intr.float())  # [B,HW,3]
    center_3D_cam = torch.zeros_like(grid_3D_cam)  # [B,HW,3]

    # print(f"pose.shape: {pose.shape}")
    # print(f"center_3D_cam.shape: {center_3D_cam.shape}")

    # transform from camera to world coordinates
    grid_3D = cam2world(grid_3D_cam, pose)  # [B,HW,3]
    center_3D = cam2world(center_3D_cam, pose)  # [B,HW,3]
    ray = grid_3D - center_3D  # [B,HW,3]

    return center_3D, ray, grid_3D_cam


def plucker_rays_seva(
    extrinsics_src, #c2w format
    extrinsics, #c2w format
    intrinsics=None,
    target_size=[72, 72],
    real_size=[72,72]
):
    # print(f"real_size: {real_size}")
    
    # intrinsics_norm = torch.zeros_like(intrinsics)
    # intrinsics_norm[..., 0, 0] = intrinsics[..., 0, 0] / real_size[1]
    # intrinsics_norm[..., 1, 1] = intrinsics[..., 1, 1] / real_size[0]
    # intrinsics_norm[..., 0, 2] = intrinsics[..., 0, 2] / real_size[1] #- 0.5
    # intrinsics_norm[..., 1, 2] = intrinsics[..., 1, 2] / real_size[0] #- 0.5
    # intrinsics_norm[..., 2, 2] = 1.0
    # print(f"intrinsics: {intrinsics}")
    # print(f"intrinsics_norm: {intrinsics_norm}")
    # print(f"target_size: {target_size}")
    # print(f"real_size: {real_size}")

    # check that intrinsics were normalized
    # if not (
    #     torch.all(intrinsics_norm[:, :2, -1] >= 0)
    #     and torch.all(intrinsics_norm[:, :2, -1] <= 1)
    # ):
    #     intrinsics_norm[:, :2] /= intrinsics.new_tensor(target_size).view(1, -1, 1) * 8
    # you should ensure the intrisics are expressed in
    # resolution-independent normalized image coordinates just performing a
    # very simple verification here checking if principal points are
    # between 0 and 1
    # assert (
    #     torch.all(intrinsics[:, :2, -1] >= 0)
    #     and torch.all(intrinsics[:, :2, -1] <= 1)
    # ), "Intrinsics should be expressed in resolution-independent normalized image coordinates."

    # c2w_src = torch.linalg.inv(extrinsics_src)
    # transform coordinates from the source camera's coordinate system to the coordinate system of the respective camera
    extrinsics_rel = torch.einsum(
        "vnm,vmp->vnp", extrinsics, c2w_src[None].repeat(extrinsics.shape[0], 1, 1)
    )

    # print(f"[seva0] intrinsics: {intrinsics}")


    intrinsics[:, :2] *= extrinsics.new_tensor(
        [
            real_size[1],  # w
            real_size[0],  # h
        ]
    ).view(1, -1, 1)
    # print(f"[seva1] intrinsics: {intrinsics}")
    centers, rays, grid_cam = get_center_and_ray(
        img_h=target_size[0],
        img_w=target_size[1],
        pose=extrinsics_rel[:, :3, :],
        intr=intrinsics,
    )
    
    rays = torch.nn.functional.normalize(rays, dim=-1)
    plucker = torch.cat((rays, torch.cross(centers, rays, dim=-1)), dim=-1)
    b, hw, _ = plucker.shape
    # print(f"[seva] plucker.shape: {plucker.shape}")
    # print(f"[seva] target_size: {target_size}")
    plucker = plucker.reshape(b, *target_size, -1)
    # plucker = plucker.permute(0, 2, 1).reshape(plucker.shape[0], -1, *target_size)
    return plucker


import torch

def plucker_raymap_torch(c2w, K, height, width,
                         normalize_d=True,
                         squeeze_batch=False,
                         eps=1e-12):
    """
    Compute Plücker coordinates (d, m) for image rays using PyTorch only.

    Args
    ----
    c2w : torch.Tensor, shape (4,4) or (N,4,4)
        Camera-to-world transforms. If assume_c2w==False, they are treated as w2c and inverted.
    K : torch.Tensor, shape (3,3) or (N,3,3)
        Intrinsics matrix (fx, 0, cx; 0, fy, cy; 0,0,1) or batched.
    height, width : int
        Image size in pixels (H, W).
    normalize_d : bool
        Normalize ray directions to unit length (default True).
    assume_c2w : bool
        If False, treat `c2w` as w2c and invert it.
    squeeze_batch : bool
        If True and input was a single camera, return (H, W, 6) instead of (1, H, W, 6).
    eps : float
        Small epsilon to avoid division by zero.

    Returns
    -------
    rays_plucker : torch.Tensor, shape (N, H, W, 6) (or (H, W, 6) if squeezed)
        Last-dim: [d_x, d_y, d_z, m_x, m_y, m_z], where m = p x d and p is camera center in world coords.
    """

    if not torch.is_tensor(c2w):
        c2w = torch.tensor(c2w)
    if not torch.is_tensor(K):
        K = torch.tensor(K)

    # convert to float and put on same device
    device = c2w.device
    dtype = torch.float32
    c2w = c2w.to(dtype=dtype, device=device)
    K = K.to(dtype=dtype, device=device)

    # ensure batch dims
    if c2w.ndim == 2:
        c2w = c2w.unsqueeze(0)  # (1,4,4)
    if K.ndim == 2:
        K = K.unsqueeze(0)      # (1,3,3)

    N = c2w.shape[0]

    # broadcast K if needed
    if K.shape[0] == 1 and N > 1:
        K = K.expand(N, -1, -1)


    # origins and rotations
    origins = c2w[:, :3, 3]    # (N,3)
    Rs = c2w[:, :3, :3]        # (N,3,3)

    # build pixel grid (u=x, v=y) with origin at top-left, u->right, v->down
    # Using indexing='ij' for (H,W) shaped outputs
    grid_y, grid_x = torch.meshgrid(torch.arange(height, device=device, dtype=dtype),
                                    torch.arange(width,  device=device, dtype=dtype),
                                    indexing='ij')  # both (H,W)
    ones = torch.ones_like(grid_x)
    # pixel coordinates (u, v, 1)
    pix = torch.stack([grid_x, grid_y, ones], dim=-1)  # (H, W, 3)
    H, W = height, width
    HW = H * W

    # flatten to (3, HW)
    pix_flat = pix.reshape(-1, 3).transpose(0, 1)  # (3, HW)
    pix_flat = pix_flat.to(dtype=dtype, device=device)

    # inverse intrinsics per camera (N,3,3)
    invK = torch.linalg.inv(K)  # (N,3,3)

    # compute camera-space directions: d_cam = invK @ pix_flat, compute for all cameras at once
    # pix_flat unsqueezed to (1,3,HW) will broadcast with (N,3,3)
    pix_flat_exp = pix_flat.unsqueeze(0)  # (1,3,HW)
    # batch matmul: (N,3,3) @ (1,3,HW) -> (N,3,HW) by broadcasting
    d_cam = torch.matmul(invK, pix_flat_exp)  # (N,3,HW)

    # normalize directions if requested
    if normalize_d:
        norms = torch.linalg.norm(d_cam, dim=1, keepdim=True)  # (N,1,HW)
        d_cam = d_cam / (norms + eps)
    # rotate to world: d_world = R @ d_cam  -> (N,3,HW)
    d_world = torch.matmul(Rs, d_cam)

    # compute moment: m = p x d  where p is origin in world coords
    p = origins.unsqueeze(-1)  # (N,3,1)
    # broadcast p to (N,3,HW) and compute cross product along vector dim=1
    p_rep = p.expand(-1, -1, HW)   # (N,3,HW)
    m_world = torch.cross(p_rep, d_world, dim=1)  # (N,3,HW)

    # reshape to (N, H, W, 3)
    d_world = d_world.permute(0, 2, 1).reshape(N, H, W, 3)
    m_world = m_world.permute(0, 2, 1).reshape(N, H, W, 3)

    # concat -> (N, H, W, 6)
    rays_plucker = torch.cat([d_world, m_world], dim=-1)

    if squeeze_batch and rays_plucker.shape[0] == 1:
        return rays_plucker[0]

    return rays_plucker


def normalized_intrinsics_plucker_raymap(c2w, intrinsics, height, width):
    """Return model-space ``[N, 6, H, W]`` rays from Infinity-normalized K.

    Infinity stores its post-crop intrinsics in normalized image coordinates.
    This helper converts a clone to pixel coordinates before using the same
    Plucker implementation used by geometry inference.  It never mutates the
    caller's intrinsics tensor.
    """
    K = torch.as_tensor(intrinsics).clone()
    if K.ndim == 2:
        K = K.unsqueeze(0)
    if K.ndim != 3 or K.shape[-2:] != (3, 3):
        raise ValueError(f"Expected normalized intrinsics [N,3,3], got {tuple(K.shape)}")
    K[:, 0, :] *= width
    K[:, 1, :] *= height
    rays = plucker_raymap_torch(c2w, K, height=height, width=width)
    return rays.permute(0, 3, 1, 2).contiguous()


def plucker_rays_paired(
    c2w_src, # [B, 1, 4, 4] c2w format, all rays computed wrt c2w_src
    c2ws, # :[B,N_views_tgt,4,4] C2W format
    intrinsics, # :[B,N_views,3,3] C2W format
    target_size=[72, 72], #h,w
    real_size=[72,72] #h,w
):
    """
    Calculates Plucker rays for 2nd view only!
    using relative pose between 1st and 2nd views
    """

    # rescale intrinsics fx,cx,fy,cy
    intrinsics[:,:,0,0] *= real_size[1]
    intrinsics[:,:,0,2] *= real_size[1]
    intrinsics[:,:,1,1] *= real_size[0]
    intrinsics[:,:,1,2] *= real_size[0]

    # calc relpose between tgt and src
    c2w_rel = torch.matmul(torch.linalg.inv(c2w_src), c2ws)
    # print(f"c2w_rel.shape: {c2w_rel.shape}")
    B,N_tgt = c2w_rel.shape[:2]

    c2w_rel = c2w_rel.reshape(B*N_tgt, *c2w_rel.shape[2:])
    intrinsics = intrinsics.reshape(B*N_tgt, *intrinsics.shape[2:])

    plucker = plucker_raymap_torch(c2w_rel,
                         intrinsics,
                         height=target_size[0],
                         width=target_size[1])
    plucker = plucker.reshape(B,N_tgt, *plucker.shape[1:])
    return plucker
