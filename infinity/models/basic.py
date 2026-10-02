"""
Definitions of blocks of VAR transformer model.
"""

import math
import os
from functools import partial
from typing import Optional, Tuple, Union, Callable, List
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import DropPath, drop_path
from torch.utils.checkpoint import checkpoint

# Import flash_attn's attention
from flash_attn import flash_attn_func                  # q, k, or v: BLHc, ret: BLHc
from flash_attn import flash_attn_varlen_kvpacked_func  # qkv: N3Hc, ret: NHc
try:
    from flash_attn import flash_attn_varlen_func       # q, k, v: total_tokens,H,c
except ImportError:
    flash_attn_varlen_func = None

from torch.nn.functional import scaled_dot_product_attention as slow_attn    # q, k, v: BHLc
from infinity.models.flex_attn import (
    SWITTI_FLEX_KERNEL_OPTIONS,
    create_switti_block_mask,
    flex_attention_available,
    get_compiled_flex_attention,
)

# from prope.torch import prope_dot_product_attention,\
#                         _prepare_apply_fns,\
#                         _rope_precompute_coeffs
# Import flash_attn's fused ops

try:
    from flash_attn.ops.fused_dense import fused_mlp_func
    flash_fused_op_installed = True
except ImportError:
    flash_fused_op_installed = False
    fused_mlp_func = None

try:
    from flash_attn.ops.rms_norm import rms_norm as rms_norm_impl
except ImportError:
    def rms_norm_impl(x, weight, epsilon):
        return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True).add_(epsilon))) * weight 


# try:
#     # from flash_attn.ops.layer_norm import dropout_add_layer_norm ##not used
#     # from flash_attn.ops.rms_norm import dropout_add_rms_norm ##not used
#     from flash_attn.ops.rms_norm import rms_norm as rms_norm_impl
#     from flash_attn.ops.fused_dense import fused_mlp_func
#     flash_fused_op_installed = True
# except ImportError:
#     dropout_add_layer_norm = dropout_add_rms_norm = fused_mlp_func = None
#     flash_fused_op_installed = False
    
#     def rms_norm_impl(x, weight, epsilon):
#         return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True).add_(epsilon))) * weight

import time

GLOBAL_SAVED_CROSS_ATTN = []
GLOBAL_SAVED_SELF_ATTN = []

MAX_DEPTH = 100.0
MAX_LOG_DEPTH = 3.0
MAX_ASINH_DEPTH = math.asinh(MAX_DEPTH)

MAX_D_F = 10.0
MAX_ASINH_D_F = math.asinh(MAX_D_F)

def precompute_rope2d_freqs_grid(dim, 
                                 dynamic_resolution_h_w, 
                                 rope2d_normalized_by_hw, 
                                 pad_to_multiplier=1, 
                                 max_height=2048 // 16, 
                                 max_width=2048 // 16, 
                                 base=10000.0, 
                                 N_views=1,
                                 device=None, scaling_factor=1.0):
    # split the dimension into half, one for x and one for y
    half_dim = dim // 2 #128/2=64
    inv_freq = 1.0 / (base ** (torch.arange(0, half_dim, 2, dtype=torch.int64).float().to(device) / half_dim)) # namely theta, 1 / (10000^(i/half_dim)), i=0,2,..., half_dim-2
    
    t_height = torch.arange(max_height, device=device, dtype=torch.int64).type_as(inv_freq)
    t_width = torch.arange(max_width, device=device, dtype=torch.int64).type_as(inv_freq)
    t_height = t_height / scaling_factor
    freqs_height = torch.outer(t_height, inv_freq)  # (max_height, dim / (1 for 1d, 2 for 2d, 3 for 3d) / 2), namely y*theta
    t_width = t_width / scaling_factor
    freqs_width = torch.outer(t_width, inv_freq)  # (max_width, dim / (1 for 1d, 2 for 2d, 3 for 3d) / 2), namely x*theta
    
    # print(f"inv_freq: {inv_freq}")
    # print(f"t_width: {t_width}")
    # print(f"t_height.shape: {t_height.shape}")
    # print(f"t_width.shape: {t_width.shape}")
    # print(f"inv_freq.shape: {inv_freq.shape}")
    # print(f"freqs_height.shape: {freqs_height.shape}")
    # print(f"freqs_width.shape: {freqs_width.shape}")

    freqs_grid_map = torch.concat([
        freqs_height[:, None, :].expand(-1, max_width, -1), # (max_height, max_width, dim / (1 for 1d, 2 for 2d, 3 for 3d) / 2)
        freqs_width[None, :, :].expand(max_height, -1, -1), # (max_height, max_width, dim / (1 for 1d, 2 for 2d, 3 for 3d) / 2)
    ], dim=-1)  # (max_height, max_width, dim / (1 for 1d, 2 for 2d, 3 for 3d))
    freqs_grid_map = torch.stack([torch.cos(freqs_grid_map), torch.sin(freqs_grid_map)], dim=0)
    # (2, max_height, max_width, dim / (1 for 1d, 2 for 2d, 3 for 3d))

    # print(f"freqs_grid_map.shape: {freqs_grid_map.shape}")

    rope2d_freqs_grid = {}
    for h_div_w in dynamic_resolution_h_w:
        scale_schedule = dynamic_resolution_h_w[h_div_w]['1M']['scales']
        _, ph, pw = scale_schedule[-1]
        max_edge_length = freqs_grid_map.shape[1]
        if ph >= pw:
            uph, upw = max_edge_length, int(max_edge_length / ph * pw)
        else:
            uph, upw = int(max_edge_length / pw * ph), max_edge_length
        # print(f"uph: {uph}")
        rope_cache_list = []
        #scale_schedule: [1*1, 2*2, 4*4]
        # print(f"scale_schedule: {scale_schedule}")
        for (_, ph, pw) in scale_schedule:
            ph_mul_pw = ph * pw
            if rope2d_normalized_by_hw == 1: # downsample
                rope_cache = F.interpolate(freqs_grid_map[:, :uph, :upw, :].permute([0,3,1,2]), size=(ph, pw), mode='bilinear', align_corners=True)
                rope_cache = rope_cache.permute([0,2,3,1]) # (2, ph, pw, half_head_dim)
            elif rope2d_normalized_by_hw == 2: # star stylee

                # get height,width of largest scale
                _, uph, upw = scale_schedule[-1]
                # get xy indices
                indices = torch.stack([
                    (torch.arange(ph) * (uph / ph)).reshape(ph, 1).expand(ph, pw),
                    (torch.arange(pw) * (upw / pw)).reshape(1, pw).expand(ph, pw),
                ], dim=-1).round().int() # (ph, pw, 2)
                indices = indices.reshape(-1, 2) # (ph*pw, 2)
                # print(f"indices.shape: {indices.shape}")
                # print(f"indices: {indices}")

                rope_cache = freqs_grid_map[:, indices[:,0], indices[:,1], :] # (2, ph*pw, half_head_dim)
                rope_cache = rope_cache.reshape(2, ph, pw, -1)
            elif rope2d_normalized_by_hw == 0:
                rope_cache = freqs_grid_map[:, :ph, :pw, :] # (2, ph, pw, half_head_dim)
            else:
                raise ValueError(f'Unknown rope2d_normalized_by_hw: {rope2d_normalized_by_hw}')
            # rope_cache: [2, ph*pw, ]
            rope_cache_list.append(rope_cache.reshape(2, ph_mul_pw, -1))
        cat_rope_cache = torch.cat(rope_cache_list, 1) # (2, seq_len, half_head_dim)
        if cat_rope_cache.shape[1] % pad_to_multiplier:
            pad = torch.zeros(2, pad_to_multiplier - cat_rope_cache.shape[1] % pad_to_multiplier, half_dim)
            cat_rope_cache = torch.cat([cat_rope_cache, pad], dim=1)
        cat_rope_cache = cat_rope_cache[:,None,None,None] # (2, 1, 1, 1, seq_len, half_dim)
        # print(f"cat_rope_cache.shape: {cat_rope_cache.shape}")
        for pn in dynamic_resolution_h_w[h_div_w]:
            scale_schedule = dynamic_resolution_h_w[h_div_w][pn]['scales']
            tmp_scale_schedule = [(1, h, w) for _, h, w in scale_schedule]
            rope2d_freqs_grid[str(tuple(tmp_scale_schedule))] = cat_rope_cache
    return rope2d_freqs_grid


def apply_rotary_emb(q, k, 
                     scale_schedule, 
                     rope2d_freqs_grid, 
                     pad_to_multiplier, 
                     rope2d_normalized_by_hw, 
                     scale_ind):
    qk = torch.stack((q, k), dim=0)  #(2, batch_size, heads, seq_len, head_dim)
    # print(f"qk.shape: {qk.shape}")
    device_type = qk.device.type
    device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        seq_len = qk.shape[3]
        start = 0
        if scale_ind >= 1:
            assert len(scale_schedule[0]) == 3
            start = np.sum([item[0] * item[1] * item[2] for item in scale_schedule[:scale_ind]])
        rope2d_freqs_grid[str(tuple(scale_schedule))] = rope2d_freqs_grid[str(tuple(scale_schedule))].to(qk.device)
        assert start+seq_len <= rope2d_freqs_grid[str(tuple(scale_schedule))].shape[4]
        rope_cache = rope2d_freqs_grid[str(tuple(scale_schedule))][:, :, :, :, start:start+seq_len] # rope_cache shape: [2, 1, 1, 1, seq_len, half_head_dim]
        
        qk = qk.reshape(*qk.shape[:-1], -1, 2) #(2, batch_size, heads, seq_len, half_head_dim, 2)
        qk = torch.stack([
            rope_cache[0] * qk[...,0] - rope_cache[1] * qk[...,1],
            rope_cache[1] * qk[...,0] + rope_cache[0] * qk[...,1],
        ], dim=-1) # (2, batch_size, heads, seq_len, half_head_dim, 2), here stack + reshape should not be concate
        qk = qk.reshape(*qk.shape[:-2], -1) #(2, batch_size, heads, seq_len, head_dim)
        q, k = qk.unbind(dim=0) # (batch_size, heads, seq_len, head_dim)
    return q, k

class FastRMSNorm(nn.Module):
    def __init__(self, C, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.C = C
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(C))
        else:
            self.register_buffer('weight', torch.ones(C))
    
    def forward(self, x):
        src_type = x.dtype
        return rms_norm_impl(x.float(), self.weight, epsilon=self.eps).to(src_type)
    
    def extra_repr(self) -> str:
        return f'C={self.C}, eps={self.eps:g}, elementwise_affine={self.elementwise_affine}'


def get_dropout_layer(p):
    return nn.Dropout(p, inplace=True) if p > 0 else nn.Identity()


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., fused_mlp=False):
        super().__init__()
        self.fused_mlp_func = fused_mlp_func if fused_mlp else None
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = get_dropout_layer(drop)
        self.heuristic = 0
    
    def forward(self, x):
        if self.fused_mlp_func is not None:
            return self.drop(self.fused_mlp_func(
                x=x,
                weight1=self.fc1.weight,
                weight2=self.fc2.weight,
                bias1=self.fc1.bias,
                bias2=self.fc2.bias,
                activation='gelu_approx',
                save_pre_act=self.training,
                return_residual=False,
                checkpoint_lvl=0,
                heuristic=self.heuristic,
                process_group=None,
            ))
        else:
            return self.drop(self.fc2( self.act(self.fc1(x)) ))
    
    def extra_repr(self) -> str:
        return f'fused_mlp={self.fused_mlp_func is not None}'


class FFNSwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features, out_features=None, drop=0., fused_mlp=False):
        super().__init__()
        self.fused_mlp_func = None
        hidden_features = round(2 * hidden_features / 3 / 256) * 256
        
        out_features = out_features or in_features
        self.fcg = nn.Linear(in_features, hidden_features, bias=False)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)
        self.drop = get_dropout_layer(drop)
    
    def forward(self, x):
        return self.drop(self.fc2( F.silu(self.fcg(x), inplace=True).mul_(self.fc1(x)) ))
    
    def extra_repr(self) -> str:
        return f'fused_mlp={self.fused_mlp_func is not None}'

@staticmethod
def _lift_K(Ks):
    out = torch.zeros(Ks.shape[:-2] + (4, 4), device=Ks.device, dtype=Ks.dtype)
    out[..., :3, :3] = Ks
    out[..., 3, 3] = 1.0
    return out

@staticmethod
def _invert_K(Ks):
    out = torch.zeros_like(Ks)
    out[..., 0, 0] = 1.0 / Ks[..., 0, 0]
    out[..., 1, 1] = 1.0 / Ks[..., 1, 1]
    out[..., 0, 2] = -Ks[..., 0, 2] / Ks[..., 0, 0]
    out[..., 1, 2] = -Ks[..., 1, 2] / Ks[..., 1, 1]
    out[..., 2, 2] = 1.0
    return out

@staticmethod
def _invert_SE3(transforms):
    Rinv = transforms[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(transforms)
    out[..., :3, :3] = Rinv
    out[..., :3, 3] = -torch.einsum("...ij,...j->...i", Rinv, transforms[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out

def get_prope_matrices(poses_c2w, intrs):
    """
    Calculates P and P_inv.
    Matches Original: input `poses` is converted to `inv(poses)` (w2c) logic.
    P = Lift(K) @ inv(poses)  (World -> Screen)
    P_inv = poses @ Lift(K)^-1 (Screen -> World)
    """
    poses_w2c = _invert_SE3(poses_c2w) # World-to-Camera (assuming poses is Camera-to-World)

    if intrs is not None:
        Ks_norm = intrs.clone()
        Ks_norm[..., 0, 2] -= 0.5
        Ks_norm[..., 1, 2] -= 0.5
        
        lifted_K = _lift_K(Ks_norm)
        # P = K @ w2c
        P = torch.einsum("...ij,...jk->...ik", lifted_K, poses_w2c)

        # P_inv
        # view_inv = self._invert_SE3(pose_inv)
        lifted_K_inv = _lift_K(_invert_K(Ks_norm))
        
        # P_inv = c2w @ K_inv
        P_inv = torch.einsum("...ij,...jk->...ik", poses_c2w, lifted_K_inv)

        P_T = P.transpose(-1,-2)
    else:
        P = poses_w2c
        P_inv = poses_c2w
        P_T = P.transpose(-1,-2)
        
    return P, P_T, P_inv


class SelfAttention(nn.Module):
    def __init__(
        self, embed_dim=768, num_heads=12,
        proj_drop=0., tau=1, cos_attn=False, customized_flash_attn=True, use_flex_attn=False, 
        batch_size=2, pad_to_multiplier=1, rope2d_normalized_by_hw=0, N_views=1
    ):
        """
        :param embed_dim: model's width
        :param num_heads: num heads of multi-head attention
        :param proj_drop: always 0 for testing
        :param tau: always 1
        :param cos_attn: always True: during attention, q and k will be L2-normalized and scaled by a head-wise learnable parameter self.scale_mul_1H11
        :param customized_flash_attn:
        """
        super().__init__()
        assert embed_dim % num_heads == 0
        self.using_flash = customized_flash_attn
        
        self.num_heads, self.head_dim = num_heads, embed_dim // num_heads
        self.tau, self.cos_attn = tau, cos_attn
        if self.cos_attn:
            self.scale = 1
            size = (1, 1, self.num_heads, 1) if self.using_flash else (1, self.num_heads, 1, 1)
            # size: 11H1 or 1H11
            self.scale_mul_1H11 = nn.Parameter(torch.full(size=size, fill_value=4.0).log(), requires_grad=True)
            self.max_scale_mul = torch.log(torch.tensor(100)).item()
        else:
            self.scale = 1 / math.sqrt(self.head_dim) / self.tau
        
        self.mat_qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.q_bias, self.v_bias = nn.Parameter(torch.zeros(embed_dim)), nn.Parameter(torch.zeros(embed_dim))
        self.register_buffer('zero_k_bias', torch.zeros(embed_dim))
        
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = get_dropout_layer(proj_drop)
        
        self.caching = False    # kv caching: only used during inference
        self.cached_k = None    # kv caching: only used during inference
        self.cached_v = None    # kv caching: only used during inference

        self.use_flex_attn = use_flex_attn
        self.pad_to_multiplier = pad_to_multiplier

        self.rope2d_normalized_by_hw = rope2d_normalized_by_hw
    
    def kv_caching(self, enable: bool): # kv caching: only used during inference
        self.caching = enable
        self.cached_k = None
        self.cached_v = None
    
    # NOTE: attn_bias_or_two_vector is None during inference
    def forward(self, 
                x, 
                attn_bias_or_two_vector: Union[torch.Tensor, Tuple[torch.IntTensor, torch.IntTensor]], 
                attn_fn=None, 
                scale_schedule=None, 
                rope2d_freqs_grid=None, 
                scale_ind=0):
        """
        :param (fp32) x: shaped (B or batch_size, L or seq_length, C or hidden_dim); if seq-parallel is used, the `L` dim would be shared
        :param (fp32) attn_bias_or_two_vector:
                if not using_flash:
                    a block-wise, lower-triangle matrix, like:
                    [[[[0, -, -, -, -, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]]
                    where 0 means visible and - means invisible (-inf)
                else:
                    a tuple of two 1-dim int vector (VAR_visible_kvlen, VAR_invisible_qlen)
        :return: shaped (B or batch_size, L or seq_length, C or hidden_dim); if seq-parallel is used, the `L` dim would be shared
        """
        # x: fp32
        B, L, C = x.shape
        
        # qkv: amp, bf16
        qkv = F.linear(input=x, weight=self.mat_qkv.weight, bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias))).view(B, L, 3, self.num_heads, self.head_dim)  # BL3Hc
        if self.using_flash: q, k, v = qkv.unbind(dim=2); L_dim = 1           # q or k or v: all are shaped in (B:batch_size, L:seq_len, H:heads, c:head_dim)
        else: q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0); L_dim = 2   # q or k or v: all are shaped in (B:batch_size, H:heads, L:seq_len, c:head_dim)
        
        if self.cos_attn:   # always True
            scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp() # 11H1 (flash), or 1H11 (not flash)
            q = F.normalize(q, dim=-1, eps=1e-12).mul(scale_mul).contiguous()   # fp32
            k = F.normalize(k, dim=-1, eps=1e-12).contiguous()                  # fp32
            v = v.contiguous()                                                  # bf16
        else:   # be contiguous, to make kernel happy
            q = q.contiguous()      # bf16
            k = k.contiguous()      # bf16
            v = v.contiguous()      # bf16

        #     q, k = apply_rotary_emb(q, k, scale_schedule, rope2d_freqs_grid, self.pad_to_multiplier, self.rope2d_normalized_by_hw, scale_ind)
        if self.caching:    # kv caching: only used during inference
            if self.cached_k is None: self.cached_k = k; self.cached_v = v
            else: k = self.cached_k = torch.cat((self.cached_k, k), dim=L_dim); v = self.cached_v = torch.cat((self.cached_v, v), dim=L_dim)
        
        

        if self.using_flash:
            if attn_bias_or_two_vector is not None: # training
                kw = dict(VAR_visible_kvlen=attn_bias_or_two_vector[0], VAR_invisible_qlen=attn_bias_or_two_vector[1])
            else:                                   # inference (autoregressive sampling)
                kw = dict()
            oup = flash_attn_func(q.to(v.dtype), k.to(v.dtype), v, dropout_p=0, softmax_scale=self.scale, **kw).view(B, L, C)
        else:
            # if self.cos_attn: q, k are in fp32; v is in bf16
            # else: q, k, v are in bf16
            if self.use_flex_attn and attn_fn is not None:
                oup = attn_fn(q, k, v, scale=self.scale).transpose(1, 2).reshape(B, L, C)
            else:
                oup = slow_attn(query=q, key=k, value=v, scale=self.scale, attn_mask=attn_bias_or_two_vector, dropout_p=0).transpose(1, 2).reshape(B, L, C)
            # oup: bf16
        
        return self.proj_drop(self.proj(oup))
    
    def extra_repr(self) -> str:
        tail = ''
        return f'using_flash={self.using_flash}, tau={self.tau}, cos_attn={self.cos_attn}{tail}'

class SelfAttentionPrefixed(nn.Module):
    def __init__(
        self, embed_dim=768, num_heads=12,
        proj_drop=0., tau=1, cos_attn=False, customized_flash_attn=True, use_flex_attn=False, 
        batch_size=2, pad_to_multiplier=1, rope2d_normalized_by_hw=0, N_views=1
    ):
        """
        :param embed_dim: model's width
        :param num_heads: num heads of multi-head attention
        :param proj_drop: always 0 for testing
        :param tau: always 1
        :param cos_attn: always True: during attention, q and k will be L2-normalized and scaled by a head-wise learnable parameter self.scale_mul_1H11
        :param customized_flash_attn:
        """
        super().__init__()
        assert embed_dim % num_heads == 0
        self.using_flash = customized_flash_attn
        
        self.num_heads, self.head_dim = num_heads, embed_dim // num_heads
        self.tau, self.cos_attn = tau, cos_attn
        if self.cos_attn:
            self.scale = 1
            size = (1, 1, self.num_heads, 1) if self.using_flash else (1, self.num_heads, 1, 1)
            # size: 11H1 or 1H11
            self.scale_mul_1H11 = nn.Parameter(torch.full(size=size, fill_value=4.0).log(), requires_grad=True)
            self.max_scale_mul = torch.log(torch.tensor(100)).item()
        else:
            self.scale = 1 / math.sqrt(self.head_dim) / self.tau
        
        self.mat_qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.q_bias, self.v_bias = nn.Parameter(torch.zeros(embed_dim)), nn.Parameter(torch.zeros(embed_dim))
        self.register_buffer('zero_k_bias', torch.zeros(embed_dim))
        
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = get_dropout_layer(proj_drop)
        
        self.caching = False    # kv caching: only used during inference
        self.cached_k = None    # kv caching: only used during inference
        self.cached_v = None    # kv caching: only used during inference

        self.use_flex_attn = use_flex_attn
        self.pad_to_multiplier = pad_to_multiplier

        self.rope2d_normalized_by_hw = rope2d_normalized_by_hw
    
    def kv_caching(self, enable: bool): # kv caching: only used during inference
        self.caching = enable
        self.cached_k = None
        self.cached_v = None
    
    # NOTE: attn_bias_or_two_vector is None during inference
    def forward(self, 
                x, 
                attn_bias_or_two_vector: Union[torch.Tensor, Tuple[torch.IntTensor, torch.IntTensor]], 
                attn_fn=None, 
                scale_schedule=None, 
                scale_ind=0):
        """
        :param (fp32) x: shaped (B or batch_size, L or seq_length, C or hidden_dim); if seq-parallel is used, the `L` dim would be shared
        :param (fp32) attn_bias_or_two_vector:
                if not using_flash:
                    a block-wise, lower-triangle matrix, like:
                    [[[[0, -, -, -, -, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]]
                    where 0 means visible and - means invisible (-inf)
                else:
                    a tuple of two 1-dim int vector (VAR_visible_kvlen, VAR_invisible_qlen)
        :return: shaped (B or batch_size, L or seq_length, C or hidden_dim); if seq-parallel is used, the `L` dim would be shared
        """
        # x: fp32
        B, L, C = x.shape
        
        # qkv: amp, bf16
        qkv = F.linear(input=x, weight=self.mat_qkv.weight, bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias))).view(B, L, 3, self.num_heads, self.head_dim)  # BL3Hc
        if self.using_flash: q, k, v = qkv.unbind(dim=2); L_dim = 1           # q or k or v: all are shaped in (B:batch_size, L:seq_len, H:heads, c:head_dim)
        else: q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0); L_dim = 2   # q or k or v: all are shaped in (B:batch_size, H:heads, L:seq_len, c:head_dim)
        
        if self.cos_attn:   # always True
            scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp() # 11H1 (flash), or 1H11 (not flash)
            q = F.normalize(q, dim=-1, eps=1e-12).mul(scale_mul).contiguous()   # fp32
            k = F.normalize(k, dim=-1, eps=1e-12).contiguous()                  # fp32
            v = v.contiguous()                                                  # bf16
        else:   # be contiguous, to make kernel happy
            q = q.contiguous()      # bf16
            k = k.contiguous()      # bf16
            v = v.contiguous()      # bf16

        if self.caching:    # kv caching: only used during inference
            if self.cached_k is None: self.cached_k = k; self.cached_v = v
            else: k = self.cached_k = torch.cat((self.cached_k, k), dim=L_dim); v = self.cached_v = torch.cat((self.cached_v, v), dim=L_dim)
        
        if self.using_flash:
            if attn_bias_or_two_vector is not None: # training
                kw = dict(VAR_visible_kvlen=attn_bias_or_two_vector[0], VAR_invisible_qlen=attn_bias_or_two_vector[1])
            else:                                   # inference (autoregressive sampling)
                kw = dict()
            oup = flash_attn_func(q.to(v.dtype), 
                                  k.to(v.dtype), 
                                  v, 
                                  dropout_p=0, 
                                  softmax_scale=self.scale, 
                                  **kw).view(B, L, C)
        else:
            # if self.cos_attn: q, k are in fp32; v is in bf16
            # else: q, k, v are in bf16
            if self.use_flex_attn and attn_fn is not None:
                oup = attn_fn(q, k, v, scale=self.scale).transpose(1, 2).reshape(B, L, C)
            else:
                oup = slow_attn(query=q, 
                                key=k, 
                                value=v, 
                                scale=self.scale, 
                                attn_mask=attn_bias_or_two_vector, 
                                dropout_p=0).transpose(1, 2).reshape(B, L, C)
            # oup: bf16
        
        return self.proj_drop(self.proj(oup))
    
    def extra_repr(self) -> str:
        tail = ''
        return f'using_flash={self.using_flash}, tau={self.tau}, cos_attn={self.cos_attn}{tail}'


class SelfAttentionPrope(nn.Module):
    def __init__(
        self, embed_dim=768, num_heads=12,
        proj_drop=0., tau=1, cos_attn=False, customized_flash_attn=True, use_flex_attn=False, 
        batch_size=2, pad_to_multiplier=1, rope2d_normalized_by_hw=0, N_views=1
    ):
        """
        Implements PRope
        :param embed_dim: model's width
        :param num_heads: num heads of multi-head attention
        :param proj_drop: always 0 for testing
        :param tau: always 1
        :param cos_attn: always True: during attention, q and k will be L2-normalized and scaled by a head-wise learnable parameter self.scale_mul_1H11
        :param customized_flash_attn:
        """
        super().__init__()
        assert embed_dim % num_heads == 0
        self.using_flash = customized_flash_attn
        
        self.num_heads, self.head_dim = num_heads, embed_dim // num_heads
        self.tau, self.cos_attn = tau, cos_attn
        if self.cos_attn:
            self.scale = 1
            size = (1, 1, self.num_heads, 1) if self.using_flash else (1, self.num_heads, 1, 1)
            # size: 11H1 or 1H11
            self.scale_mul_1H11 = nn.Parameter(torch.full(size=size, fill_value=4.0).log(), requires_grad=True)
            self.max_scale_mul = torch.log(torch.tensor(100)).item()
        else:
            self.scale = 1 / math.sqrt(self.head_dim) / self.tau
        
        self.mat_qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.q_bias, self.v_bias = nn.Parameter(torch.zeros(embed_dim)), nn.Parameter(torch.zeros(embed_dim))
        self.register_buffer('zero_k_bias', torch.zeros(embed_dim))
        
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = get_dropout_layer(proj_drop)
        
        self.caching = False    # kv caching: only used during inference
        self.cached_k = None    # kv caching: only used during inference
        self.cached_v = None    # kv caching: only used during inference

        self.use_flex_attn = use_flex_attn
        self.pad_to_multiplier = pad_to_multiplier

        self.rope2d_normalized_by_hw = rope2d_normalized_by_hw
    
    def kv_caching(self, enable: bool): # kv caching: only used during inference
        self.caching = enable
        self.cached_k = None
        self.cached_v = None

    def prope_dot_product_infinity(
        self,
        q: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
        k: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
        v: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
        *,
        viewmats: torch.Tensor,  # (batch, cameras, 4, 4)
        Ks: Optional[torch.Tensor],  # (batch, cameras, 3, 3)
        scale_schedule,
        scale_ind=None,
        scale=None,
        attn_mask=None,
        **kwargs,
    ) -> torch.Tensor:
        """Similar to torch.nn.functional.scaled_dot_product_attention, but applies PRoPE-style
        positional encoding.

        Currently, we assume that the sequence length is equal to:

            cameras * patches_x * patches_y

        And token ordering allows the `(seqlen,)` axis to be reshaped into
        `(cameras, patches_x, patches_y)`.
        """
        # We're going to assume self-attention: all inputs are the same shape.
        (batch, num_heads, seqlen, head_dim) = q.shape
        cameras = viewmats.shape[1]
        assert q.shape == k.shape == v.shape
        assert viewmats.shape == (batch, cameras, 4, 4)
        assert Ks is None or Ks.shape == (batch, cameras, 3, 3)
        # assert seqlen == cameras * patches_x * patches_y
        n_cameras = viewmats.shape[1]
        viewmats = torch.linalg.inv(viewmats)

        if scale_ind is not None:
            scale_schedule = [scale_schedule[scale_ind]]

        tok_start = 0
        
        k_chunks = []
        v_chunks = []
        outs = []

        patches_xs = [px for _,px,py in scale_schedule]
        patches_ys = [py for _,px,py in scale_schedule]

        device=viewmats.device
        pos_x_list = []
        for px, py in zip(patches_xs, patches_ys):
            pos_x_scale = torch.tile(torch.arange(px, device=device), (py * cameras,))
            pos_x_list.append(pos_x_scale)
        pos_x_total = torch.cat(pos_x_list, dim=0)
        coeffs_x = _rope_precompute_coeffs(pos_x_total, freq_base=100.0, freq_scale=1.0, feat_dim=head_dim // 4)

        pos_y_list = []
        for px, py in zip(patches_xs, patches_ys):
            pos_y_scale = torch.tile(torch.repeat_interleave(torch.arange(py, device=device), px), (cameras,))
            pos_y_list.append(pos_y_scale)
        pos_y_total = torch.cat(pos_y_list, dim=0)
        coeffs_y = _rope_precompute_coeffs(pos_y_total, freq_base=100.0, freq_scale=1.0, feat_dim=head_dim // 4)

        for i in range(len(scale_schedule)):
        
            patches_x = scale_schedule[i][-1]
            patches_y = scale_schedule[i][-2]
            tok_end = tok_start + n_cameras*patches_x*patches_y

            apply_fn_q, apply_fn_kv, apply_fn_o = _prepare_apply_fns(
                head_dim=head_dim,
                viewmats=viewmats,
                Ks=Ks,
                patches_x=patches_x,
                patches_y=patches_y,
                coeffs_x=(coeffs_x[0][:,:, tok_start:tok_end], coeffs_x[1][:,:, tok_start:tok_end]),
                coeffs_y=(coeffs_y[0][:,:, tok_start:tok_end], coeffs_y[1][:,:, tok_start:tok_end])
            )

            q_chunk = apply_fn_q(q[:, :, tok_start:tok_end])    # shape (B, H, qlen, D)
            k_chunk = apply_fn_kv(k[:, :, tok_start:tok_end])   # shape (B, H, klen_chunk, D)
            v_chunk = apply_fn_kv(v[:, :, tok_start:tok_end])   # shape (B, H, vlen_chunk, D)

            if self.cached_k is None:
                self.cached_k = k_chunk
                self.cached_v = v_chunk
            else:
                k_chunk = self.cached_k = torch.cat((self.cached_k, k_chunk), dim=2)
                v_chunk = self.cached_v = torch.cat((self.cached_v, v_chunk), dim=2)

            out_i = F.scaled_dot_product_attention(
                query=q_chunk,
                key=k_chunk,
                value=v_chunk,
                # scale=scale,
                **kwargs
            )
            # print(f"out_i.device: {out_i.device}")
            out_i = apply_fn_o(out_i)

            # out[:,:,tok_start:tok_end] = out_i
            outs.append(out_i)

            del out_i, q_chunk, k_chunk, v_chunk

            tok_start = tok_end

        # if train mode -> clean cache here, test mode cache is cleaned outside
        if not self.caching:
            self.cached_k = None
            self.cached_v = None


        outs = torch.cat(outs, dim=2) if len(outs) > 1 else outs[0]

        assert outs.shape == (batch, num_heads, seqlen, head_dim)
        return outs

    # NOTE: attn_bias_or_two_vector is None during inference
    def forward(self, 
                x, 
                attn_bias_or_two_vector: Union[torch.Tensor, Tuple[torch.IntTensor, torch.IntTensor]], 
                attn_fn=None, 
                scale_schedule=None, 
                rope2d_freqs_grid=None, 
                scale_ind=None,
                poses=None,
                intrs=None,
                input_size=None,
                **kwargs):
        """
        :param (fp32) x: shaped (B or batch_size, L or seq_length, C or hidden_dim); if seq-parallel is used, the `L` dim would be shared
        :param (fp32) attn_bias_or_two_vector:
                if not using_flash:
                    a block-wise, lower-triangle matrix, like:
                    [[[[0, -, -, -, -, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]]
                    where 0 means visible and - means invisible (-inf)
                else:
                    a tuple of two 1-dim int vector (VAR_visible_kvlen, VAR_invisible_qlen)
        :return: shaped (B or batch_size, L or seq_length, C or hidden_dim); if seq-parallel is used, the `L` dim would be shared
        """
        # x: fp32
        B, L, C = x.shape

        
        # qkv: amp, bf16
        # qkv: [B,L,3,16,128] [2,3,3,16,128]
        qkv = F.linear(input=x, weight=self.mat_qkv.weight, bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias))).view(B, L, 3, self.num_heads, self.head_dim)  # BL3Hc
        # print(f"qkv.shape: {qkv.shape}")
        if self.using_flash:
            q, k, v = qkv.unbind(dim=2)
            L_dim = 1           # q or k or v: all are shaped in (B:batch_size, L:seq_len, H:heads, c:head_dim)
        else: 
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
            L_dim = 2   # q or k or v: all are shaped in (B:batch_size, H:heads, L:seq_len, c:head_dim)
        
        if self.cos_attn:   # always True
            scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp() # 11H1 (flash), or 1H11 (not flash)
            q = F.normalize(q, dim=-1, eps=1e-12).mul(scale_mul).contiguous()   # fp32
            k = F.normalize(k, dim=-1, eps=1e-12).contiguous()                  # fp32
            v = v.contiguous()                                                  # bf16
        else:   # be contiguous, to make kernel happy
            q = q.contiguous()      # bf16
            k = k.contiguous()      # bf16
            v = v.contiguous()      # bf16
        oup = self.prope_dot_product_infinity(q,k,v,
                                    viewmats=poses,
                                    Ks=intrs,
                                    scale_schedule=scale_schedule,
                                    scale_ind=scale_ind,
                                    scale=self.scale,
                                    attn_mask=attn_bias_or_two_vector)
        oup = oup.transpose(1,2).reshape(B,L,C)

        return self.proj_drop(self.proj(oup))
    
    def extra_repr(self) -> str:
        tail = ''
        return f'using_flash={self.using_flash}, tau={self.tau}, cos_attn={self.cos_attn}{tail}'


@torch.compile
def apply_cos_attn_fused(q, k, scale_mul, target_dtype):
    # Compute mathematically stable fp32 norm, scale, then cast down
    q_norm = F.normalize(q.to(torch.float32), dim=-1, eps=1e-12) * scale_mul
    k_norm = F.normalize(k.to(torch.float32), dim=-1, eps=1e-12)
    return q_norm.to(target_dtype), k_norm.to(target_dtype)


def _build_switti_varlen_metadata(
    scale_schedule,
    batch_size,
    n_views,
    sequence_length,
    device,
):
    """Build packed-sequence offsets for batch-major, scale-major SWITTI tokens."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if n_views < 1:
        raise ValueError(f"n_views must be positive, got {n_views}")
    if not scale_schedule:
        raise ValueError("scale_schedule must contain at least one scale")

    scale_lengths = [
        n_views * math.prod(int(dim) for dim in scale)
        for scale in scale_schedule
    ]
    expected_length = sum(scale_lengths)
    if expected_length != sequence_length:
        raise ValueError(
            "Packed SWITTI attention requires an unpadded scale-major sequence: "
            f"expected {expected_length} tokens from scale_schedule and n_views={n_views}, "
            f"but received {sequence_length}."
        )

    segment_lengths = torch.tensor(
        scale_lengths * batch_size,
        dtype=torch.int32,
        device=device,
    )
    cu_seqlens = torch.zeros(
        segment_lengths.numel() + 1,
        dtype=torch.int32,
        device=device,
    )
    cu_seqlens[1:] = torch.cumsum(segment_lengths, dim=0, dtype=torch.int32)
    return cu_seqlens, max(scale_lengths)

class SelfAttentionPropeOptimized(nn.Module):
    def __init__(
        self, embed_dim=768, num_heads=12,
        proj_drop=0., tau=1, cos_attn=False, customized_flash_attn=True, use_flex_attn=False, 
        batch_size=2, pad_to_multiplier=1, rope2d_normalized_by_hw=0, N_views=1,
        switti_attn_backend="sdpa",
    ):
        """
        Implements PRope
        :param embed_dim: model's width
        :param num_heads: num heads of multi-head attention
        :param proj_drop: always 0 for testing
        :param tau: always 1
        :param cos_attn: always True: during attention, q and k will be L2-normalized and scaled by a head-wise learnable parameter self.scale_mul_1H11
        :param customized_flash_attn:
        """
        super().__init__()
        assert embed_dim % num_heads == 0
        self.using_flash = customized_flash_attn
        
        self.num_heads, self.head_dim = num_heads, embed_dim // num_heads
        self.tau, self.cos_attn = tau, cos_attn
        if self.cos_attn:
            self.scale = 1
            size = (1, 1, self.num_heads, 1) if self.using_flash else (1, self.num_heads, 1, 1)
            # size: 11H1 or 1H11
            self.scale_mul_1H11 = nn.Parameter(torch.full(size=size, fill_value=4.0).log(), requires_grad=True)
            self.max_scale_mul = torch.log(torch.tensor(100)).item()
        else:
            self.scale = 1 / math.sqrt(self.head_dim) / self.tau
        
        self.mat_qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.q_bias, self.v_bias = nn.Parameter(torch.zeros(embed_dim)), nn.Parameter(torch.zeros(embed_dim))
        self.register_buffer('zero_k_bias', torch.zeros(embed_dim))
        
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = get_dropout_layer(proj_drop)
        
        self.caching = False    # kv caching: only used during inference
        self.cached_k = None    # kv caching: only used during inference
        self.cached_v = None    # kv caching: only used during inference
        
        # max_seq_len = N_views*521
        # self.max_seq_len = max_seq_len
        # self.register_buffer("kv_cache", None, persistent=False)
        # self.cache_index = 0

        self.use_flex_attn = use_flex_attn
        self.pad_to_multiplier = pad_to_multiplier
        if switti_attn_backend not in {"sdpa", "flash_varlen", "flex"}:
            raise ValueError(
                "switti_attn_backend must be 'sdpa',  "
                "'flash_varlen', or 'flex', got "
                f"{switti_attn_backend!r}."
            )
        if switti_attn_backend == "flash_varlen" and customized_flash_attn:
            raise ValueError(
                "switti_attn_backend='flash_varlen' requires the customized dense "
                "FlashAttention path to be disabled (--flash=0)."
            )
        if switti_attn_backend == "flash_varlen" and use_flex_attn:
            raise ValueError(
                "switti_attn_backend='flash_varlen' cannot be combined with FlexAttention."
            )
        if switti_attn_backend == "flash_varlen" and pad_to_multiplier > 1:
            raise ValueError(
                "switti_attn_backend='flash_varlen' requires an unpadded sequence; "
                f"got pad_to_multiplier={pad_to_multiplier}."
            )
        if switti_attn_backend == "flash_varlen" and flash_attn_varlen_func is None:
            raise RuntimeError(
                "switti_attn_backend='flash_varlen' requires a FlashAttention build "
                "that exposes flash_attn_varlen_func."
            )
        if switti_attn_backend == "flash_varlen" and self.head_dim > 256:
            raise ValueError(
                "FlashAttention varlen supports head dimensions up to 256, got "
                f"head_dim={self.head_dim}."
            )
        if switti_attn_backend == "flex" and customized_flash_attn:
            raise ValueError(
                "switti_attn_backend='flex' requires the customized dense "
                "FlashAttention path to be disabled (--flash=0)."
            )
        if switti_attn_backend == "flex" and use_flex_attn:
            raise ValueError(
                "switti_attn_backend='flex' is separate from the legacy AR "
                "FlexAttention path; set use_flex_attn=False."
            )
        if switti_attn_backend == "flex" and pad_to_multiplier > 1:
            raise ValueError(
                "switti_attn_backend='flex' currently requires an unpadded "
                f"sequence; got pad_to_multiplier={pad_to_multiplier}."
            )
        if switti_attn_backend == "flex" and not flex_attention_available:
            raise RuntimeError(
                "switti_attn_backend='flex' requires PyTorch FlexAttention "
                "(torch>=2.5.1)."
            )
        self.switti_attn_backend = switti_attn_backend
        self.rope2d_normalized_by_hw = rope2d_normalized_by_hw
    
    def kv_caching(self, enable: bool): # kv caching: only used during inference
        self.caching = enable
        self.cached_k = None
        self.cached_v = None

        # if enable:
        #     # Pre-allocate cache: (2, B, num_heads, max_seq_len, head_dim)
        #     # We treat K and V together for easier management
        #     self.kv_cache = None 
        #     self.cache_index = 0
        # else:
        #     self.kv_cache = None

    # def _update_kv_cache(self, k, v):
    #     B, H, S, D = k.shape
        
    #     # Initialize cache lazily to handle batch size dynamically
    #     if self.kv_cache is None:
    #          self.kv_cache = torch.zeros(
    #              2, B, H, self.max_seq_len, D, 
    #              dtype=k.dtype, device=k.device
    #          )
        
    #     # Write to pre-allocated buffer
    #     end_idx = self.cache_index + S
    #     self.kv_cache[0, :, :, self.cache_index:end_idx, :] = k
    #     self.kv_cache[1, :, :, self.cache_index:end_idx, :] = v
    #     self.cache_index = end_idx
        
    #     # Return full valid sequence
    #     return (
    #         self.kv_cache[0, :, :, :self.cache_index, :], 
    #         self.kv_cache[1, :, :, :self.cache_index, :]
    #     )

    def prope_dot_product_vectorized(
        self, q, k, v,
        poses_c2w, Ks,
        scale_schedule,
        scale_ind=None,
        attn_mask=None,
        prope_cache=None,
        **kwargs
    ):
        """
        Vectorized implementation that avoids Python loops over sequence chunks.
        """
        B, num_heads, seqlen, head_dim = q.shape
        N_views = poses_c2w.shape[1]
        device = q.device

        cache_key = f"tgt_scale_{scale_ind}" if scale_ind is not None else "tgt_full"
        cached_geom = None

        
        if prope_cache is not None and cache_key in prope_cache:
            cached_geom = prope_cache.get(cache_key)

        else:
            # 1. Select schedule (Single step vs Full sequence)
            if scale_ind is not None:
                current_schedule = [scale_schedule[scale_ind]]
            else:
                current_schedule = scale_schedule

            # 2. Compute Camera Matrices
            P, P_T, P_inv = get_prope_matrices(poses_c2w=poses_c2w, intrs=Ks)

            pos_x_list, pos_y_list = [], []
            P_T_list, P_inv_list, P_list = [], [], []
            is_single_scale = (len(current_schedule) == 1)

            for _, px, py in current_schedule:
                num_repeats = px * py
                
                x_grid = torch.arange(px, device=device).repeat(py * N_views)
                y_grid = torch.arange(py, device=device).repeat_interleave(px).repeat(N_views)

                pos_x_list.append(x_grid)
                pos_y_list.append(y_grid)
                
                if not is_single_scale:
                    P_T_list.append(P_T.repeat_interleave(num_repeats, dim=1))
                    P_inv_list.append(P_inv.repeat_interleave(num_repeats, dim=1))
                    P_list.append(P.repeat_interleave(num_repeats, dim=1))

            pos_x_total = torch.cat(pos_x_list, dim=0)
            pos_y_total = torch.cat(pos_y_list, dim=0)
            coeffs_x = self._rope_precompute_coeffs(pos_x_total, 100.0, 1.0, head_dim // 4)
            coeffs_y = self._rope_precompute_coeffs(pos_y_total, 100.0, 1.0, head_dim // 4)

            if is_single_scale:
                # FAST PATH: Keep matrices compact (B, V, 4, 4)
                P_T_seq, P_inv_seq, P_seq = P_T, P_inv, P
            else:
                # SLOW PATH: Concatenate expanded matrices (B, L, 4, 4)
                P_T_seq = torch.cat(P_T_list, dim=1) 
                P_inv_seq = torch.cat(P_inv_list, dim=1)
                P_seq = torch.cat(P_list, dim=1)

            cached_geom = (P_T_seq, P_inv_seq, P_seq, coeffs_x, coeffs_y)

            if prope_cache is not None:
                prope_cache[cache_key] = cached_geom

        if scale_ind is not None:
            scale_schedule = [scale_schedule[scale_ind]]

        P_T_seq, P_inv_seq, P_seq, coeffs_x, coeffs_y = cached_geom

        # Preserve the projected autocast dtype for backends that require
        # fp16/bf16. ProPE coefficients promote the transformed tensors to
        # fp32; dense SDPA and SWITTI FlexAttention intentionally retain that
        # post-ProPE precision.
        projected_attention_dtype = v.dtype

        # Apply transforms to Q, K, V
        q = self._apply_transform(q, P_T_seq, coeffs_x, coeffs_y, inverse_rope=False)
        k = self._apply_transform(k, P_inv_seq, coeffs_x, coeffs_y, inverse_rope=False)
        v = self._apply_transform(v, P_inv_seq, coeffs_x, coeffs_y, inverse_rope=False)

        # --- ATTENTION PHASE ---
        # if self.caching:
        #     k, v = self._update_kv_cache(k, v)
        if self.caching:
            if self.cached_k is None:
                self.cached_k = k
                self.cached_v = v
            else:
                k = self.cached_k = torch.cat((self.cached_k, k), dim=2)
                v = self.cached_v = torch.cat((self.cached_v, v), dim=2)
        
        # Teacher-forced SWITTI packs every (batch item, scale) pair as an
        # independent sequence. Progressive inference keeps using SDPA because
        # scale_ind is set and its Q/K/V are already current-scale-only.
        if self.switti_attn_backend == "flash_varlen" and scale_ind is None:
            if self.caching:
                raise RuntimeError(
                    "Packed SWITTI FlashAttention cannot be combined with KV caching."
                )
            if q.device.type != "cuda":
                raise RuntimeError(
                    "Packed SWITTI FlashAttention requires CUDA tensors, got "
                    f"device={q.device}."
                )
            if projected_attention_dtype not in {torch.float16, torch.bfloat16}:
                raise RuntimeError(
                    "Packed SWITTI FlashAttention requires fp16 or bf16 QKV from "
                    f"autocast, got dtype={projected_attention_dtype}."
                )

            metadata_key = (
                "switti_varlen",
                B,
                N_views,
                tuple(tuple(int(dim) for dim in scale) for scale in scale_schedule),
                q.device.type,
                q.device.index,
            )
            metadata = None if prope_cache is None else prope_cache.get(metadata_key)
            if metadata is None:
                metadata = _build_switti_varlen_metadata(
                    scale_schedule=scale_schedule,
                    batch_size=B,
                    n_views=N_views,
                    sequence_length=seqlen,
                    device=q.device,
                )
                if prope_cache is not None:
                    prope_cache[metadata_key] = metadata
            cu_seqlens, max_seqlen = metadata

            q_attention = q.to(projected_attention_dtype).contiguous()
            k_attention = k.to(projected_attention_dtype).contiguous()
            v_attention = v.to(projected_attention_dtype).contiguous()

            q_packed = q_attention.permute(0, 2, 1, 3).contiguous().view(-1, num_heads, head_dim)
            k_packed = k_attention.permute(0, 2, 1, 3).contiguous().view(-1, num_heads, head_dim)
            v_packed = v_attention.permute(0, 2, 1, 3).contiguous().view(-1, num_heads, head_dim)
            out = flash_attn_varlen_func(
                q=q_packed,
                k=k_packed,
                v=v_packed,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                dropout_p=0.0,
                softmax_scale=kwargs.get("scale"),
                causal=False,
            )
            out = out.view(B, seqlen, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
        elif self.switti_attn_backend == "flex" and scale_ind is None:
            if self.caching:
                raise RuntimeError(
                    "SWITTI FlexAttention cannot be combined with KV caching."
                )
            if q.device.type != "cuda":
                raise RuntimeError(
                    "SWITTI FlexAttention requires CUDA tensors, got "
                    f"device={q.device}."
                )
            if any(tensor.dtype != torch.float32 for tensor in (q, k, v)):
                raise RuntimeError(
                    "SWITTI FlexAttention expects ProPE-transformed Q/K/V in "
                    "float32, got dtypes="
                    f"{(q.dtype, k.dtype, v.dtype)}."
                )
            if attn_mask is not None:
                raise RuntimeError(
                    "SWITTI FlexAttention received a dense attention mask; "
                    "the caller must skip dense mask allocation."
                )

            metadata_key = (
                "switti_flex_block_mask",
                N_views,
                tuple(tuple(int(dim) for dim in scale) for scale in scale_schedule),
                seqlen,
                q.device.type,
                q.device.index,
            )
            block_mask = None if prope_cache is None else prope_cache.get(metadata_key)
            if block_mask is None:
                block_mask = create_switti_block_mask(
                    block_scales=scale_schedule,
                    n_views=N_views,
                    sequence_length=seqlen,
                    device=q.device,
                )
                if prope_cache is not None:
                    prope_cache[metadata_key] = block_mask

            # Retain the post-ProPE FP32 tensors. This makes the candidate a
            # precision-matched comparison with the existing dense SDPA path
            # and avoids BF16 rounding of the unusually large ProPE logits.

            q_attention = q.to(projected_attention_dtype).contiguous()
            k_attention = k.to(projected_attention_dtype).contiguous()
            v_attention = v.to(projected_attention_dtype).contiguous()

            # q_attention = q.contiguous()
            # k_attention = k.contiguous()
            # v_attention = v.contiguous()
            
            
            # print(f"q_attention.dtype: {q_attention.dtype}")
            # print(f"k_attention.dtype: {k_attention.dtype}")
            # print(f"v_attention.dtype: {v_attention.dtype}")
            
            out = get_compiled_flex_attention()(
                q_attention,
                k_attention,
                v_attention,
                block_mask=block_mask,
                scale=kwargs.get("scale"),
                kernel_options=SWITTI_FLEX_KERNEL_OPTIONS,
            )
        else:
            # Global dense/current-scale Attention call.
            q_attention, k_attention, v_attention = q, k, v
            ##NOTE comment out for SA visualization:
            out = F.scaled_dot_product_attention(
                query=q_attention,
                key=k_attention,
                value=v_attention,
                attn_mask=attn_mask,
                **kwargs
            )

        ##NOTE uncomment for SA visualization: 
        # scale = 1.0 / math.sqrt(q.size(-1))
        # attn_weight = q @ k.transpose(-2, -1) * scale
        # attn_weight = torch.softmax(attn_weight, dim=-1)        
        # global GLOBAL_SAVED_SELF_ATTN
        # GLOBAL_SAVED_SELF_ATTN.append(attn_weight.detach().cpu())
        # out = attn_weight @ v

        # Apply output transform
        out = self._apply_transform(out, P_seq, coeffs_x, coeffs_y, inverse_rope=True)
        
        return out

    # Helper to apply the 3-part block diagonal transform efficiently
    def _apply_transform(self, x, mat_seq_or_cameras, coeffs_x, coeffs_y, inverse_rope=False):
        # Split features: [Half (Proj), Quarter (RoPE X), Quarter (RoPE Y)]

        B, H, L, D = x.shape

        # x: (B, H, L, D)
        d_half = D // 2
        d_quart = D // 4
        
        # split into projective/ropex/ropey
        x_proj, x_rope_x, x_rope_y = torch.split(x, [d_half, d_quart, d_quart], dim=-1)
        
        # if single_scale(i.e. inference)
        if (mat_seq_or_cameras.dim() == 4 and 
            mat_seq_or_cameras.shape[1] < L and 
            L % mat_seq_or_cameras.shape[1] == 0):

            # Fast Path: Broadcast over patches
            mat = mat_seq_or_cameras 
            cameras = mat.shape[1]
            patches_per_camera = L // cameras
            
            x_proj_c = x_proj.contiguous()
            x_proj_reshaped = x_proj_c.view(B, H, cameras, patches_per_camera, d_half // 4, 4)
            
            # Einsum: bcij (Mat) * bncpkj (Vec) -> bncpki
            proj_out = torch.einsum("bcij,bncpkj->bncpki", mat, x_proj_reshaped)
            x_proj_out = proj_out.reshape(B, H, L, d_half).contiguous()
        
        # if all scales(i.e. train)
        else:
            mat = mat_seq_or_cameras
            x_proj_c = x_proj.contiguous()
            x_proj_reshaped = x_proj_c.view(B, H, L, -1, 4)
            x_proj_out = torch.einsum("blij, bhlkj -> bhlki", mat, x_proj_reshaped).reshape(B, H, L, d_half).contiguous()
        
        # 2. Apply RoPE
        x_rope_x_out = self._rope_apply_coeffs(x_rope_x, coeffs_x, inverse=inverse_rope)
        x_rope_y_out = self._rope_apply_coeffs(x_rope_y, coeffs_y, inverse=inverse_rope)
        
        return torch.cat([x_proj_out, x_rope_x_out, x_rope_y_out], dim=-1)

    # NOTE: attn_bias_or_two_vector is None during inference
    def forward(self, 
                x, 
                attn_bias_or_two_vector: Union[torch.Tensor, Tuple[torch.IntTensor, torch.IntTensor]], 
                attn_fn=None,
                scale_schedule=None, 
                rope2d_freqs_grid=None, 
                scale_ind=None,
                poses=None,
                intrs=None,
                input_size=None):
        """
        :param (fp32) x: shaped (B or batch_size, L or seq_length, C or hidden_dim); if seq-parallel is used, the `L` dim would be shared
        :param (fp32) attn_bias_or_two_vector:
                if not using_flash:
                    a block-wise, lower-triangle matrix, like:
                    [[[[0, -, -, -, -, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]]
                    where 0 means visible and - means invisible (-inf)
                else:
                    a tuple of two 1-dim int vector (VAR_visible_kvlen, VAR_invisible_qlen)
        :return: shaped (B or batch_size, L or seq_length, C or hidden_dim); if seq-parallel is used, the `L` dim would be shared
        """
        # x: fp32
        B, L, C = x.shape

        # qkv: amp, bf16
        # qkv: [B,L,3,16,128] [2,3,3,16,128]
        qkv = F.linear(input=x, weight=self.mat_qkv.weight, bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias))).view(B, L, 3, self.num_heads, self.head_dim)  # BL3Hc
        
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
            
        if self.cos_attn:   # always True


            scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp() # 11H1 (flash), or 1H11 (not flash)
            q,k = apply_cos_attn_fused(q,k,scale_mul,v.dtype)

            # q
            # q = F.normalize(q, dim=-1, eps=1e-12).mul(scale_mul).to(v.dtype).contiguous()   # fp32
            # k = F.normalize(k, dim=-1, eps=1e-12).to(v.dtype).contiguous()                  # fp32
            q = q.contiguous()      # bf16
            k = k.contiguous()      # bf16
            v = v.contiguous()      # bf16                                             # bf16
        else:   # be contiguous, to make kernel happy
            q = q.contiguous()      # bf16
            k = k.contiguous()      # bf16
            v = v.contiguous()      # bf16

        # print(f"q.dtype: {q.dtype}")
        # print(f"k.dtype: {k.dtype}")
        # print(f"v.dtype: {v.dtype}")

        oup = self.prope_dot_product_vectorized(q,k,v,
                                                poses_c2w=poses,
                                                Ks=intrs,
                                                scale_schedule=scale_schedule,
                                                scale_ind=scale_ind,
                                                attn_mask=attn_bias_or_two_vector,
                                                prope_cache=rope2d_freqs_grid,
                                                scale=self.scale)
        oup = oup.transpose(1,2).reshape(B,L,C)

        return self.proj_drop(self.proj(oup))
    
    def extra_repr(self) -> str:
        tail = ''
        return (
            f'using_flash={self.using_flash}, switti_attn_backend={self.switti_attn_backend}, '
            f'tau={self.tau}, cos_attn={self.cos_attn}{tail}'
        )

    # --- Helper Static Methods (Inlined for speed/simplicity) ---
    @staticmethod
    def _rope_precompute_coeffs(positions, freq_base, freq_scale, feat_dim):
        num_freqs = feat_dim // 2
        freqs = freq_scale * (freq_base ** (-torch.arange(num_freqs, device=positions.device) / num_freqs))
        angles = positions[:, None] * freqs[None, :] # (Seq, Freqs)
        # Reshape for broadcasting: (1, 1, Seq, Freqs)
        angles = angles.view(1, 1, positions.shape[0], num_freqs)
        return torch.cos(angles), torch.sin(angles)

    @staticmethod
    def _rope_apply_coeffs(feats, coeffs, inverse=False):
        cos, sin = coeffs
             
        x_in = feats[..., : feats.shape[-1] // 2]
        y_in = feats[..., feats.shape[-1] // 2 :]
        
        if not inverse:
            return torch.cat([cos * x_in + sin * y_in, -sin * x_in + cos * y_in], dim=-1)
        else:
            return torch.cat([cos * x_in - sin * y_in, sin * x_in + cos * y_in], dim=-1)


class SelfAttentionPropeOptimized2(nn.Module):
    def __init__(
        self, embed_dim=768, num_heads=12,
        proj_drop=0., tau=1, cos_attn=False, customized_flash_attn=True, use_flex_attn=False, 
        batch_size=2, pad_to_multiplier=1, rope2d_normalized_by_hw=0, N_views=1
    ):
        """
        Implements PRope
        :param embed_dim: model's width
        :param num_heads: num heads of multi-head attention
        :param proj_drop: always 0 for testing
        :param tau: always 1
        :param cos_attn: always True: during attention, q and k will be L2-normalized and scaled by a head-wise learnable parameter self.scale_mul_1H11
        :param customized_flash_attn:
        """
        super().__init__()
        assert embed_dim % num_heads == 0
        self.using_flash = customized_flash_attn
        
        self.num_heads, self.head_dim = num_heads, embed_dim // num_heads
        self.tau, self.cos_attn = tau, cos_attn
        if self.cos_attn:
            self.scale = 1
            size = (1, 1, self.num_heads, 1) if self.using_flash else (1, self.num_heads, 1, 1)
            # size: 11H1 or 1H11
            self.scale_mul_1H11 = nn.Parameter(torch.full(size=size, fill_value=4.0).log(), requires_grad=True)
            self.max_scale_mul = torch.log(torch.tensor(100)).item()
        else:
            self.scale = 1 / math.sqrt(self.head_dim) / self.tau
        
        self.mat_qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.q_bias, self.v_bias = nn.Parameter(torch.zeros(embed_dim)), nn.Parameter(torch.zeros(embed_dim))
        self.register_buffer('zero_k_bias', torch.zeros(embed_dim))
        
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = get_dropout_layer(proj_drop)
        
        self.caching = False    # kv caching: only used during inference
        self.cached_k = None    # kv caching: only used during inference
        self.cached_v = None    # kv caching: only used during inference

        self.use_flex_attn = use_flex_attn
        self.pad_to_multiplier = pad_to_multiplier

        self.rope2d_normalized_by_hw = rope2d_normalized_by_hw
    
    def kv_caching(self, enable: bool): # kv caching: only used during inference
        self.caching = enable
        self.cached_k = None
        self.cached_v = None

    def prope_dot_product_vectorized(
        self, q, k, v,
        poses_c2w, Ks,
        scale_schedule,
        scale_ind=None,
        attn_mask=None,
        prope_cache=None,
        **kwargs
    ):
        """
        Vectorized implementation that avoids Python loops over sequence chunks.
        """
        B, num_heads, seqlen, head_dim = q.shape
        N_views = poses_c2w.shape[1]
        device = q.device
        viewmats = torch.linalg.inv(poses_c2w)


        cache_key = f"tgt_scale_{scale_ind}" if scale_ind is not None else "tgt_full"
        cached_geom = None
        if prope_cache is not None:
            cached_geom = prope_cache.get(cache_key)

            # print(f"prope_cache.keys(): {prope_cache.keys()}")

        if cached_geom is None:
            # 1. Select schedule (Single step vs Full sequence)
            if scale_ind is not None:
                current_schedule = [scale_schedule[scale_ind]]
            else:
                current_schedule = scale_schedule

            # 2. Compute Camera Matrices
            P, P_T, P_inv = get_prope_matrices(poses_c2w=poses_c2w, intrs=None)
            N_views = poses_c2w.shape[1]

            pos_x_list, pos_y_list = [], []
            P_T_expanded_list, P_inv_expanded_list, P_expanded_list = [], [], []

            for _, px, py in current_schedule:
                num_repeats = px * py
                
                x_grid = torch.arange(px, device=device).repeat(py * N_views)
                y_grid = torch.arange(py, device=device).repeat_interleave(px).repeat(N_views)

                pos_x_list.append(x_grid)
                pos_y_list.append(y_grid)
                
                P_T_expanded_list.append(P_T.repeat_interleave(num_repeats, dim=1))
                P_inv_expanded_list.append(P_inv.repeat_interleave(num_repeats, dim=1))
                P_expanded_list.append(P.repeat_interleave(num_repeats, dim=1))

            pos_x_total = torch.cat(pos_x_list, dim=0)
            pos_y_total = torch.cat(pos_y_list, dim=0)
            
            P_T_seq = torch.cat(P_T_expanded_list, dim=1) 
            P_inv_seq = torch.cat(P_inv_expanded_list, dim=1)
            P_seq = torch.cat(P_expanded_list, dim=1)

            coeffs_x = self._rope_precompute_coeffs(pos_x_total, 100.0, 1.0, head_dim // 4)
            coeffs_y = self._rope_precompute_coeffs(pos_y_total, 100.0, 1.0, head_dim // 4)

            cached_geom = (P_T_seq, P_inv_seq, P_seq, coeffs_x, coeffs_y)

            if prope_cache is not None:
                prope_cache[cache_key] = cached_geom


        if scale_ind is not None:
            scale_schedule = [scale_schedule[scale_ind]]

        P_T_seq, P_inv_seq, P_seq, coeffs_x, coeffs_y = cached_geom

        # Apply transforms to Q, K, V
        q = self._apply_transform(q, P_T_seq, coeffs_x, coeffs_y, inverse_rope=False)
        k = self._apply_transform(k, P_inv_seq, coeffs_x, coeffs_y, inverse_rope=False)
        v = self._apply_transform(v, P_inv_seq, coeffs_x, coeffs_y, inverse_rope=False)


        # --- ATTENTION PHASE ---
        if self.caching:
            if self.cached_k is None:
                self.cached_k = k
                self.cached_v = v
            else:
                k = self.cached_k = torch.cat([self.cached_k, k], dim=2)
                v = self.cached_v = torch.cat([self.cached_v, v], dim=2)
        
        # Global Attention call
        out = F.scaled_dot_product_attention(
            query=q, 
            key=k, 
            value=v, 
            attn_mask=attn_mask, 
            # scale=self.scale, 
            **kwargs
        )

        # Apply output transform
        out = self._apply_transform(out, P_seq, coeffs_x, coeffs_y, inverse_rope=True)
        
        return out

    # Helper to apply the 3-part block diagonal transform efficiently
    def _apply_transform(self, x, mat_seq, coeffs_x, coeffs_y, inverse_rope=False):
        # Split features: [Half (Proj), Quarter (RoPE X), Quarter (RoPE Y)]

        B, H, L, D = x.shape

        # x: (B, H, L, D)
        d_half = D // 2
        d_quart = D // 4
        
        # split into projective/ropex/ropey
        x_proj, x_rope_x, x_rope_y = torch.split(x, [d_half, d_quart, d_quart], dim=-1)
        
        # 1. Apply Projection Matrix (Batched MatMul)
        # x_proj: (B, H, L, D/2) -> reshape to (B, H, L, D/8, 4)
        # mat_seq: (B, L, 4, 4) -> broadcast over H and D/8
        # Logic: We multiply the last dim 4 by the 4x4 matrix
        x_proj_reshaped = x_proj.view(B, H, L, -1, 4)
        # Einsum: b=batch, h=head, l=seq, k=chunk_idx, i/j=matrix dims
        # mat_seq is (b, l, i, j)
        x_proj_out = torch.einsum("blij, bhlkj -> bhlki", mat_seq, x_proj_reshaped)
        x_proj_out = x_proj_out.reshape(B, H, L, d_half)
        
        # 2. Apply RoPE
        x_rope_x_out = self._rope_apply_coeffs(x_rope_x, coeffs_x, inverse=inverse_rope)
        x_rope_y_out = self._rope_apply_coeffs(x_rope_y, coeffs_y, inverse=inverse_rope)
        
        return torch.cat([x_proj_out, x_rope_x_out, x_rope_y_out], dim=-1)

    # NOTE: attn_bias_or_two_vector is None during inference
    def forward(self, 
                x, 
                attn_bias_or_two_vector: Union[torch.Tensor, Tuple[torch.IntTensor, torch.IntTensor]], 
                attn_fn=None, 
                scale_schedule=None, 
                rope2d_freqs_grid=None, 
                scale_ind=None,
                poses=None,
                intrs=None,
                input_size=None):
        """
        :param (fp32) x: shaped (B or batch_size, L or seq_length, C or hidden_dim); if seq-parallel is used, the `L` dim would be shared
        :param (fp32) attn_bias_or_two_vector:
                if not using_flash:
                    a block-wise, lower-triangle matrix, like:
                    [[[[0, -, -, -, -, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]]
                    where 0 means visible and - means invisible (-inf)
                else:
                    a tuple of two 1-dim int vector (VAR_visible_kvlen, VAR_invisible_qlen)
        :return: shaped (B or batch_size, L or seq_length, C or hidden_dim); if seq-parallel is used, the `L` dim would be shared
        """
        # x: fp32
        B, L, C = x.shape

        # qkv: amp, bf16
        # qkv: [B,L,3,16,128] [2,3,3,16,128]
        qkv = F.linear(input=x, weight=self.mat_qkv.weight, bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias))).view(B, L, 3, self.num_heads, self.head_dim)  # BL3Hc
        
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
            
        if self.cos_attn:   # always True
            scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp() # 11H1 (flash), or 1H11 (not flash)
            q = F.normalize(q, dim=-1, eps=1e-12).mul(scale_mul).contiguous()   # fp32
            k = F.normalize(k, dim=-1, eps=1e-12).contiguous()                  # fp32
            v = v.contiguous()                                                  # bf16
        else:   # be contiguous, to make kernel happy
            q = q.contiguous()      # bf16
            k = k.contiguous()      # bf16
            v = v.contiguous()      # bf16

        oup = self.prope_dot_product_vectorized(q,k,v,
                                                poses_c2w=poses,
                                                Ks=intrs,
                                                scale_schedule=scale_schedule,
                                                scale_ind=scale_ind,
                                                attn_mask=attn_bias_or_two_vector,
                                                prope_cache=rope2d_freqs_grid)
        oup = oup.transpose(1,2).reshape(B,L,C)

        return self.proj_drop(self.proj(oup))
    
    def extra_repr(self) -> str:
        tail = ''
        return f'using_flash={self.using_flash}, tau={self.tau}, cos_attn={self.cos_attn}{tail}'

    # --- Helper Static Methods (Inlined for speed/simplicity) ---
    @staticmethod
    def _rope_precompute_coeffs(positions, freq_base, freq_scale, feat_dim):
        num_freqs = feat_dim // 2
        freqs = freq_scale * (freq_base ** (-torch.arange(num_freqs, device=positions.device) / num_freqs))
        angles = positions[:, None] * freqs[None, :] # (Seq, Freqs)
        # Reshape for broadcasting: (1, 1, Seq, Freqs)
        angles = angles.view(1, 1, positions.shape[0], num_freqs)
        return torch.cos(angles), torch.sin(angles)

    @staticmethod
    def _rope_apply_coeffs(feats, coeffs, inverse=False):
        cos, sin = coeffs
        # Handle broadcasting if coeffs are smaller than feats
        if cos.shape[2] != feats.shape[2]:
             # This happens if we cached coeffs but seq length grew. 
             # In this specific vectorized impl, we usually regenerate coeffs, so this is just a safeguard.
             pass 

        # print(f"[sa] feats.shape: {feats.shape}")
        # print(f"[sa] cos.shape: {cos.shape}")
        # print(f"[sa] sin.shape: {sin.shape}")
             
        x_in = feats[..., : feats.shape[-1] // 2]
        y_in = feats[..., feats.shape[-1] // 2 :]
        
        if not inverse:
            return torch.cat([cos * x_in + sin * y_in, -sin * x_in + cos * y_in], dim=-1)
        else:
            return torch.cat([cos * x_in - sin * y_in, sin * x_in + cos * y_in], dim=-1)


def _sample_depth_map(
    pixel_depths: torch.Tensor,  # (batch, cameras, image_height, image_width, 1)
    patches_x: int,
    patches_y: int,
    offsets: List[Tuple[float, float]],
) -> torch.Tensor:
    # assert pixel_depths.ndim == 5 and pixel_depths.shape[-1] == 1
    batch, cameras, image_height, image_width, _ = pixel_depths.shape
    # if image_width % patches_x != 0 or image_height % patches_y != 0:
    #     raise ValueError("Image resolution must be divisible by the patch grid dimensions.")

    patch_w = image_width // patches_x
    patch_h = image_height // patches_y
    device = pixel_depths.device
    dtype = pixel_depths.dtype
    pixel_depths = torch.clamp(pixel_depths, min=1e-4, max=MAX_DEPTH)

    num_offsets = len(offsets)

    # Grid over patch indices; match meshgrid ordering used in _get_point_coords.
    x_idx = torch.arange(patches_x, device=device, dtype=dtype)
    y_idx = torch.arange(patches_y, device=device, dtype=dtype)
    grid_x, grid_y = torch.meshgrid(x_idx, y_idx, indexing="xy")
    base_x = grid_x * float(patch_w)
    base_y = grid_y * float(patch_h)

    depths_flat = pixel_depths.reshape(batch * cameras, 1, image_height, image_width)
    samples = []
    for ox, oy in offsets:
        ox_t = torch.as_tensor(ox, device=device, dtype=dtype)
        oy_t = torch.as_tensor(oy, device=device, dtype=dtype)
        sample_x = base_x + ox_t * patch_w
        sample_y = base_y + oy_t * patch_h

        # Convert to normalized grid coordinates in [-1, 1].
        x_norm = sample_x / image_width * 2.0 - 1.0
        y_norm = sample_y / image_height * 2.0 - 1.0
        grid = torch.stack((x_norm, y_norm), dim=-1).permute(1, 0, 2)
        grid = grid.unsqueeze(0).expand(batch * cameras, -1, -1, -1)
        sampled = F.grid_sample(
            depths_flat,
            grid,
            mode="nearest",
            padding_mode="border",
            align_corners=False,
        )
        samples.append(sampled.permute(0, 1, 3, 2))

    sampled_depths = torch.stack(samples, dim=-1)
    sampled_depths = sampled_depths.reshape(batch, cameras, patches_x * patches_y, num_offsets, 1)

    return sampled_depths

def _prepare_depths(
    predicted_d: Optional[torch.Tensor],  # (batch, seqlen, 1 or 2)
    context_depths: Optional[torch.Tensor],  # (batch, cameras, image_height, image_width, 1)
    depth_type: str = 'none',
    batch: int = 1,
    num_cameras: int = 2,
    num_patches: int = 1024,
    patches_x: int = 32,
    patches_y: int = 32,
    num_rays_per_patch: int = 3,
    offsets = [[0.0, 0.0]],
):
    predicted_d = predicted_d.reshape(batch, num_cameras, num_patches, 1, 2)
    predicted_logd = predicted_d[..., 0:1]
    # predicted_sigma = torch.exp(predicted_d[..., 1:2])
    predicted_sigma = predicted_d[..., 1:2]

    predicted_d1 = torch.exp(torch.clamp(predicted_logd - predicted_sigma, max=MAX_LOG_DEPTH))
    predicted_d2 = torch.exp(torch.clamp(predicted_logd + predicted_sigma, max=MAX_LOG_DEPTH))

    # print(f"predicted_d1.shape: {predicted_d1.shape}")
    # print(f"predicted_d2.shape: {predicted_d2.shape}")

    depths = torch.stack([predicted_d1, predicted_d2], dim=0)  # (2, batch, num_cameras, num_patches, 1)
    depths = depths.expand(-1, -1, -1, -1, num_rays_per_patch, -1)
    
    if torch.isnan(depths).any() or torch.isinf(depths).any():
        raise ValueError("NaN/inf values found in predicted depths.")
    

    if 'known' in depth_type:
        num_contexts = context_depths.shape[1]
        context_depths_sampled = _sample_depth_map(
            context_depths, patches_x, patches_y, offsets=offsets
        ).unsqueeze(0).expand(2, -1, -1, -1, -1, -1)
        # (2, batch, num_cameras, num_patches, num_rays_per_patch, 1)
        depths = depths.contiguous()
        depths[:, :, :num_contexts] = context_depths_sampled

    return depths


def _get_cam_centers(
    c2ws: torch.Tensor,  # (batch, num_cameras, 4, 4)
    num_patches: int,
) -> torch.Tensor:
    # return the camera centers in homogenous coordinates
    device = c2ws.device
    batches = c2ws.shape[0]
    num_cameras = c2ws.shape[1]

    # print(f"c2ws.shape: {c2ws.shape}")
    cam_centers = c2ws[:, :, :, 3]  # (batch, num_cameras, 4)
    cam_centers = cam_centers.view(batches, num_cameras, 1, 4).expand(batches, num_cameras, num_patches, 4)
    return cam_centers

def _get_point_coords(
        P_inv: torch.Tensor,  # (batch, num_cameras, 4, 4)
        patches_x: int, 
        patches_y: int, 
        offsets: List[Tuple[float, float]],
        depths: torch.Tensor = None,  # (2, batch, num_cameras, num_patches, num_rays_per_patch, 1)
    ) -> torch.Tensor:
    # return the pixel space 3d homogenous coordinates
    device = P_inv.device
    batches = P_inv.shape[0]
    num_cameras = P_inv.shape[1]
    num_patches = patches_x * patches_y
    num_rays_per_patch = len(offsets)
    u_base, v_base = torch.meshgrid(
        torch.arange(patches_x, device=device),
        torch.arange(patches_y, device=device),
        indexing="xy",
    )  # [H, W]

    coords = []
    for offset in offsets:
        u = ((u_base + offset[0]) / patches_x) - 0.5 #Since we assume normalized K where px=py=0
        v = ((v_base + offset[1]) / patches_y) - 0.5
        coords.append(torch.stack([u, v], dim=-1).reshape(-1, 2))  # [num_patches, 2]
    coords = torch.stack(coords, dim=1) # (num_patches, num_rays_per_patch, 2)
    # assert coords.shape == (num_patches, num_rays_per_patch, 2)
    coords = coords.view(1, 1, num_patches, num_rays_per_patch, 2).expand(batches, num_cameras, -1, -1, -1) 
    # (batch, num_cameras, num_patches, num_rays_per_patch, 2)

    if depths is not None:
        # depths = torch.clip(depths, min=1e-2)
        # gives [u, v, 1, 1/d]
        depths = torch.clamp(depths, min=1e-2, max=MAX_DEPTH)
        disparity = 1 / depths
        if depths.ndim == 5:
            assert depths.shape == (batches, num_cameras, num_patches, num_rays_per_patch, 1)
            coords_4d = torch.cat([coords, torch.ones_like(disparity), disparity], dim=-1)  # (batch, num_cameras, num_patches, num_rays_per_patch, 4)
            coords_4d = torch.einsum("bcij,bcprj->bcpri", P_inv, coords_4d)
        elif depths.ndim == 6: # when two depths along ray is provided
            assert depths.shape == (2, batches, num_cameras, num_patches, num_rays_per_patch, 1)
            coords = coords.unsqueeze(0).expand(2, -1, -1, -1, -1, -1)
            coords_4d = torch.cat([coords, torch.ones_like(disparity), disparity], dim=-1)  # (2, batch, num_cameras, num_patches, num_rays_per_patch, 4)
            coords_4d = torch.einsum("bcij,ebcprj->ebcpri", P_inv, coords_4d)
    else:
        # if no depth provided, gives points at infinity (directions)
        depths = torch.ones_like(coords[..., :1])
        scales = torch.zeros_like(coords[..., :1])

        coords_4d = torch.cat([coords, depths, scales], dim=-1)  # (batch, num_cameras, num_patches, num_rays_per_patch, 4)
        coords_4d = torch.einsum("bcij,bcprj->bcpri", P_inv, coords_4d)
    
    return coords_4d

@torch.compile
def _prepare_rope_coeff_uniformd(
    positions: dict[str, torch.Tensor], # (batch, num_cameras, num_patches, coord_dim)
    num_freqs: int,
    freq_base: float,
    batch: int,
    num_cameras: int,
    num_patches: int,
):
    coord_dim = 0
    for key, value in positions.items():
        coord_dim += value.shape[-1]

        device = value.device
    # print(f"positions: {positions}")
    cosine_list = []
    sine_list = []
    for pos_name, pos in positions.items():
        # print(f"pos_name: {pos_name} | pos.shape: {pos.shape}")
        if pos_name in ['p0', 'pd_3d']:
            max_period = 1.0 * 4
        elif pos_name in ['pinf_dir', 'pd_dir', 'p0_dir']:
            max_period = 2.0 * 4
        elif pos_name in ['pd_disparity', 'p0_disparity']:
            max_period = 20.0 * 4
            # pos = torch.clamp(pos, min=-20.0, max=20.0)
            pos = torch.clamp(pos, min=0.0, max=20.0)
        elif pos_name in ['pd_depth', 'p0_depth']:
            max_period = MAX_D_F * 2 * 4
            pos = torch.clamp(pos, min=-MAX_D_F, max=MAX_D_F)
        elif pos_name in ['pd_asinh_depth', 'p0_asinh_depth']:
            max_period = MAX_ASINH_D_F * 2 * 4
            pos = torch.clamp(pos, min=-MAX_ASINH_D_F, max=MAX_ASINH_D_F)
        else: 
            raise ValueError(f"Unknown position name: {pos_name}")
        
        min_period = max_period / (freq_base ** (num_freqs - 1))
        freqs = _get_frequency(num_freqs,
                                max_period=max_period,
                                min_period=min_period).to(device)
        rope_angle = torch.einsum("f,bcpd->bcpfd", freqs, pos)

        # print(f"freqs.shape: {freqs.shape}")
        # print(f"rope_angle.shape: {rope_angle.shape}")

        # In _prepare_rope_coeff_uniformd, before the loop logic:

        # print(f"rope_angle.dtype: {rope_angle.dtype}")
        rope_angle = rope_angle.to(torch.float32)

        # print(f"rope_angle: {rope_angle}")
        if rope_angle.shape[0] == batch * 2:
            rope_angles1 = rope_angle[:batch]
            rope_angles2 = rope_angle[batch:]

            # same_mask = torch.isclose(rope_angles1, rope_angles2, atol=1e-2, rtol=0)
            same_mask = torch.isclose(rope_angles1, rope_angles2, atol=1e-3, rtol=1e-3)

            cosine1 = torch.cos(rope_angles1)
            cosine2 = torch.cos(rope_angles2)
            sine1 = torch.sin(rope_angles1)
            sine2 = torch.sin(rope_angles2)
            delta = rope_angles2 - rope_angles1
            delta_safe = torch.where(same_mask, torch.ones_like(delta), delta)
            E_cosine = (sine2 - sine1) / delta_safe
            E_sine = (cosine1 - cosine2) / delta_safe

            cosine_final = torch.where(same_mask, cosine1, E_cosine)
            sine_final = torch.where(same_mask, sine1, E_sine)

            # print(f"cosine_final.shape: {cosine_final.shape}")
            # print(f"cosine_final: {cosine_final}")
            # print(f"sine_final.shape: {sine_final.shape}")

            assert cosine_final.abs().max() <= 1.0 + 1e-2
            assert sine_final.abs().max() <= 1.0 + 1e-2
            cosine_list.append(cosine_final)
            sine_list.append(sine_final)

        elif rope_angle.shape[0] == batch:
            cosine_final = torch.cos(rope_angle)
            sine_final = torch.sin(rope_angle)
            cosine_list.append(cosine_final)
            sine_list.append(sine_final)
        else:
            raise ValueError(f"Unexpected rope_angle batch size. rope_angle shape: {rope_angle.shape}")

    #     print(f"cosine_final.shape: {cosine_final.shape}")
    #     print(f"sine_final.shape: {sine_final.shape}")
        
    # print(f"len(cosine_list): {len(cosine_list)}")
    # print(f"len(sine_list): {len(sine_list)}")
    # print(f"num_freqs: {num_freqs}")
    # print(f"coord_dim: {coord_dim}")

    # print(f"torch.cat(cosine_list, dim=-1).shape: {torch.cat(cosine_list, dim=-1).shape}")
    # print(f"torch.cat(sine_list, dim=-1).shape: {torch.cat(sine_list, dim=-1).shape}")

    cosine_out = torch.cat(cosine_list, dim=-1).reshape(batch, num_cameras * num_patches, num_freqs * coord_dim).contiguous()
    sine_out = torch.cat(sine_list, dim=-1).reshape(batch, num_cameras * num_patches, num_freqs * coord_dim).contiguous()
    return cosine_out, sine_out


def _get_frequency(
    num_freqs: int,
    max_period: float,
    min_period: float,
):
    log_min_frequency = torch.log(torch.tensor(2 * torch.pi / max_period))
    log_max_frequency = torch.log(torch.tensor(2 * torch.pi / min_period))
    log_freqs = torch.linspace(
        log_min_frequency,
        log_max_frequency,
        num_freqs,
    )
    freqs = torch.exp(log_freqs)
    # ratio = freqs[1] / freqs[0]
    # print(f"max/min: {max_period}, {min_period}, Frequency ratio: {ratio:.4f}, freqs: {freqs}")
    return freqs

def _transform_to_query_frame(
    points_world: torch.Tensor,  # (2, batch, num_cameras, num_patches, num_rays_per_patch, 4)
    P: torch.Tensor,  # (batch, 4, 4)
    w2c: torch.Tensor,  # (batch, 4, 4)
    transform_type: str = 'pj', # pj or 3d, default: 3d
    denc_type: str = 'd', # inv_d, d, asinh_d, default: d
    norm_by: str = 'w' # length or w
):
    """
    By default: transform_type == 3d.
    Applied w2c transform(correspoding to query ) to points world(camera centers, i.e. translation)
    """

    if transform_type == '3d':
        points_cam = torch.einsum("bij,...bcprj->...bcpri", w2c, points_world) # (batch, num_cameras, num_patches, 4)
        if norm_by == 'w':
            points_cam = points_cam / torch.clamp(points_cam[..., -1:], min=1e-4)  # normalize
        elif norm_by == 'length':
            points_cam = points_cam / (torch.norm(points_cam[..., :3], dim=-1, keepdim=True) + 1e-6)
        return points_cam[..., :3]
    
    elif transform_type == 'pj':
        # points at depth
        points_cam = torch.einsum("bij,...bcprj->...bcpri", P, points_world)
        safe_abs = torch.sqrt(points_cam[..., 2:3].pow(2) + 1e-9)
        z = torch.clamp(safe_abs, min=1e-4)
        # z = _clamp_zero(points_cam[..., 2:3], min_abs_value=1e-4)
        w = torch.clamp(points_cam[..., -1:], min=1e-4)
        pd_dir = points_cam[..., :3] / (torch.norm(points_cam[..., :3], dim=-1, keepdim=True) + 1e-6)
        if denc_type == 'inv_d':
            pd_disparity = w / z
            return pd_dir, pd_disparity
        elif denc_type == 'd':
            pd_depth = z / w
            return pd_dir, pd_depth
        elif denc_type == 'asinh_d':
            pd_depth = z / w
            pd_depth = torch.asinh(pd_depth)
            return pd_dir, pd_depth

# @torch.compile
def _apply_rope_coeffs(
    feats: torch.Tensor, # (batch, num_heads, seqlen, feat_dim)
    cos:  torch.Tensor, # (batch, seqlen, feat_dim // 2)
    sin:    torch.Tensor, # (batch, seqlen, feat_dim // 2)
    inverse: bool = False,
    interleaved: bool = False,
):
    """Apply ROPE coefficients to an input array."""
    (batch, num_heads, seqlen, feat_dim) = feats.shape
    # print(f"feats.shape: {feats.shape}")
    # print(f"cos.shape: {cos.shape}")
    assert cos.shape == (batch, seqlen, feat_dim // 2)
    assert sin.shape == (batch, seqlen, feat_dim // 2)
    cos = cos.unsqueeze(1)  # (batch, 1, seqlen, feat_dim // 2)
    sin = sin.unsqueeze(1)      # (batch, 1, seqlen, feat_dim // 2)

    if interleaved:
        x1 = feats[..., 0::2]
        x2 = feats[..., 1::2]
    else:
        x1, x2 = torch.chunk(feats, 2, dim=-1)  # each is (batch, num_heads, seqlen, feat_dim // 2)
    

    if inverse:
        x_rotated_1 = x1 * cos + x2 * sin
        x_rotated_2 = -x1 * sin + x2 * cos
    else:
        x_rotated_1 = x1 * cos - x2 * sin
        x_rotated_2 = x1 * sin + x2 * cos

    if interleaved:
        out = torch.empty_like(feats)
        out[..., 0::2] = x_rotated_1
        out[..., 1::2] = x_rotated_2
    else:
        out = torch.cat([x_rotated_1, x_rotated_2], dim=-1)

    assert out.shape == feats.shape, "Input/output shapes should match."
    return out


class SelfAttentionRayRope(nn.Module):
    def __init__(
        self, embed_dim=768, num_heads=12,
        proj_drop=0., tau=1, cos_attn=False, customized_flash_attn=True, use_flex_attn=False, 
        batch_size=2, pad_to_multiplier=1, rope2d_normalized_by_hw=0, N_views=1
    ):
        """
        Implements PRope
        :param embed_dim: model's width
        :param num_heads: num heads of multi-head attention
        :param proj_drop: always 0 for testing
        :param tau: always 1
        :param cos_attn: always True: during attention, q and k will be L2-normalized and scaled by a head-wise learnable parameter self.scale_mul_1H11
        :param customized_flash_attn:
        """
        super().__init__()
        assert embed_dim % num_heads == 0
        self.using_flash = customized_flash_attn
        
        self.num_heads, self.head_dim = num_heads, embed_dim // num_heads
        
        self.mat_qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.q_bias, self.v_bias = nn.Parameter(torch.zeros(embed_dim)), nn.Parameter(torch.zeros(embed_dim))
        self.register_buffer('zero_k_bias', torch.zeros(embed_dim))
        
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = get_dropout_layer(proj_drop)
        
        self.caching = False    # kv caching: only used during inference
        self.cached_k = None    # kv caching: only used during inference
        self.cached_v = None    # kv caching: only used during inference

        self.cached_k_per_view = None
        self.cached_v_per_view = None

        self.use_pd = True
        self.pd_type = "pj"

        self.use_p0 = True
        self.p0_type = "3d"

        self.pinf_type = "none"
        self.use_pinf = False

        init_depth = 0.0
        init_sigma = 3.0
        self.d_proj_weight = nn.Parameter(torch.zeros((2, embed_dim)))
        self.d_proj_bias = torch.nn.Parameter(torch.tensor([init_depth, init_sigma]))

        self.patches_x = 16
        self.patches_y = 16
        self.num_patches = self.patches_x * self.patches_y

        self.num_rays_per_patch = 3
        self.rope_coord_dim = 3 * self.use_p0 + self.num_rays_per_patch * 3 * (int(self.use_pd) + int(self.use_pinf))
        self.rope_mat_dim = 2 * self.rope_coord_dim
        self.num_rope_freqs = self.head_dim // self.rope_mat_dim

        self.offsets = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        self.depth_type = "predict_dsig"
        self.denc_type = "d"

        self.freq_base = 3.0
        self.apply_vo = True

    
    def kv_caching(self, enable: bool): # kv caching: only used during inference
        self.caching = enable
        self.cached_k = None
        self.cached_v = None

        self.cached_k_per_view = None
        self.cached_v_per_view = None

    # NOTE: attn_bias_or_two_vector is None during inference
    def forward(self, 
                x, 
                attn_bias_or_two_vector: Union[torch.Tensor, Tuple[torch.IntTensor, torch.IntTensor]], 
                attn_fn=None, 
                scale_schedule=None, 
                rope2d_freqs_grid=None, 
                scale_ind=None,
                poses=None,
                intrs=None,
                input_size=None):
        """
        :param (fp32) x: shaped (B or batch_size, L or seq_length, C or hidden_dim); if seq-parallel is used, the `L` dim would be shared
        :param (fp32) attn_bias_or_two_vector:
                if not using_flash:
                    a block-wise, lower-triangle matrix, like:
                    [[[[0, -, -, -, -, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]]
                    where 0 means visible and - means invisible (-inf)
                else:
                    a tuple of two 1-dim int vector (VAR_visible_kvlen, VAR_invisible_qlen)
        :return: shaped (B or batch_size, L or seq_length, C or hidden_dim); if seq-parallel is used, the `L` dim would be shared
        """
        # x: fp32
        B, L, C = x.shape
        N_views = poses.shape[1]

        # raw_d: [B, N_views*N_tokens_per_view, 2]. 2:depth+confidence
        raw_d = F.linear(x, self.d_proj_weight, self.d_proj_bias)


        # qkv: amp, bf16
        # qkv: [B,L,3,16,128] [2,3,3,16,128]
        qkv = F.linear(input=x, weight=self.mat_qkv.weight, bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias))).view(B, L, 3, self.num_heads, self.head_dim)  # BL3Hc
        
        # [B,L,3, H, D//H] -> 3x [B,H,L,D//H]
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

        # print(f"[SA]q.shape: {q.shape}")
        # print(f"[SA]k.shape: {k.shape}")
        # print(f"[SA]v.shape: {v.shape}")
        # print(f"[SA]raw_d.shape: {raw_d.shape}")
        

        c2ws = poses  # [B, N, 4, 4] per camera
        Ks = intrs  # [B, N, 3, 3] per camera
        viewmats = torch.inverse(c2ws)

        if scale_ind is not None:
            scale_schedule = [scale_schedule[scale_ind]]

        token_schedule = [h_i*w_i  for _,h_i,w_i in scale_schedule]

        # print(f"N_views: {N_views}")
        
        if self.caching and self.cached_k_per_view is None:
            self.cached_k_per_view = [None] * N_views
            self.cached_v_per_view = [None] * N_views

        tok_start = 0
        out_final = []

        for i in range(len(token_schedule)):

            num_patches_i = token_schedule[i]
            tok_end = tok_start + N_views*num_patches_i

            self._precompute_and_cache_apply_fns(
                        w2cs=viewmats, 
                        Ks=Ks,
                        num_patches=num_patches_i
                    )

            # q,k,v: [B,H,L,D//H]
            # sdpa_fn call
            apply_fn_q, all_apply_fns_kv, apply_fn_o = self._prepare_apply_fns(
                num_cameras=N_views,
                patches_x=scale_schedule[i][-2],
                patches_y=scale_schedule[i][-1],
                predicted_d=raw_d[:,tok_start:tok_end],
                
                # positions_collector=positions_debug,
            )

            q_scale = q[:,:,tok_start:tok_end]
            k_scale = k[:,:,tok_start:tok_end]
            v_scale = v[:,:,tok_start:tok_end]

            q_scale = apply_fn_q(q_scale)
            out_scale = torch.zeros_like(q_scale)

            # q.shape: torch.Size([2, 16, 3, 128])
            # cos_Q.shape: torch.Size([2, 3, 60])
            # sin_Q.shape: torch.Size([2, 3, 60])
            # print(f"len(all_apply_fns_kv)")

            
            for cam_idx, apply_fn_kv in enumerate(all_apply_fns_kv):
                k_view_new = apply_fn_kv(k_scale)
                v_view_new = apply_fn_kv(v_scale)

                if self.caching:

                    k_cache = self.cached_k_per_view[cam_idx]
                    v_cache = self.cached_v_per_view[cam_idx]

                    if k_cache is None:
                        self.cached_k_per_view[cam_idx] = k_view_new
                        self.cached_v_per_view[cam_idx] = v_view_new
                        k_view_total = k_view_new
                        v_view_total = v_view_new
                    else:
                        self.cached_k_per_view[cam_idx] = torch.cat([k_cache, k_view_new], dim=2)
                        self.cached_v_per_view[cam_idx] = torch.cat([v_cache, v_view_new], dim=2)
                        
                        k_view_total = self.cached_k_per_view[cam_idx]
                        v_view_total = self.cached_v_per_view[cam_idx]

                    # if self.cached_k is None:
                    #     self.cached_k = k_idx
                    #     self.cached_v = v_idx
                    # else:
                    #     k_idx = self.cached_k = torch.cat((self.cached_k, k_idx), dim=2)
                    #     v_idx = self.cached_v = torch.cat((self.cached_v, v_idx), dim=2)
                else:
                    k_view_total = k_view_new
                    v_view_total = v_view_new


                view_len = num_patches_i
                view_start = cam_idx * view_len
                view_end = (cam_idx + 1) * view_len

                q_view = q_scale[:,:,view_start:view_end]

                # print(f"[SA] q_view.shape: {q_view.shape}")
                # print(f"[SA] q_view.shape: {q_view.shape}")
                # print(f"[SA] q_view.shape: {q_view.shape}")

                # q_idx = q[:, :, cam_idx * num_patches_i : (cam_idx + 1) * num_patches_i, :]
                out_view = F.scaled_dot_product_attention(
                    query=q_view.contiguous(),
                    key=k_view_total.contiguous(),
                    value=v_view_total.contiguous(),
                    # attn_mask=attn_bias_or_two_vector,
                )
                out_scale[:,:,view_start:view_end] = out_view
                # out[:, :, cam_idx * num_patches_i : (cam_idx + 1) * num_patches_i, :] = out_idx

            out_scale = apply_fn_o(out_scale)
            out_final.append(out_scale)
            tok_start = tok_end

        out = torch.cat(out_final, dim=2)




        # print(f"raw_d.shape: {raw_d.shape}")
        # print(f"self.p0_world.shape: {self.p0_world.shape}")
        # print(f"self.P.shape: {self.P.shape}")
        # print(f"out.shape: {out.shape}")


        out = out.transpose(1,2).reshape(B,L,C)

        return self.proj_drop(self.proj(out))
    

    def _precompute_and_cache_apply_fns(
        self, 
        w2cs: torch.Tensor, 
        Ks: Optional[torch.Tensor],
        num_patches: int,
        context_depths: Optional[torch.Tensor] = None,
    ):
        (batch, num_cameras, _, _) = w2cs.shape
        # assert w2cs.shape == (batch, num_cameras, 4, 4)
        # assert Ks is None or Ks.shape == (batch, num_cameras, 3, 3)
        
        self.batch = batch
        self.num_cameras = num_cameras
        self.context_depths = context_depths

        self.w2cs = w2cs  # (batch, cameras, 4, 4)
        self.c2ws = _invert_SE3(w2cs)  # (batch, cameras, 4, 4)

        Ks_norm = Ks.clone()
        Ks_norm[..., 0, 2] -= 0.5
        Ks_norm[..., 1, 2] -= 0.5
        
        # Compute the camera projection matrices we use in PRoPE.
        self.P = torch.einsum("...ij,...jk->...ik", _lift_K(Ks_norm), w2cs)
        self.P_T = self.P.transpose(-1, -2)
        self.P_inv = torch.einsum(
            "...ij,...jk->...ik",
            self.c2ws,
            _lift_K(_invert_K(Ks_norm)),
        )
        # assert self.P.shape == self.P_inv.shape == (batch, num_cameras, 4, 4)
        # assert self.head_dim % (2*self.rope_coord_dim) == 0, f"rope_dim={self.head_dim} must be multiple of 2*coord_dim={2*self.rope_coord_dim}"

        # get the ray segments in world coordinates
        self.p0_world = _get_cam_centers(self.c2ws, num_patches).unsqueeze(-2) # (batch, num_cameras, num_patches, 1, 4)
        # self.pinf_world = _get_point_coords(self.P_inv, self.patches_x, self.patches_y, self.offsets) # (batch, num_cameras, num_patches, num_rays_per_patch, 4)

        return
    

    # @torch.compile()
    def _prepare_apply_fns(
        self,
        num_cameras,
        patches_x,
        patches_y,
        predicted_d: Optional[torch.Tensor] = None,  # (batch, seqlen, 1 or 2)
        # debug: bool = False,
        # positions_collector: Optional[dict] = None,
    ) -> list[Callable[[torch.Tensor], torch.Tensor]]:
        """Prepare transforms for PRoPE-style positional encoding.
        
        [_transform_to_query_frame] points_world.shape: torch.Size([2, 3, 1, 1, 4])
        [_transform_to_query_frame] points_world.shape: torch.Size([2, 2, 3, 1, 3, 4])

        points_world.shape: torch.Size([1, 3, 256, 1, 4])
        points_world.shape: torch.Size([2, 1, 3, 256, 3, 4])

        
        """
        # (batch, num_cameras, _, _) = w2cs.shape
        batch = self.batch
        # num_cameras = self.num_cameras
        # patches_x = self.patches_x
        # patches_y = self.patches_y

        num_rays_per_patch = self.num_rays_per_patch
        num_patches = patches_x * patches_y

        # depths: [2,B,N_views,N_patches_per_view,num_rays_per_patch,1]
        # 2: predicted_d1, predicted_d2, i.e. d_min, d_max
        depths = _prepare_depths(predicted_d, self.context_depths, self.depth_type, 
                batch=batch, num_cameras=num_cameras, num_patches=num_patches, 
                patches_x=patches_x, patches_y=patches_y, 
                num_rays_per_patch=num_rays_per_patch, offsets=self.offsets)
    
        # print(f"depths.shape: {depths.shape}")
    
        # pd_world: [2, B, N_views, N_patches_per_view, num_rays_per_patch, 4]
        pd_world = _get_point_coords(self.P_inv, patches_x, patches_y, self.offsets, depths) # (2, batch, num_cameras, num_patches, num_rays_per_patch, 4)

        # print(f"pd_world.shape: {pd_world.shape}")

        positions_Q = defaultdict(list)
        # if positions_collector is not None:
        #     positions_collector["KV"] = []
        all_apply_fns_kv = []
        for cam_idx in range(num_cameras):
            positions_KV = {}

            P_q = self.P[:, cam_idx, :, :]  # (batch, 4, 4)
            w2c_q = self.w2cs[:, cam_idx, :, :]  # (batch, 4, 4)

            # self.p0_type: "3d"
            # self.denc_type: "d"
            # transform camera centers to view0 coord system
            p0_3d = _transform_to_query_frame(self.p0_world, P_q, w2c_q, self.p0_type, self.denc_type)
            p0_3d = p0_3d.flatten(-2, -1)
            positions_KV['p0'] = p0_3d
            positions_Q['p0'].append(p0_3d[:, cam_idx])


            pd_dir, pd_d = _transform_to_query_frame(pd_world, P_q, w2c_q, self.pd_type, self.denc_type)
            pd_dir = pd_dir.flatten(0, 1)[..., :2].flatten(start_dim=-2, end_dim=-1)
            pd_d = pd_d.flatten(0, 1).flatten(start_dim=-2, end_dim=-1)
            positions_KV['pd_dir'] = pd_dir
            positions_Q['pd_dir'].append(pd_dir[:, cam_idx])

            if self.denc_type == 'inv_d':
                positions_KV['pd_disparity'] = pd_d
                positions_Q['pd_disparity'].append(pd_d[:, cam_idx])
            elif self.denc_type == 'd':
                positions_KV['pd_depth'] = pd_d
                positions_Q['pd_depth'].append(pd_d[:, cam_idx])
            elif self.denc_type == 'asinh_d':
                positions_KV['pd_asinh_depth'] = pd_d
                positions_Q['pd_asinh_depth'].append(pd_d[:, cam_idx])

            
            cos_KV, sin_KV = _prepare_rope_coeff_uniformd(positions_KV, self.num_rope_freqs, self.freq_base, batch, num_cameras, num_patches) # (batch, num_cameras, num_patches, num_freqs, num_coord)

            # --- FIX START: PAD KV ---
            target_dim = self.head_dim // 2
            if cos_KV.shape[-1] < target_dim:
                pad_amt = target_dim - cos_KV.shape[-1]
                # Pad cosine with 1.0 (Identity)
                cos_KV = F.pad(cos_KV, (0, pad_amt), value=1.0)
                # Pad sine with 0.0 (No rotation)
                sin_KV = F.pad(sin_KV, (0, pad_amt), value=0.0)
            # --- FIX END ---

            # print(f"cos_KV.shape: {cos_KV.shape}")
            # print(f"sin_KV.shape: {sin_KV.shape}")

            apply_fn_kv = partial(_apply_rope_coeffs, cos=cos_KV, sin=sin_KV, inverse=True)
            
            all_apply_fns_kv.append(apply_fn_kv)


        for key, val in positions_Q.items():
            positions_Q[key] = torch.stack(val, dim=1)  # (batch, num_cameras, num_patches, ...)

        # if positions_collector is not None:
        #     positions_collector["Q"] = positions_Q

        # print(f"positions_Q.keys(): {positions_Q.keys()}")

        # print(f"positions_Q['p0'].shape: {positions_Q['p0'].shape}")
        # print(f"positions_Q['pd_dir'].shape: {positions_Q['pd_dir'].shape}")
        # print(f"positions_Q['pd_depth'].shape: {positions_Q['pd_depth'].shape}")

        cos_Q, sin_Q = _prepare_rope_coeff_uniformd(positions_Q, 
                                                    self.num_rope_freqs, 
                                                    self.freq_base, 
                                                    batch, 
                                                    num_cameras, 
                                                    num_patches)

        # --- FIX START: PAD Q ---
        if cos_Q.shape[-1] < target_dim:
            pad_amt = target_dim - cos_Q.shape[-1]
            cos_Q = F.pad(cos_Q, (0, pad_amt), value=1.0)
            cos_Q = cos_Q.contiguous() # Ensure contiguous after padding
            
            sin_Q = F.pad(sin_Q, (0, pad_amt), value=0.0)
            sin_Q = sin_Q.contiguous()
        # --- FIX END ---

        # print(f"cos_Q.shape: {cos_Q.shape}")
        # print(f"sin_Q.shape: {sin_Q.shape}")

        apply_fn_q = partial(_apply_rope_coeffs, cos=cos_Q, sin=sin_Q, inverse=True)
        apply_fn_o = partial(_apply_rope_coeffs, cos=cos_Q, sin=sin_Q, inverse=False)

        # if (not torch.isfinite(cos_Q).all()) or (not torch.isfinite(sin_Q).all()):
        #     raise ValueError("NaN/inf values found in rope_matrices_Q.")

        return apply_fn_q, all_apply_fns_kv, apply_fn_o

    def extra_repr(self) -> str:
        tail = ''
        return f''


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from functools import partial
# from typing import Optional, Union, Tuple

# Import helper functions from your rayrope.py
# from rayrope import (
#     _prepare_depths,
#     _get_cam_centers,
#     _get_point_coords,
#     _transform_to_query_frame,
#     _prepare_rope_coeff_uniformd,
#     _apply_rope_coeffs,
#     _invert_SE3,
#     normalize_K,
#     _lift_K,
#     _invert_K,
# )

class SelfAttentionRayRope2(nn.Module):
    def __init__(
        self, 
        embed_dim=768, 
        num_heads=12,
        proj_drop=0., 
        tau=1, 
        cos_attn=False, 
        # RayRoPE specific args
        num_rays_per_patch=3,
        depth_type='predict_dsig', 
        freq_base=3.0,
        image_width=256,
        image_height=256,
        init_depth=0.0,
        init_sigma=3.0,
        **kwargs,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.tau = tau
        self.cos_attn = cos_attn

        # --- RayRoPE Configuration (Hardcoded for d_pj+0_3d and denc_type='d') ---
        self.depth_type = depth_type
        self.denc_type = 'd'  # Hardcoded
        self.num_rays_per_patch = num_rays_per_patch
        self.freq_base = freq_base
        self.image_width = image_width
        self.image_height = image_height
        
        # Hardcoded: pos_enc_type = 'd_pj+0_3d'
        # 1. p0 (3d): 1 point * 3 coords = 3
        # 2. pd (pj): num_rays * 3 coords (2 dir + 1 depth) = num_rays * 3
        self.rope_coord_dim = 3 + (self.num_rays_per_patch * 3)
        self.rope_mat_dim = 2 * self.rope_coord_dim
        
        # assert self.head_dim % (self.rope_coord_dim * 2) == 0, \
        #     f"head_dim={self.head_dim} must be multiple of rope_coord_dim * 2 ({self.rope_coord_dim * 2})"
        self.num_rope_freqs = self.head_dim // self.rope_mat_dim

        # Ray offsets setup
        if self.num_rays_per_patch == 3:
            self.offsets = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        elif self.num_rays_per_patch == 2:
            self.offsets = [[0.0, 0.0], [1.0, 1.0]]
        else:
            self.offsets = [[0.5, 0.5]]

        # --- Layers ---
        if self.cos_attn:
            self.scale = 1
            # Standard shape for non-flash (1, H, 1, 1)
            size = (1, self.num_heads, 1, 1)
            self.scale_mul_1H11 = nn.Parameter(torch.full(size=size, fill_value=4.0).log(), requires_grad=True)
            self.max_scale_mul = torch.log(torch.tensor(100)).item()
        else:
            self.scale = 1 / (self.head_dim ** 0.5) / self.tau

        self.mat_qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(embed_dim))
        self.v_bias = nn.Parameter(torch.zeros(embed_dim))
        self.register_buffer('zero_k_bias', torch.zeros(embed_dim))

        # Depth Prediction Head
        if 'predict_dsig' in depth_type:
            self.d_proj_weight = nn.Parameter(torch.zeros((2, embed_dim)))
            self.d_proj_bias = nn.Parameter(torch.tensor([init_depth, init_sigma]))
        elif 'predict_d' in depth_type:
            self.d_proj_weight = nn.Parameter(torch.zeros((1, embed_dim)))
            self.d_proj_bias = nn.Parameter(torch.tensor([init_depth]))
        else:
            self.d_proj_weight = None

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.caching = False
        self.cached_k = None # Stores UNROTATED keys
        self.cached_v = None # Stores UNROTATED values
        self.cached_p0_world = None # Stores world coords
        self.cached_pd_world = None

        self.using_flash = False

    def kv_caching(self, enable: bool):
        self.caching = enable
        self.cached_k = None
        self.cached_v = None
        self.cached_p0_world = None
        self.cached_pd_world = None

    def _compute_world_coords(self, batch, num_cameras, patches_x, patches_y, c2ws, P_inv, predicted_d_chunk):
        """
        Compute world coordinates for the CURRENT chunk of tokens.
        """
        num_patches = patches_x * patches_y
        
        # 1. Prepare Depths
        depths = _prepare_depths(
            predicted_d_chunk, 
            context_depths=None,
            depth_type=self.depth_type,
            batch=batch, num_cameras=num_cameras, num_patches=num_patches,
            patches_x=patches_x, patches_y=patches_y,
            num_rays_per_patch=self.num_rays_per_patch, offsets=self.offsets
        )

        # 2. Get World Coordinates
        # p0_world: (Batch, num_cameras, num_patches, 1, 4)
        p0_world = _get_cam_centers(c2ws, num_patches).unsqueeze(-2) 
        
        # pd_world: (2, Batch, num_cameras, num_patches, num_rays, 4)
        # Note the leading '2' for depth uncertainty intervals
        pd_world = _get_point_coords(P_inv, patches_x, patches_y, self.offsets, depths)
        
        # Flatten num_cameras and num_patches into a single sequence dimension
        
        # p0: (B, C, P, 1, 4) -> (B, Seq, 1, 4)
        p0_world = p0_world.flatten(1, 2)
        
        # pd: (2, B, C, P, R, 4) -> (2, B, Seq, R, 4)
        # We flatten dims 2 and 3 (Cam and Patch)
        pd_world = pd_world.flatten(2, 3)
        
        return p0_world, pd_world

    def _get_rope_fns_for_query(self,
                                query_cam_idx, 
                                P, 
                                w2cs, 
                                p0_world_history, 
                                pd_world_history):
        """
        Generate RoPE functions for a SPECIFIC query camera, applying to the ENTIRE history.
        """
        P_q = P[:, query_cam_idx]       # (Batch, 4, 4)
        w2c_q = w2cs[:, query_cam_idx]  # (Batch, 4, 4)
        
        batch_size = P_q.shape[0]
        
        positions_KV = {}

        # --- Transform p0 (Camera Centers) ---
        # History: (Batch, Seq, 1, 4)
        # Input to transform: (Batch, 1, Seq, 1, 4) -> Fake '1' Camera dim
        p0_input = p0_world_history.unsqueeze(1)
        
        # Output from transform: (Batch, 1, Seq, 1, 3)
        p0_3d = _transform_to_query_frame(p0_input, P_q, w2c_q, transform_type='3d', denc_type='d')
        
        # CORRECT FIX: Squeeze the RAY dimension (-2), but KEEP the CAMERA dimension (1).
        # Result: (Batch, 1, Seq, 3) -> Matches 'bcpd' for einsum
        p0_3d = p0_3d.squeeze(-2) 
        positions_KV['p0'] = p0_3d

        # --- Transform pd (Points at Depth) ---
        # History: (2, Batch, Seq, Rays, 4)
        # Input to transform: (2, Batch, 1, Seq, Rays, 4) -> Fake '1' Camera dim at dim 2
        pd_input = pd_world_history.unsqueeze(2)
        
        # Output: (2, Batch, 1, Seq, Rays, 3) and (2, Batch, 1, Seq, Rays, 1)
        pd_dir, pd_d = _transform_to_query_frame(pd_input, P_q, w2c_q, transform_type='pj', denc_type='d')
        
        # Flatten the first two dims (2, Batch) -> (2*Batch, ...)
        # Keep dim 1 (Camera=1) intact.
        
        # pd_dir: (2*B, 1, S, R, 3)
        pd_dir = pd_dir.flatten(0, 1)
        # Flatten Rays*Coords: (2*B, 1, S, R*2)
        pd_dir = pd_dir[..., :2].flatten(start_dim=-2, end_dim=-1)
        
        # pd_d: (2*B, 1, S, R, 1) -> (2*B, 1, S, R*1)
        pd_d = pd_d.flatten(0, 1).flatten(start_dim=-2, end_dim=-1)
        
        positions_KV['pd_dir'] = pd_dir
        positions_KV['pd_depth'] = pd_d
        
        # Compute Cos/Sin
        total_seq_len = p0_world_history.shape[1]
        
        # num_cameras=1 matches the singleton dimension we preserved
        cos, sin = _prepare_rope_coeff_uniformd(
            positions_KV, self.num_rope_freqs, self.freq_base, 
            batch=batch_size, num_cameras=1, num_patches=total_seq_len
        )
        
        # Pad to head_dim // 2
        target_dim = self.head_dim // 2
        if cos.shape[-1] < target_dim:
            pad_amt = target_dim - cos.shape[-1]
            cos = F.pad(cos, (0, pad_amt), value=1.0)
            sin = F.pad(sin, (0, pad_amt), value=0.0)
            
        return partial(_apply_rope_coeffs, cos=cos, sin=sin)

    def forward(self, 
                x, 
                attn_bias_or_two_vector: Union[torch.Tensor, Tuple[torch.IntTensor, torch.IntTensor]], 
                attn_fn=None, 
                scale_schedule=None, 
                rope2d_freqs_grid=None, 
                scale_ind=None,
                poses=None,
                intrs=None,
                input_size=None):
        B, L, C = x.shape
        
        # 1. QKV Proj
        qkv = F.linear(input=x, weight=self.mat_qkv.weight, 
                      bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias)))
        qkv = qkv.view(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0) # (B, H, L, D)

        # Depth Prediction
        predicted_d = None
        if self.d_proj_weight is not None:
            predicted_d = F.linear(x, self.d_proj_weight, self.d_proj_bias)

        # 2. Camera Matrices
        num_cameras = poses.shape[1]
        c2ws = poses
        w2cs = _invert_SE3(c2ws)
        Ks_norm = intrs.clone()
        Ks_norm[..., 0, 2] -= 0.5
        Ks_norm[..., 1, 2] -= 0.5
        P = torch.einsum("...ij,...jk->...ik", _lift_K(Ks_norm), w2cs)
        P_inv = torch.einsum("...ij,...jk->...ik", c2ws, _lift_K(_invert_K(Ks_norm)))

        # Handle Scale Schedule
        if scale_ind is not None:
            scale_schedule = [scale_schedule[scale_ind]]

        tok_start = 0
        outs = []

        print(f"scale_schedule: {scale_schedule}")
        print(f"q.shape: {q.shape}")
        print(f"k.shape: {k.shape}")
        print(f"v.shape: {v.shape}")
        print(f"predicted_d.shape: {predicted_d.shape}")
        
        
        # 3. Multiscale Loop
        for scale_idx, item in enumerate(scale_schedule):
            patches_x, patches_y = item[-1], item[-2]
            chunk_len = num_cameras * patches_x * patches_y
            tok_end = tok_start + chunk_len

            # A. Extract Chunk from INPUT
            q_chunk = q[:, :, tok_start:tok_end]
            k_chunk = k[:, :, tok_start:tok_end]
            v_chunk = v[:, :, tok_start:tok_end]
            d_chunk = predicted_d[:, tok_start:tok_end] if predicted_d is not None else None

            # B. Compute World Coords for this chunk
            p0_chunk, pd_chunk = self._compute_world_coords(
                B, num_cameras, patches_x, patches_y, 
                c2ws, P_inv, d_chunk
            )

            print(f"p0_chunk.shape: {p0_chunk.shape}")
            print(f"pd_chunk.shape: {pd_chunk.shape}")

            # --- FIX 2 & 3: Robust Caching & Indexing ---
            
            # Calculate where the NEW chunk starts within the TOTAL history.
            # If cached_k is None, offset is 0. 
            # If cached_k exists, offset is its current length.
            # C. Update Caches (UNROTATED K/V and World Coords)
            if self.cached_k is None:
                # Initialize history
                history_offset = 0
                self.cached_k = k_chunk
                self.cached_v = v_chunk
                self.cached_p0_world = p0_chunk
                self.cached_pd_world = pd_chunk
            else:
                # Update history
                history_offset = self.cached_k.shape[2] 
                
                # K/V are (Batch, Head, Seq, Dim) -> Concat on dim 2
                self.cached_k = torch.cat((self.cached_k, k_chunk), dim=2)
                self.cached_v = torch.cat((self.cached_v, v_chunk), dim=2)
                
                # p0 is (Batch, Seq, 1, 4) -> Concat on dim 1
                self.cached_p0_world = torch.cat((self.cached_p0_world, p0_chunk), dim=1)
                
                # FIX: pd is (2, Batch, Seq, Ray, 4) -> Concat on dim 2
                self.cached_pd_world = torch.cat((self.cached_pd_world, pd_chunk), dim=2)

            # D. Per-Camera Attention
            chunk_outs = []
            num_patches = patches_x * patches_y
            
            for cam_idx in range(num_cameras):
                # 1. Prepare Q for this camera (Current Chunk Only)
                q_cam_start_local = cam_idx * num_patches
                q_cam_end_local = (cam_idx + 1) * num_patches
                q_cam = q_chunk[:, :, q_cam_start_local:q_cam_end_local]

                # 2. Get RoPE coeffs for FULL HISTORY
                rope_fn = self._get_rope_fns_for_query(
                    cam_idx, P, w2cs, 
                    self.cached_p0_world, self.cached_pd_world
                )
                
                full_cos = rope_fn.keywords['cos']
                full_sin = rope_fn.keywords['sin']
                
                # 3. Slice angles for Q
                # Q is located at [history_offset + local_start : history_offset + local_end]
                # This ensures Q gets the angles corresponding to its ACTUAL position in history/time.
                abs_q_start = history_offset + q_cam_start_local
                abs_q_end = history_offset + q_cam_end_local
                
                cos_q = full_cos[:, abs_q_start:abs_q_end]
                sin_q = full_sin[:, abs_q_start:abs_q_end]
                
                q_cam_rot = _apply_rope_coeffs(q_cam, cos_q, sin_q, inverse=True)
                
                # 4. Rotate K/V (Full History)
                k_hist_rot = _apply_rope_coeffs(self.cached_k, full_cos, full_sin, inverse=True)
                v_hist_rot = _apply_rope_coeffs(self.cached_v, full_cos, full_sin, inverse=True)
                
                # 5. Attention
                out_cam = F.scaled_dot_product_attention(
                    q_cam_rot, k_hist_rot, v_hist_rot, 
                    dropout_p=self.proj_drop.p if self.training else 0.0
                )
                
                # 6. Output Transform
                out_cam = _apply_rope_coeffs(out_cam, cos_q, sin_q, inverse=False)
                
                chunk_outs.append(out_cam)
            
            chunk_out = torch.cat(chunk_outs, dim=2)
            outs.append(chunk_out)
            
            tok_start = tok_end

        # Cleanup if caching is disabled (Training mode)
        if not self.caching:
            self.cached_k = None
            self.cached_v = None
            self.cached_p0_world = None
            self.cached_pd_world = None

        outs = torch.cat(outs, dim=2)
        outs = outs.transpose(1, 2).reshape(B, L, C)
        return self.proj_drop(self.proj(outs))

    def extra_repr(self) -> str:
        tail = ''
        return f''


class SelfAttentionRayRopeUnified(nn.Module):
    def __init__(
        self, 
        embed_dim=768, 
        num_heads=12,
        proj_drop=0., 
        tau=1, 
        cos_attn=False, 
        num_rays_per_patch=3,
        depth_type='predict_dsig', 
        freq_base=3.0,
        apply_vo=True,
        **kwargs,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.tau = tau
        self.cos_attn = cos_attn

        # --- Geometric Config ---
        self.depth_type = depth_type
        self.num_rays_per_patch = num_rays_per_patch
        self.freq_base = freq_base
        self.apply_vo = apply_vo
        
        # Dimensions for RoPE
        self.rope_coord_dim = 3 + (self.num_rays_per_patch * 3)
        self.rope_mat_dim = 2 * self.rope_coord_dim
        self.num_rope_freqs = self.head_dim // self.rope_mat_dim

        # Ray Offsets
        if self.num_rays_per_patch == 3:
            self.offsets = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        elif self.num_rays_per_patch == 2:
            self.offsets = [[0.0, 0.0], [1.0, 1.0]]
        else:
            self.offsets = [[0.5, 0.5]]

        # --- Layers ---
        if self.cos_attn:
            self.scale_mul = nn.Parameter(torch.tensor(0.0), requires_grad=True)
            self.max_scale_mul = math.log(100.0)
        else:
            self.scale = 1 / (self.head_dim ** 0.5)

        self.mat_qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(embed_dim))
        self.v_bias = nn.Parameter(torch.zeros(embed_dim))
        self.register_buffer('zero_k_bias', torch.zeros(embed_dim))

        # Depth Head
        if 'predict_dsig' in depth_type:
            self.d_proj_weight = nn.Parameter(torch.zeros((2, embed_dim)))
            self.d_proj_bias = nn.Parameter(torch.tensor([0.0, 3.0]))
        else:
            self.d_proj_weight = None

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # --- Caching State ---
        self.caching = False
        self.per_camera_caches = None 

        self.using_flash = False

    def kv_caching(self, enable: bool):
        """Toggle inference caching mode."""
        self.caching = enable
        self.per_camera_caches = None # Reset on toggle

    def reset_cache(self, num_cameras):
        """Initialize empty caches for N cameras."""
        self.per_camera_caches = [{'k': None, 'v': None} for _ in range(num_cameras)]

    # -----------------------------------------------------------------------
    # PATH A: TRAINING HELPERS (Vectorized Grid)
    # -----------------------------------------------------------------------
    def _precompute_full_grid(self, scale_schedule, num_cameras, device):
        all_coords_list = []
        cam_id_list = []
        
        for item in scale_schedule:
            px, py = item[-1], item[-2]
            num_patches = px * py
            
            y_base = torch.arange(py, device=device, dtype=torch.float32)
            x_base = torch.arange(px, device=device, dtype=torch.float32)
            grid_y, grid_x = torch.meshgrid(y_base, x_base, indexing='ij')
            grid_x = grid_x.flatten()
            grid_y = grid_y.flatten()

            scale_coords = []
            for (ox, oy) in self.offsets:
                u = ((grid_x.unsqueeze(-1) + ox) / px) - 0.5
                v = ((grid_y.unsqueeze(-1) + oy) / py) - 0.5
                scale_coords.append(torch.stack([u, v], dim=-1))
            
            scale_coords = torch.cat(scale_coords, dim=1) 
            scale_coords_expanded = scale_coords.repeat(num_cameras, 1, 1)
            all_coords_list.append(scale_coords_expanded)
            
            c_ids = torch.arange(num_cameras, device=device).repeat_interleave(num_patches)
            cam_id_list.append(c_ids)

        all_coords_norm = torch.cat(all_coords_list, dim=0)
        cam_ids = torch.cat(cam_id_list, dim=0)
        return all_coords_norm, cam_ids

    # -----------------------------------------------------------------------
    # PATH B: INFERENCE HELPERS (Single Chunk Grid)
    # -----------------------------------------------------------------------
    def _compute_chunk_coords(self, patches_x, patches_y, device):
        """Compute grid just for the current inference chunk."""
        y_base = torch.arange(patches_y, device=device, dtype=torch.float32)
        x_base = torch.arange(patches_x, device=device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(y_base, x_base, indexing='ij')
        grid_x = grid_x.flatten()
        grid_y = grid_y.flatten()

        scale_coords = []
        for (ox, oy) in self.offsets:
            u = ((grid_x.unsqueeze(-1) + ox) / patches_x) - 0.5
            v = ((grid_y.unsqueeze(-1) + oy) / patches_y) - 0.5
            scale_coords.append(torch.stack([u, v], dim=-1))
        
        return torch.cat(scale_coords, dim=1) # (P, R, 2)

    # -----------------------------------------------------------------------
    # SHARED HELPERS (World Coords & RoPE)
    # -----------------------------------------------------------------------
    def _compute_world_coords_vectorized(self, P_inv_seq, all_coords_norm, predicted_d):
        
        # predicted_d: [B, L, 1, 2(D,Sigma)]
        if predicted_d.ndim == 4:
             logd = predicted_d[..., 0:1]
             sigma = predicted_d[..., 1:2]
        else:
             logd = predicted_d[..., 0:1].unsqueeze(-2)
             sigma = predicted_d[..., 1:2].unsqueeze(-2)

        B, L = logd.shape[0], logd.shape[1]
        R = all_coords_norm.shape[1]
        
        d1 = torch.exp(torch.clamp(logd - sigma, max=3.0))
        d2 = torch.exp(torch.clamp(logd + sigma, max=3.0))
        # depths: [2,B,L,R,1]
        depths = torch.stack([d1, d2], dim=0).expand(-1, -1, -1, R, -1)
        # disparity: [2,B,L,R,1]
        disparity = 1.0 / torch.clamp(depths, min=1e-4, max=100.0)

        # coords_input: [2,B,N_patches,N_rays, 2(u,v)]
        coords_input = all_coords_norm.unsqueeze(0).unsqueeze(0).expand(2, B, -1, -1, -1)
        ones = torch.ones_like(disparity)
        
        # [uv,1,1/d] in global coords
        coords_4d = torch.cat([coords_input, ones, disparity], dim=-1)
        print(f"coords_4d.shape: {coords_4d.shape}")
        P_inv_broad = P_inv_seq.unsqueeze(0).unsqueeze(3)
        pd_world = torch.einsum("...ij,...j->...i", P_inv_broad, coords_4d)
        
        return pd_world

    def _get_rope_fns_batched(self, P, w2c, p0, pd, seq_len):
        # p0 in cam coords: [B*N_views, 1, L, 3(xyz)] D=3
        p0_3d = _transform_to_query_frame(p0, P, w2c, transform_type='3d', denc_type='d').squeeze(-2)
       
        # pd projected: 
        # pd_dir: [2,B*N_views,1,L,R,3(xyz)] D=3
        # pd_d: [2,B*N_views,1,L,R,1(depth)] D=1
        pd_dir, pd_d = _transform_to_query_frame(pd, P, w2c, transform_type='pj', denc_type='d')
        
        print(f"p0_3d.shape: {p0_3d.shape}")
        print(f"pd_dir.shape: {pd_dir.shape}")
        print(f"pd_d.shape: {pd_d.shape}")

        # pd_dir [2*B*N_views, 1, L, 6(R*xy)] z removed 
        pd_dir = pd_dir.flatten(0, 1)[..., :2].flatten(start_dim=-2, end_dim=-1)
        # pd_d: [2*B*N_Views, 1, L, R]
        pd_d = pd_d.flatten(0, 1).flatten(start_dim=-2, end_dim=-1)
        
        print(f"[flatten] pd_dir.shape: {pd_dir.shape}")
        print(f"[flatten] pd_d.shape: {pd_d.shape}")
        
        positions = {'p0': p0_3d, 'pd_dir': pd_dir, 'pd_depth': pd_d}
        cos, sin = _prepare_rope_coeff_uniformd(
            positions, self.num_rope_freqs, self.freq_base, 
            batch=P.shape[0], num_cameras=1, num_patches=seq_len
        )
        print(f"cos.shape: {cos.shape}")
        print(f"sin.shape: {sin.shape}")
        
        target = self.head_dim // 2
        if cos.shape[-1] < target:
            pad = target - cos.shape[-1]
            cos = F.pad(cos, (0, pad), value=1.0)
            sin = F.pad(sin, (0, pad), value=0.0)
            
        return cos, sin

    # -----------------------------------------------------------------------
    # MAIN FORWARD
    # -----------------------------------------------------------------------
    def forward(self, 
                x, 
                attn_bias_or_two_vector: Union[torch.Tensor, Tuple[torch.IntTensor, torch.IntTensor]], 
                attn_fn=None, 
                scale_schedule=None, 
                rope2d_freqs_grid=None, 
                scale_ind=None,
                poses=None,
                intrs=None,
                input_size=None):
        
        B, L, C = x.shape
        device = x.device
        num_cameras = poses.shape[1]
        
        # QKV Proj
        qkv = F.linear(x, self.mat_qkv.weight, torch.cat((self.q_bias, self.zero_k_bias, self.v_bias)))
        qkv = qkv.view(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0) 

        if self.cos_attn:
            scale_mul = self.scale_mul.clamp_max(self.max_scale_mul).exp()
            q = F.normalize(q, dim=-1, eps=1e-12).mul(scale_mul)
            k = F.normalize(k, dim=-1, eps=1e-12)

        # Depth Pred
        predicted_d = None
        if self.d_proj_weight is not None:
            predicted_d = F.linear(x, self.d_proj_weight, self.d_proj_bias) # (B, L, 2)
            predicted_d = predicted_d.view(B, L, 1, 2)

        # Common Matrices
        c2ws = poses 
        w2cs = _invert_SE3(c2ws)
        Ks_norm = intrs.clone()
        Ks_norm[..., 0, 2] -= 0.5
        Ks_norm[..., 1, 2] -= 0.5
        P_inv_all = torch.einsum("...ij,...jk->...ik", c2ws, _lift_K(_invert_K(Ks_norm)))


        # =========================================================================
        # BRANCH 1: TRAINING (Vectorized, No Cache)
        # =========================================================================
        if not self.caching:
            
            # A. Precompute Full Grid
            # all_coords_norm: [N_Views*N_tokens_total, N_views, 2(uv)]
            all_coords_norm, cam_ids = self._precompute_full_grid(scale_schedule, num_cameras, device)
            print(f"all_coords_norm.shape: {all_coords_norm.shape}")
            print(f"cam_ids.shape: {cam_ids.shape}")
            print(f"cam_ids[:20]: {cam_ids[:20]}")
            # B. Gather P_inv & Compute World Coords
            P_inv_seq = P_inv_all[:, cam_ids]
            
            # [2(d1,d2),B,N_views*N_tokens_total,R,4]
            pd_world = self._compute_world_coords_vectorized(P_inv_seq, all_coords_norm, predicted_d)
            # [B,N_views*N_tokens_total,4]
            p0_world = c2ws[:, cam_ids][..., :, 3] 

            print(f"pd_world.shape: {pd_world.shape}")
            print(f"p0_world.shape: {p0_world.shape}")
            # C. View Expansion (Parallelize N views)
            p0_world_exp = p0_world.unsqueeze(1).repeat(1, num_cameras, 1, 1).unsqueeze(-2) 
            pd_world_exp = pd_world.unsqueeze(2).repeat(1, 1, num_cameras, 1, 1, 1)

            BN = B * num_cameras
            w2cs_flat = w2cs.view(BN, 4, 4)
            Ps_flat = torch.einsum("...ij,...jk->...ik", _lift_K(Ks_norm), w2cs).view(BN, 4, 4)
            
            p0_in = p0_world_exp.view(BN, L, 1, 4).unsqueeze(1) 
            pd_in = pd_world_exp.view(2, BN, L, self.num_rays_per_patch, 4).unsqueeze(2)

            print(f"p0_in.shape: {p0_in.shape}")
            print(f"pd_in.shape: {pd_in.shape}")

            # D. RoPE & Attention
            cos_full, sin_full = self._get_rope_fns_batched(Ps_flat, w2cs_flat, p0_in, pd_in, L)

            q_exp = q.repeat_interleave(num_cameras, dim=0)
            k_exp = k.repeat_interleave(num_cameras, dim=0)
            v_exp = v.repeat_interleave(num_cameras, dim=0)

            q_rot = self._apply_rope_coeffs(q_exp, cos_full, sin_full, inverse=True)
            k_rot = self._apply_rope_coeffs(k_exp, cos_full, sin_full, inverse=True)
            v_rot = self._apply_rope_coeffs(v_exp, cos_full, sin_full, inverse=True) if self.apply_vo else v_exp
            
            # print(f"q_rot.shape: {q_rot.shape}")
            # print(f"k_rot.shape: {k_rot.shape}")
            # print(f"v_rot.shape: {v_rot.shape}")

            # Single Causal Attention
            out = F.scaled_dot_product_attention(
                q_rot, k_rot, v_rot, 
                attn_mask=attn_bias_or_two_vector,
                dropout_p=self.proj_drop.p if self.training else 0.0,
                is_causal=False 
            )

            exit()

            if self.apply_vo:
                out = self._apply_rope_coeffs(out, cos_full, sin_full, inverse=False)

            # E. Gather Valid Outputs
            out = out.view(B, num_cameras, self.num_heads, L, self.head_dim)
            gather_ids = cam_ids.view(1, 1, 1, L, 1).expand(B, 1, self.num_heads, L, self.head_dim)
            final_out = torch.gather(out, 1, gather_ids).squeeze(1) 
            final_out = final_out.transpose(1, 2).reshape(B, L, C)
            
            return self.proj_drop(self.proj(final_out))

        # =========================================================================
        # BRANCH 2: INFERENCE (Cached, Chunk-by-Chunk)
        # =========================================================================
        else:
            if self.per_camera_caches is None:
                self.reset_cache(num_cameras)
                
            # Current Chunk Info
            # In inference, x is usually just the new tokens (the current scale)
            # scale_ind tells us which scale we are on.
            if scale_ind is None:
                # Fallback if user forgot scale_ind during inference loop
                scale_ind = 0 
                
            px, py = scale_schedule[scale_ind][-1], scale_schedule[scale_ind][-2]
            # chunk_L = num_cameras * px * py
            
            # A. Chunk Grid & World Coords
            # chunk_coords: [N_patches, N_rays, 2(u,v)]
            chunk_coords = self._compute_chunk_coords(px, py, device) # (P, R, 2)
            
            print(f"scale_ind: {scale_ind} chunk_coords.shape: {chunk_coords.shape}")
            print(f'predicted_d.shape: {predicted_d.shape}')

            # Expand chunk coords for all cameras: (C*P, R, 2)
            chunk_coords_all = chunk_coords.repeat(num_cameras, 1, 1)
            
            
            # We need P_inv for the CURRENT chunk of tokens. 
            # The chunk contains [Cam0_Tokens, Cam1_Tokens...]
            # We construct P_inv_seq for this chunk
            P_inv_seq = P_inv_all.repeat_interleave(px*py, dim=1) # (B, C*P, 4, 4)
            
            # Compute World Coords for Chunk 
            # pd_chunk: [2(d1,d2),B,L,R,4(xyzw)]
            pd_chunk = self._compute_world_coords_vectorized(P_inv_seq, chunk_coords_all, predicted_d)
            
            # p0 for chunk (Expand c2ws: B, C, 4, 4 -> B, C*P, 4, 4)
            # p0_chunk: [B,L,4(xyzw)]
            p0_chunk = c2ws.repeat_interleave(px*py, dim=1)[..., :, 3]

            print(f"pd_chunk.shape: {pd_chunk.shape}")
            print(f"p0_chunk.shape: {p0_chunk.shape}")

            

            # B. Update Caches (Per-Camera Rotation)
            # We loop over cameras to create N view-specific caches
            
            # Pre-calculate chunk's RoPE for all views to save time
            # We can vectorize this "Update" step too!
            # Chunk is size L_new. We want to rotate it N times.
            # Use same logic as training but just for L_new.
            
            chunk_len = x.shape[1]
            BN = B * num_cameras
            
            # p0_chunk: [B,L,4]
            # p0_chunk_exp: [B,N_views,L,4]
            p0_chunk_exp = p0_chunk.unsqueeze(1).repeat(1, num_cameras, 1, 1).unsqueeze(-2)
            
            # pd_chunk: [B,2,L,R,4]
            # pd_chunk_exp: [2,B,N_views,L,R,4]
            pd_chunk_exp = pd_chunk.unsqueeze(2).repeat(1, 1, num_cameras, 1, 1, 1)
            
            w2cs_flat = w2cs.view(BN, 4, 4)
            Ps_flat = torch.einsum("...ij,...jk->...ik", _lift_K(Ks_norm), w2cs).view(BN, 4, 4)

            # p0_in: [B*N_views,1,L,1,4]
            p0_in = p0_chunk_exp.view(BN, chunk_len, 1, 4).unsqueeze(1)
            # [2,B*N_views,1,L,R,4]
            pd_in = pd_chunk_exp.view(2, BN, chunk_len, self.num_rays_per_patch, 4).unsqueeze(2)

            print(f"p0_in.shape: {p0_in.shape}")
            print(f"pd_in.shape: {pd_in.shape}")

            # P: K*w2cs
            # cos/sin_chunk: [B*N_views, R, head_dim]
            cos_chunk, sin_chunk = self._get_rope_fns_batched(Ps_flat, 
                                                              w2cs_flat, 
                                                              p0_in, 
                                                              pd_in, 
                                                              chunk_len)
            
            print(f"cos_chunk.shape: {cos_chunk.shape}")
            print(f"sin_chunk.shape: {sin_chunk.shape}")

            # Rotate Chunk for all views
            q_exp = q.repeat_interleave(num_cameras, dim=0)
            k_exp = k.repeat_interleave(num_cameras, dim=0)
            v_exp = v.repeat_interleave(num_cameras, dim=0)

            print(f"q.shape: {q.shape}")
            print(f"q_exp.shape: {q_exp.shape}")
            

            # Store cos/sin for Q later (we need them for attention step)
            # But wait, Q is specific to a camera. 
            # In inference, we have Qs for Cam0, Qs for Cam1... 
            # q_exp contains [Q_all_view0, Q_all_view1...]
            # We only need Q_cam0_view0, Q_cam1_view1.
            
            # Let's rotate K/V first (full N views)
            k_rot_chunk = self._apply_rope_coeffs(k_exp, cos_chunk, sin_chunk, inverse=True)
            v_rot_chunk = self._apply_rope_coeffs(v_exp, cos_chunk, sin_chunk, inverse=True) if self.apply_vo else v_exp

            # Update Caches
            # k_rot_chunk is (B*N, H, L_new, D)
            # Split back to cameras
            k_rot_per_cam = k_rot_chunk.view(B, num_cameras, self.num_heads, chunk_len, self.head_dim)
            v_rot_per_cam = v_rot_chunk.view(B, num_cameras, self.num_heads, chunk_len, self.head_dim)
            
            for c in range(num_cameras):
                k_new = k_rot_per_cam[:, c]
                v_new = v_rot_per_cam[:, c]
                
                if self.per_camera_caches[c]['k'] is None:
                    self.per_camera_caches[c]['k'] = k_new
                    self.per_camera_caches[c]['v'] = v_new
                else:
                    self.per_camera_caches[c]['k'] = torch.cat([self.per_camera_caches[c]['k'], k_new], dim=2)
                    self.per_camera_caches[c]['v'] = torch.cat([self.per_camera_caches[c]['v'], v_new], dim=2)

            # C. Attention (Per-Camera)
            # We process this chunk-by-chunk logic for output
            # q_exp is (B*N, ...). We need to select valid Qs.
            # Q for Cam0 is in the first (px*py) slots of x.
            # But q_exp exploded x N times.
            # Q_Cam0 needs RoPE from View0. -> index 0 in BN
            # Q_Cam1 needs RoPE from View1. -> index N+1 in BN? No.
            
            # Let's simplify: Loop over cameras for the Attention Step (since it's fast anyway with cache)
            chunk_outs = []
            num_patches = px * py
            
            # reshape Q back to (B, L, ...) to slice
            # Actually use the original 'q' (B, H, L, D) and slice it
            # But we need the Rotation Coeffs for Q.
            # cos_chunk is (B*N, L, D/2). 
            # View0 coeffs are at batch index 0. View1 at index 1 (assuming B=1).
            
            cos_view = cos_chunk.view(B, num_cameras, chunk_len, -1)
            sin_view = sin_chunk.view(B, num_cameras, chunk_len, -1)
            
            for c in range(num_cameras):
                # 1. Slice Q for Camera C
                start = c * num_patches
                end = (c+1) * num_patches
                q_cam = q[:, :, start:end] 
                
                # 2. Get RoPE for View C (slice from precomputed batch)
                # We need coeffs corresponding to the tokens Q_cam.
                # cos_view[b, c] gives coeffs for ALL tokens in chunk as seen by C.
                # We only want the coeffs for the specific tokens Q_cam.
                cos_q = cos_view[:, c, start:end]
                sin_q = sin_view[:, c, start:end]
                
                # 3. Rotate Q
                q_rot = self._apply_rope_coeffs(q_cam, cos_q, sin_q, inverse=True)
                
                # 4. Fetch Cache
                k_cache = self.per_camera_caches[c]['k']
                v_cache = self.per_camera_caches[c]['v']
                
                # 5. Attention
                out_cam = F.scaled_dot_product_attention(
                    q_rot, k_cache, v_cache, dropout_p=0.0
                )
                
                if self.apply_vo:
                    out_cam = self._apply_rope_coeffs(out_cam, cos_q, sin_q, inverse=False)
                    
                chunk_outs.append(out_cam)
                
            final_out = torch.cat(chunk_outs, dim=2)
            final_out = final_out.transpose(1, 2).reshape(B, chunk_len, C)
            
            return self.proj_drop(self.proj(final_out))

    # --- Helpers ---
    @staticmethod
    def _apply_rope_coeffs(feats, cos, sin, inverse=False):
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        x1, x2 = torch.chunk(feats, 2, dim=-1)
        if inverse:
            return torch.cat([x1 * cos + x2 * sin, -x1 * sin + x2 * cos], dim=-1)
        else:
            return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    def extra_repr(self) -> str:
        tail = ''
        return f''

class CrossAttentionRayRope(nn.Module):
    def __init__(
        self, embed_dim=768, kv_dim=4096, num_heads=12,
        proj_drop=0., tau=1, cos_attn=False,
    ):
        """
        Implements PRope
        :param embed_dim: model's width
        :param num_heads: num heads of multi-head attention
        :param proj_drop: always 0 for testing
        :param tau: always 1
        :param cos_attn: always True: during attention, q and k will be L2-normalized and scaled by a head-wise learnable parameter self.scale_mul_1H11
        :param customized_flash_attn:
        """
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.kv_dim = kv_dim

        self.num_heads, self.head_dim = num_heads, embed_dim // num_heads
        
        self.mat_q = nn.Linear(embed_dim, embed_dim, bias=True)
        self.mat_kv = nn.Linear(kv_dim, embed_dim*2, bias=False)
        self.v_bias = nn.Parameter(torch.zeros(embed_dim))
        self.register_buffer('zero_k_bias', torch.zeros(embed_dim))
        
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = get_dropout_layer(proj_drop)
        
        self.caching = False    # kv caching: only used during inference
        self.cached_k = None    # kv caching: only used during inference
        self.cached_v = None    # kv caching: only used during inference

        self.use_pd = True
        self.pd_type = "pj"

        self.use_p0 = True
        self.p0_type = "3d"

        self.pinf_type = "none"
        self.use_pinf = False

        init_depth = 0.0
        init_sigma = 3.0
        self.d_proj_weight = nn.Parameter(torch.zeros((2, embed_dim)))
        self.d_proj_bias = torch.nn.Parameter(torch.tensor([init_depth, init_sigma]))

        # self.patches_x = 16
        # self.patches_y = 16
        # self.num_patches = self.patches_x * self.patches_y

        self.num_rays_per_patch = 3
        # self.rope_coord_dim = 3*1 + 3*3*(1+0)
        self.rope_coord_dim = 3 * self.use_p0 + self.num_rays_per_patch * 3 * (int(self.use_pd) + int(self.use_pinf))
        self.rope_mat_dim = 2 * self.rope_coord_dim
        self.num_rope_freqs = self.head_dim // self.rope_mat_dim

        self.offsets = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        self.depth_type = "predict_dsig"
        self.denc_type = "d"

        self.freq_base = 3.0
        self.apply_vo = True

    
    def kv_caching(self, enable: bool): # kv caching: only used during inference
        self.caching = enable
        self.cached_k = None
        self.cached_v = None

    # NOTE: attn_bias_or_two_vector is None during inference
    def forward(self, 
                q,
                ca_kv, #[bs, N_views_src*N_tokens_per_view, 2048]
                poses,
                poses_src,
                intrs,
                intrs_src, 
                scale_schedule=None,
                scale_ind=None):
        """
        :param (fp32) x: shaped (B or batch_size, L or seq_length, C or hidden_dim); if seq-parallel is used, the `L` dim would be shared
        :param (fp32) attn_bias_or_two_vector:
                if not using_flash:
                    a block-wise, lower-triangle matrix, like:
                    [[[[0, -, -, -, -, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]]
                    where 0 means visible and - means invisible (-inf)
                else:
                    a tuple of two 1-dim int vector (VAR_visible_kvlen, VAR_invisible_qlen)
        :return: shaped (B or batch_size, L or seq_length, C or hidden_dim); if seq-parallel is used, the `L` dim would be shared
        """
        # x: fp32
        B, L, C = q.shape
        N_views = poses.shape[1]
        N_views_src = poses_src.shape[1]

        # print(f"[CA] ca_kv.shape: {ca_kv.shape}")
        # print(f"[CA] q.shape: {q.shape}")
        # raw_d: [B, N_views*N_tokens_per_view, 2]. 2:depth+confidence
        raw_d = F.linear(q, self.d_proj_weight, self.d_proj_bias)
        raw_d_kv = F.linear(ca_kv, self.d_proj_weight, self.d_proj_bias)

        q = self.mat_q(q).view(B, L, self.num_heads, self.head_dim).contiguous().permute(0,2,1,3)

        patches_per_view_src = ca_kv.shape[1] // N_views_src
        ca_kv = F.linear(ca_kv, 
                              weight=self.mat_kv.weight, 
                              bias=torch.cat((self.zero_k_bias, self.v_bias)))
        ca_kv = ca_kv.view(B, N_views_src*patches_per_view_src, 2, -1)
        
        k,v = ca_kv.unbind(dim=2)
        k = k.reshape(B, N_views_src*patches_per_view_src, self.num_heads, self.head_dim).permute(0,2,1,3)
        v = v.reshape(B, N_views_src*patches_per_view_src, self.num_heads, self.head_dim).permute(0,2,1,3)
        k = k.contiguous()
        v = v.contiguous()

        c2ws = poses  # [B, N, 4, 4] per camera
        Ks = intrs  # [B, N, 3, 3] per camera
        viewmats = torch.inverse(c2ws)

        c2ws_src = poses_src
        Ks_src = intrs_src
        viewmats_src = torch.inverse(c2ws_src)


        scale_schedule_last = scale_schedule[-1]
        # print(f"scale_schedule_last: {scale_schedule_last}")
        if scale_ind is not None:
            scale_schedule = [scale_schedule[scale_ind]]

        token_schedule = [h_i*w_i  for _,h_i,w_i in scale_schedule]

        # print(f"N_views: {N_views}")
        tok_start = 0
        out_final = []

        for i in range(len(token_schedule)):

            num_patches_i = token_schedule[i]
            tok_end = tok_start + N_views*num_patches_i

            self._precompute_and_cache_apply_fns(
                        w2cs=viewmats, 
                        Ks=Ks,
                        w2cs_kv=viewmats_src,
                        Ks_kv=Ks_src,
                        num_patches=num_patches_i,
                        num_patches_kv=patches_per_view_src,
                    )

            # q,k,v: [B,H,L,D//H]
            # sdpa_fn call
            apply_fn_q, all_apply_fns_kv, apply_fn_o = self._prepare_apply_fns(
                num_cameras=N_views,
                num_cameras_kv=N_views_src,
                patches_x=scale_schedule[i][-2],
                patches_y=scale_schedule[i][-1],
                patches_x_kv=scale_schedule_last[-2],
                patches_y_kv=scale_schedule_last[-1],
                predicted_d=raw_d[:, tok_start:tok_end],
                predicted_d_kv=raw_d_kv,
            )

            q_scale = q[:,:,tok_start:tok_end]
            q_scale = apply_fn_q(q_scale)

            out_scale = torch.zeros_like(q_scale)
            

            # print(f"k.shape: {k.shape}")
            # print(f"v.shape: {v.shape}")

            for cam_idx, apply_fn_kv in enumerate(all_apply_fns_kv):
                k_idx = apply_fn_kv(k)
                v_idx = apply_fn_kv(v)

                view_start = cam_idx*num_patches_i
                view_end = (cam_idx+1)*num_patches_i
                q_view = q_scale[:,:,view_start:view_end]

                # q_idx = q[:, :, cam_idx * num_patches_i : (cam_idx + 1) * num_patches_i, :]
                out_view = F.scaled_dot_product_attention(
                    query=q_view.contiguous(),
                    key=k_idx.contiguous(),
                    value=v_idx.contiguous(),
                )
                out_scale[:, :, view_start:view_end] = out_view

            out_scale = apply_fn_o(out_scale)
            out_final.append(out_scale)
            # out = out.contiguous()

        # print(f"raw_d.shape: {raw_d.shape}")
        # print(f"self.p0_world.shape: {self.p0_world.shape}")
        # print(f"self.P.shape: {self.P.shape}")
        # print(f"out.shape: {out.shape}")

        out = torch.cat(out_final, dim=2)
        out = out.transpose(1,2).reshape(B,L,C)

        return self.proj_drop(self.proj(out))
    

    def _precompute_and_cache_apply_fns(
        self, 
        w2cs: torch.Tensor, 
        Ks: Optional[torch.Tensor],
        w2cs_kv: torch.Tensor,
        Ks_kv: Optional[torch.Tensor],
        context_depths: Optional[torch.Tensor] = None,
        num_patches=None,
        num_patches_kv=None,
    ):
        (batch, num_cameras, _, _) = w2cs.shape
        (batch_kv, num_cameras_kv, _, _) = w2cs_kv.shape
        assert batch == batch_kv, "Batch size for Q and KV must be the same."
        
        self.batch = batch
        self.num_cameras = num_cameras
        self.num_cameras_kv = num_cameras_kv

        # Note: different from rayrope.py, here we assume context_depths are for KV views
        self.context_depths = context_depths

        self.w2cs = w2cs  # (batch, cameras, 4, 4)
        self.c2ws = _invert_SE3(w2cs)  # (batch, cameras, 4, 4)
        Ks_norm = Ks.clone()
        Ks_norm[..., 0, 2] -= 0.5
        Ks_norm[..., 1, 2] -= 0.5

        # Ks_norm = normalize_K(Ks, self.image_width, self.image_height)
        self.w2cs_kv = w2cs_kv  # (batch, cameras_kv, 4, 4)
        self.c2ws_kv = _invert_SE3(w2cs_kv)  # (batch, cameras_kv, 4, 4)
        # Ks_norm_kv = normalize_K(Ks_kv, self.image_width, self.image_height)
        Ks_norm_kv = Ks_kv.clone()
        Ks_norm_kv[..., 0, 2] -= 0.5
        Ks_norm_kv[..., 1, 2] -= 0.5

        # Compute the camera projection matrices we use in PRoPE.
        self.P = torch.einsum("...ij,...jk->...ik", _lift_K(Ks_norm), w2cs)
        self.P_T = self.P.transpose(-1, -2)
        self.P_inv = torch.einsum(
            "...ij,...jk->...ik",
            self.c2ws,
            _lift_K(_invert_K(Ks_norm)),
        )
        self.P_kv = torch.einsum("...ij,...jk->...ik", _lift_K(Ks_norm_kv), w2cs_kv)
        self.P_T_kv = self.P_kv.transpose(-1, -2)
        self.P_inv_kv = torch.einsum(
            "...ij,...jk->...ik",
            self.c2ws_kv,
            _lift_K(_invert_K(Ks_norm_kv)),
        )


        # get the ray segments in world coordinates
        self.p0_world = _get_cam_centers(self.c2ws, num_patches).unsqueeze(-2) # (batch, num_cameras, num_patches, 1, 4)
        # self.pinf_world = _get_point_coords(self.P_inv, self.patches_x, self.patches_y, self.offsets) # (batch, num_cameras, num_patches, num_rays_per_patch, 4)
        
        self.p0_world_kv = _get_cam_centers(self.c2ws_kv, num_patches_kv).unsqueeze(-2) # (batch, num_cameras_kv, num_patches, 1, 4)
        # self.pinf_world_kv = _get_point_coords(self.P_inv_kv, self.patches_x_kv, self.patches_y_kv, self.offsets) # (batch, num_cameras_kv, num_patches, num_rays_per_patch, 4)
        
        return
    

    # @torch.compile()
    def _prepare_apply_fns(
        self,
        num_cameras,
        num_cameras_kv,
        patches_x,
        patches_y,
        patches_x_kv,
        patches_y_kv,
        predicted_d: Optional[torch.Tensor] = None,  # (batch, seqlen, 1 or 2)
        predicted_d_kv: Optional[torch.Tensor] = None,
        # debug: bool = False,
        # positions_collector: Optional[dict] = None,
    ) -> list[Callable[[torch.Tensor], torch.Tensor]]:
        """Prepare transforms for PRoPE-style positional encoding.
        
        [_transform_to_query_frame] points_world.shape: torch.Size([2, 3, 1, 1, 4])
        [_transform_to_query_frame] points_world.shape: torch.Size([2, 2, 3, 1, 3, 4])

        points_world.shape: torch.Size([1, 3, 256, 1, 4])
        points_world.shape: torch.Size([2, 1, 3, 256, 3, 4])

        
        """
        # (batch, num_cameras, _, _) = w2cs.shape
        batch = self.batch
        # num_cameras = self.num_cameras
        # patches_x = self.patches_x
        # patches_y = self.patches_y
        # print(f"[CA]predicted_d.shape: {predicted_d.shape}")
        # print(f"[CA]predicted_d_kv.shape: {predicted_d_kv.shape}")

        num_rays_per_patch = self.num_rays_per_patch
        num_patches = patches_x * patches_y
        num_patches_kv = patches_x_kv * patches_y_kv

        # depths: [2,B,N_views,N_patches_per_view,num_rays_per_patch,1]
        # 2: predicted_d1, predicted_d2, i.e. d_min, d_max
        depths = _prepare_depths(predicted_d, self.context_depths, self.depth_type, 
                batch=batch, num_cameras=num_cameras, num_patches=num_patches, 
                patches_x=patches_x, patches_y=patches_y, 
                num_rays_per_patch=num_rays_per_patch, offsets=self.offsets)
        
        depths_kv = _prepare_depths(predicted_d_kv, self.context_depths, self.depth_type, 
                batch=batch, num_cameras=num_cameras_kv, num_patches=num_patches_kv, 
                patches_x=patches_x_kv, patches_y=patches_y_kv, 
                num_rays_per_patch=num_rays_per_patch, offsets=self.offsets)
    
        # print(f"depths.shape: {depths.shape}")
    
        # pd_world: [2, B, N_views, N_patches_per_view, num_rays_per_patch, 4]
        pd_world = _get_point_coords(self.P_inv, patches_x, patches_y, self.offsets, depths) # (2, batch, num_cameras, num_patches, num_rays_per_patch, 4)
        pd_world_kv = _get_point_coords(self.P_inv_kv, patches_x_kv, patches_y_kv, self.offsets, depths_kv) # (2, batch, num_cameras_kv, num_patches, num_rays_per_patch, 4)
        # print(f"pd_world.shape: {pd_world.shape}")

        positions_Q = defaultdict(list)
        # if positions_collector is not None:
        #     positions_collector["KV"] = []
        all_apply_fns_kv = []
        for cam_idx in range(num_cameras):
            positions_KV = {}

            P_q = self.P[:, cam_idx, :, :]  # (batch, 4, 4)
            w2c_q = self.w2cs[:, cam_idx, :, :]  # (batch, 4, 4)

            # self.p0_type: "3d"
            # self.denc_type: "d"
            # self.pd_type: 'pj'
            # transform camera centers to view0 coord system
            p0_3d = _transform_to_query_frame(self.p0_world, P_q, w2c_q, self.p0_type, self.denc_type)
            p0_3d = p0_3d.flatten(-2, -1)
            positions_Q['p0'].append(p0_3d[:, cam_idx])

            p0_3d_kv = _transform_to_query_frame(self.p0_world_kv, P_q, w2c_q, self.p0_type, self.denc_type)
            p0_3d_kv = p0_3d_kv.flatten(-2, -1)
            positions_KV['p0'] = p0_3d_kv            

            # points at infinity
            # if self.pinf_type == '3d':
            #     pinf_3d = _transform_to_query_frame(self.pinf_world, P_q, w2c_q, self.pinf_type, self.denc_type, norm_by='length')
            #     pinf_3d = pinf_3d.flatten(-2, -1)
            #     positions_KV['pinf_dir'] = pinf_3d
            #     positions_Q['pinf_dir'].append(pinf_3d[:, cam_idx])
            # elif self.pinf_type == 'pj':
            #     pinf_dir, _ = _transform_to_query_frame(self.pinf_world, P_q, w2c_q, self.pinf_type, self.denc_type)
            #     pinf_dir = pinf_dir.flatten(-2, -1)
            #     positions_KV['pinf_dir'] = pinf_dir
            #     positions_Q['pinf_dir'].append(pinf_dir[:, cam_idx])
            #     # no need to include depth here

            pd_dir, pd_d = _transform_to_query_frame(pd_world, P_q, w2c_q, self.pd_type, self.denc_type)
            pd_dir = pd_dir.flatten(0, 1)[..., :2].flatten(start_dim=-2, end_dim=-1)
            pd_d = pd_d.flatten(0, 1).flatten(start_dim=-2, end_dim=-1)
            positions_Q['pd_dir'].append(pd_dir[:, cam_idx])

            pd_dir_kv, pd_d_kv = _transform_to_query_frame(pd_world_kv, P_q, w2c_q, self.pd_type, self.denc_type)
            pd_dir_kv = pd_dir_kv.flatten(0, 1)[..., :2].flatten(start_dim=-2, end_dim=-1)
            pd_d_kv = pd_d_kv.flatten(0, 1).flatten(start_dim=-2, end_dim=-1)
            positions_KV['pd_dir'] = pd_dir_kv

            if self.denc_type == 'inv_d':
                positions_KV['pd_disparity'] = pd_d_kv
                positions_Q['pd_disparity'].append(pd_d[:, cam_idx])
            elif self.denc_type == 'd':
                positions_KV['pd_depth'] = pd_d_kv
                positions_Q['pd_depth'].append(pd_d[:, cam_idx])
            elif self.denc_type == 'asinh_d':
                positions_KV['pd_asinh_depth'] = pd_d_kv
                positions_Q['pd_asinh_depth'].append(pd_d[:, cam_idx])


            cos_KV, sin_KV = _prepare_rope_coeff_uniformd(positions_KV, 
                                                          self.num_rope_freqs, 
                                                          self.freq_base, 
                                                          batch, 
                                                          num_cameras_kv, 
                                                          num_patches_kv) # (batch, num_cameras, num_patches, num_freqs, num_coord)

            # --- FIX START: PAD KV ---
            target_dim = self.head_dim // 2
            if cos_KV.shape[-1] < target_dim:
                pad_amt = target_dim - cos_KV.shape[-1]
                # Pad cosine with 1.0 (Identity)
                cos_KV = F.pad(cos_KV, (0, pad_amt), value=1.0)
                # Pad sine with 0.0 (No rotation)
                sin_KV = F.pad(sin_KV, (0, pad_amt), value=0.0)
            # --- FIX END ---

            # print(f"cos_KV.shape: {cos_KV.shape}")
            # print(f"sin_KV.shape: {sin_KV.shape}")

            apply_fn_kv = partial(_apply_rope_coeffs, cos=cos_KV, sin=sin_KV, inverse=True)
            
            all_apply_fns_kv.append(apply_fn_kv)

        for key, val in positions_Q.items():
            positions_Q[key] = torch.stack(val, dim=1)  # (batch, num_cameras, num_patches, ...)

        # if positions_collector is not None:
        #     positions_collector["Q"] = positions_Q

        # print(f"positions_Q.keys(): {positions_Q.keys()}")

        # print(f"positions_Q['p0'].shape: {positions_Q['p0'].shape}")
        # print(f"positions_Q['pd_dir'].shape: {positions_Q['pd_dir'].shape}")
        # print(f"positions_Q['pd_depth'].shape: {positions_Q['pd_depth'].shape}")

        cos_Q, sin_Q = _prepare_rope_coeff_uniformd(positions_Q, 
                                                    self.num_rope_freqs, 
                                                    self.freq_base, 
                                                    batch, 
                                                    num_cameras, 
                                                    num_patches)

        # --- FIX START: PAD Q ---
        if cos_Q.shape[-1] < target_dim:
            pad_amt = target_dim - cos_Q.shape[-1]
            cos_Q = F.pad(cos_Q, (0, pad_amt), value=1.0)
            cos_Q = cos_Q.contiguous() # Ensure contiguous after padding
            
            sin_Q = F.pad(sin_Q, (0, pad_amt), value=0.0)
            sin_Q = sin_Q.contiguous()
        # --- FIX END ---

        # print(f"cos_Q.shape: {cos_Q.shape}")
        # print(f"sin_Q.shape: {sin_Q.shape}")

        apply_fn_q = partial(_apply_rope_coeffs, cos=cos_Q, sin=sin_Q, inverse=True)
        apply_fn_o = partial(_apply_rope_coeffs, cos=cos_Q, sin=sin_Q, inverse=False)

        # if (not torch.isfinite(cos_Q).all()) or (not torch.isfinite(sin_Q).all()):
        #     raise ValueError("NaN/inf values found in rope_matrices_Q.")

        return apply_fn_q, all_apply_fns_kv, apply_fn_o

    def extra_repr(self) -> str:
        tail = ''
        return f''

    

class SelfAttentionPropePrefixed(nn.Module):
    def __init__(
        self, embed_dim=768, num_heads=12,
        proj_drop=0., tau=1, cos_attn=False, customized_flash_attn=True, use_flex_attn=False, 
        batch_size=2, pad_to_multiplier=1, rope2d_normalized_by_hw=0, N_views=1
    ):
        """
        Implements PRope
        :param embed_dim: model's width
        :param num_heads: num heads of multi-head attention
        :param proj_drop: always 0 for testing
        :param tau: always 1
        :param cos_attn: always True: during attention, q and k will be L2-normalized and scaled by a head-wise learnable parameter self.scale_mul_1H11
        :param customized_flash_attn:
        """
        super().__init__()
        assert embed_dim % num_heads == 0
        self.using_flash = customized_flash_attn
        
        self.num_heads, self.head_dim = num_heads, embed_dim // num_heads
        self.tau, self.cos_attn = tau, cos_attn
        if self.cos_attn:
            self.scale = 1
            size = (1, 1, self.num_heads, 1) if self.using_flash else (1, self.num_heads, 1, 1)
            # size: 11H1 or 1H11
            self.scale_mul_1H11 = nn.Parameter(torch.full(size=size, fill_value=4.0).log(), requires_grad=True)
            self.max_scale_mul = torch.log(torch.tensor(100)).item()
        else:
            self.scale = 1 / math.sqrt(self.head_dim) / self.tau
        
        self.mat_qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.q_bias, self.v_bias = nn.Parameter(torch.zeros(embed_dim)), nn.Parameter(torch.zeros(embed_dim))
        self.register_buffer('zero_k_bias', torch.zeros(embed_dim))
        
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = get_dropout_layer(proj_drop)
        
        self.caching = False    # kv caching: only used during inference
        self.cached_k = None    # kv caching: only used during inference
        self.cached_v = None    # kv caching: only used during inference

        self.use_flex_attn = use_flex_attn
        self.pad_to_multiplier = pad_to_multiplier

        self.rope2d_normalized_by_hw = rope2d_normalized_by_hw
    
    def kv_caching(self, enable: bool): # kv caching: only used during inference
        self.caching = enable
        self.cached_k = None
        self.cached_v = None

    def prope_dot_product_vectorized(
        self, q, k, v, 
              poses, intrs,
              poses_src, intrs_src, 
              scale_schedule, 
              scale_ind=None, 
              attn_mask=None, 
              **kwargs
    ):
        """
        Vectorized implementation that avoids Python loops over sequence chunks.
        """
        B, num_heads, seqlen, head_dim = q.shape
        N_views = poses.shape[1]
        N_views_src = poses_src.shape[1]
        device = q.device

        token_schedule = [h_i*w_i  for _,h_i,w_i in scale_schedule]
        L_prefix = N_views_src * np.sum(token_schedule)

        # --- PREPARATION PHASE (Run once per batch) ---

        # during inference this is done at first step and then cached
        if self.cached_k is None:
            # 1. prepare PROPE for src prefix
            P_src, P_T_src, P_inv_src = get_prope_matrices(poses_c2w=poses_src, intrs=None)

            pos_x_src = []
            pos_y_src = []
            P_T_expanded_src = []
            P_inv_expanded_src = []
            P_expanded_src = []


            px_last = scale_schedule[-1][1]
            py_last = scale_schedule[-1][2]

            for _, px, py in scale_schedule:
                num_repeats = px * py
                
                # Grid generation
                x_grid = torch.arange(px, device=device).repeat(py * N_views_src)
                y_grid = torch.arange(py, device=device).repeat_interleave(px).repeat(N_views_src)

                # x_grid = px_last * ((x_grid+0.5) / px)
                # y_grid = py_last * ((y_grid+0.5) / py)

                x_grid = ((x_grid+0.5) / px)
                y_grid = ((y_grid+0.5) / py)

                pos_x_src.append(x_grid)
                pos_y_src.append(y_grid)
                
                # Matrix Expansion: Repeat camera matrices to match token count
                # P shape: (B, C, 4, 4) -> (B, C * px * py, 4, 4)
                P_T_expanded_src.append(P_T_src.repeat_interleave(num_repeats, dim=1))
                P_inv_expanded_src.append(P_inv_src.repeat_interleave(num_repeats, dim=1))
                P_expanded_src.append(P_src.repeat_interleave(num_repeats, dim=1))

            pos_x_src = torch.cat(pos_x_src, dim=0)
            pos_y_src = torch.cat(pos_y_src, dim=0)
            P_T_expanded_src = torch.cat(P_T_expanded_src, dim=1) 
            P_inv_expanded_src = torch.cat(P_inv_expanded_src, dim=1)
            P_expanded_src = torch.cat(P_expanded_src, dim=1)

        if scale_ind is not None:
            scale_schedule = [scale_schedule[scale_ind]]

        

        # 2. prepare PROPE for tgt
        P, P_T, P_inv = get_prope_matrices(poses_c2w=poses, intrs=None)

        pos_x = []
        pos_y = []
        P_T_expanded = []
        P_inv_expanded = []
        P_expanded = []

        px_last = scale_schedule[-1][1]
        py_last = scale_schedule[-1][2]

        for _, px, py in scale_schedule:
            num_repeats = px * py
            
            # Grid generation
            x_grid = torch.arange(px, device=device).repeat(py * N_views)
            y_grid = torch.arange(py, device=device).repeat_interleave(px).repeat(N_views)
            
            # x_grid = px_last * ((x_grid+0.5) / px)
            # y_grid = py_last * ((y_grid+0.5) / py)

            x_grid = ((x_grid+0.5) / px)
            y_grid = ((y_grid+0.5) / py)
            
            pos_x.append(x_grid)
            pos_y.append(y_grid)

            # print(f"px: {px} [tgt]x_grid: {x_grid}")
            # print(f"py: {py} [tgt]y_grid: {y_grid}")
            
            # Matrix Expansion: Repeat camera matrices to match token count
            # P shape: (B, C, 4, 4) -> (B, C * px * py, 4, 4)
            P_T_expanded.append(P_T.repeat_interleave(num_repeats, dim=1))
            P_inv_expanded.append(P_inv.repeat_interleave(num_repeats, dim=1))
            P_expanded.append(P.repeat_interleave(num_repeats, dim=1))

        pos_x = torch.cat(pos_x, dim=0)
        pos_y = torch.cat(pos_y, dim=0)
        P_T_expanded = torch.cat(P_T_expanded, dim=1) 
        P_inv_expanded = torch.cat(P_inv_expanded, dim=1)
        P_expanded = torch.cat(P_expanded, dim=1)

        # concat only at first step, then prefixed
        if self.cached_k is None:
            # 3. stack together prefix and tgt
            pos_x = torch.cat((pos_x_src, pos_x), dim=0)
            pos_y = torch.cat((pos_y_src, pos_y), dim=0)
            
            P_T_expanded = torch.cat((P_T_expanded_src, P_T_expanded), dim=1)
            P_inv_expanded = torch.cat((P_inv_expanded_src, P_inv_expanded), dim=1)
            P_expanded = torch.cat((P_expanded_src, P_expanded), dim=1)

        # 4. compute RoPE coeffs
        coeffs_x = self._rope_precompute_coeffs(pos_x, 10000.0, 1.0, head_dim // 4)
        coeffs_y = self._rope_precompute_coeffs(pos_y, 10000.0, 1.0, head_dim // 4)

        # print(f"q.shape: {q.shape}")
        # print(f"P_T_expanded.shape: {P_T_expanded.shape}")

        # 5. Apply transforms to Q, K, V
        q = self._apply_transform(q, P_T_expanded, coeffs_x, coeffs_y, inverse_rope=False)
        k = self._apply_transform(k, P_inv_expanded, coeffs_x, coeffs_y, inverse_rope=False)
        v = self._apply_transform(v, P_expanded, coeffs_x, coeffs_y, inverse_rope=False)

        # 6. self attn        
        if self.caching:
            if self.cached_k is None:
                self.cached_k = k
                self.cached_v = v
            else:
                # print(f"self.cached_k.shape: {self.cached_k.shape}")
                k = self.cached_k = torch.cat([self.cached_k, k], dim=2)
                v = self.cached_v = torch.cat([self.cached_v, v], dim=2)
        
        # print(f"[1]q.shape: {q.shape}")
        # print(f"[1]k.shape: {k.shape}")

        out = F.scaled_dot_product_attention(
            query=q, 
            key=k, 
            value=v, 
            attn_mask=attn_mask, 
            # scale=self.scale, 
            **kwargs
        )
        # print(f"out.shape: {out.shape}")
        
        # Apply output transform
        out = self._apply_transform(out, P_expanded, coeffs_x, coeffs_y, inverse_rope=True)
        
        return out

    # Helper to apply the 3-part block diagonal transform efficiently
    def _apply_transform(self, x, mat_seq, coeffs_x, coeffs_y, inverse_rope=False):
        # Split features: [Half (Proj), Quarter (RoPE X), Quarter (RoPE Y)]

        B, H, L, D = x.shape

        # x: (B, H, L, D)
        d_half = D // 2
        d_quart = D // 4
        
        x_proj, x_rope_x, x_rope_y = torch.split(x, [d_half, d_quart, d_quart], dim=-1)
        
        # 1. Apply Projection Matrix (Batched MatMul)
        # x_proj: (B, H, L, D/2) -> reshape to (B, H, L, D/8, 4)
        # mat_seq: (B, L, 4, 4) -> broadcast over H and D/8
        # Logic: We multiply the last dim 4 by the 4x4 matrix
        x_proj_reshaped = x_proj.view(B, H, L, -1, 4)
        # Einsum: b=batch, h=head, l=seq, k=chunk_idx, i/j=matrix dims
        # mat_seq is (b, l, i, j)
        x_proj_out = torch.einsum("blij, bhlkj -> bhlki", mat_seq, x_proj_reshaped)
        x_proj_out = x_proj_out.reshape(B, H, L, d_half)
        
        # 2. Apply RoPE
        x_rope_x_out = self._rope_apply_coeffs(x_rope_x, coeffs_x, inverse=inverse_rope)
        x_rope_y_out = self._rope_apply_coeffs(x_rope_y, coeffs_y, inverse=inverse_rope)
        
        return torch.cat([x_proj_out, x_rope_x_out, x_rope_y_out], dim=-1)

    # NOTE: attn_bias_or_two_vector is None during inference
    def forward(self, 
                x,
                attn_bias_or_two_vector: Union[torch.Tensor, Tuple[torch.IntTensor, torch.IntTensor]], 
                attn_fn=None, 
                scale_schedule=None, 
                scale_ind=None,
                poses=None,
                intrs=None,
                poses_src=None,
                intrs_src=None,
                input_size=None):

        # x: fp32
        B, L, C = x.shape

        # qkv: amp, bf16
        # qkv: [B,L,3,16,128] [2,3,3,16,128]
        qkv = F.linear(input=x, weight=self.mat_qkv.weight, bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias))).view(B, L, 3, self.num_heads, self.head_dim)  # BL3Hc
        
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
            
        if self.cos_attn:   # always True
            scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp() # 11H1 (flash), or 1H11 (not flash)
            q = F.normalize(q, dim=-1, eps=1e-12).mul(scale_mul).contiguous()   # fp32
            k = F.normalize(k, dim=-1, eps=1e-12).contiguous()                  # fp32
            v = v.contiguous()                                                  # bf16
        else:   # be contiguous, to make kernel happy
            q = q.contiguous()      # bf16
            k = k.contiguous()      # bf16
            v = v.contiguous()      # bf16

        oup = self.prope_dot_product_vectorized(q,k,v,
                                                poses=poses,
                                                intrs=intrs,
                                                poses_src=poses_src,
                                                intrs_src=intrs_src,
                                                scale_schedule=scale_schedule,
                                                scale_ind=scale_ind,
                                                attn_mask=attn_bias_or_two_vector)
        oup = oup.transpose(1,2).reshape(B,L,C)

        return self.proj_drop(self.proj(oup))
    
    def extra_repr(self) -> str:
        tail = ''
        return f'using_flash={self.using_flash}, tau={self.tau}, cos_attn={self.cos_attn}{tail}'

    # --- Helper Static Methods (Inlined for speed/simplicity) ---

    @staticmethod
    def _lift_K(Ks):
        out = torch.zeros(Ks.shape[:-2] + (4, 4), device=Ks.device, dtype=Ks.dtype)
        out[..., :3, :3] = Ks
        out[..., 3, 3] = 1.0
        return out

    @staticmethod
    def _invert_K(Ks):
        out = torch.zeros_like(Ks)
        out[..., 0, 0] = 1.0 / Ks[..., 0, 0]
        out[..., 1, 1] = 1.0 / Ks[..., 1, 1]
        out[..., 0, 2] = -Ks[..., 0, 2] / Ks[..., 0, 0]
        out[..., 1, 2] = -Ks[..., 1, 2] / Ks[..., 1, 1]
        out[..., 2, 2] = 1.0
        return out

    @staticmethod
    def _invert_SE3(transforms):
        Rinv = transforms[..., :3, :3].transpose(-1, -2)
        out = torch.zeros_like(transforms)
        out[..., :3, :3] = Rinv
        out[..., :3, 3] = -torch.einsum("...ij,...j->...i", Rinv, transforms[..., :3, 3])
        out[..., 3, 3] = 1.0
        return out

    @staticmethod
    def _rope_precompute_coeffs(positions, freq_base, freq_scale, feat_dim):
        num_freqs = feat_dim // 2
        freqs = freq_scale * (freq_base ** (-torch.arange(num_freqs, device=positions.device) / num_freqs))
        angles = positions[:, None] * freqs[None, :] # (Seq, Freqs)
        # Reshape for broadcasting: (1, 1, Seq, Freqs)
        angles = angles.view(1, 1, positions.shape[0], num_freqs)
        return torch.cos(angles), torch.sin(angles)

    @staticmethod
    def _rope_apply_coeffs(feats, coeffs, inverse=False):
        cos, sin = coeffs
        # Handle broadcasting if coeffs are smaller than feats
        if cos.shape[2] != feats.shape[2]:
             # This happens if we cached coeffs but seq length grew. 
             # In this specific vectorized impl, we usually regenerate coeffs, so this is just a safeguard.
             raise RuntimeError(f"RoPE coeffs length {cos.shape[2]} < features length {feats.shape[2]}")
             
        x_in = feats[..., : feats.shape[-1] // 2]
        y_in = feats[..., feats.shape[-1] // 2 :]
        
        if not inverse:
            return torch.cat([cos * x_in + sin * y_in, -sin * x_in + cos * y_in], dim=-1)
        else:
            return torch.cat([cos * x_in - sin * y_in, sin * x_in + cos * y_in], dim=-1)

class CrossAttention(nn.Module):
    def __init__(
        self, for_attn_pool=False, embed_dim=768, kv_dim=4096, num_heads=12,
        proj_drop=0., cos_attn=False,
    ):
        """
        :param for_attn_pool: only used in VAR.text_proj_for_sos
        :param embed_dim: Q's dim
        :param kv_dim: K's and V's dim
        :param num_heads: num heads of multi-head attention
        :param proj_drop: proj drop out
        :param cos_attn: during attention, q and k will be L2-normalized and scaled by a head-wise learnable parameter self.scale_mul_1H11
        """
        cos_attn = False    # TODO: never use cos attn in cross attention with T5 kv
        super().__init__()
        self.for_attn_pool = for_attn_pool
        self.embed_dim = embed_dim
        self.kv_dim = kv_dim
        assert embed_dim % num_heads == 0
        self.num_heads, self.head_dim = num_heads, embed_dim // num_heads  # =64
        self.cos_attn = cos_attn
        if self.cos_attn:
            self.scale = 1
            self.scale_mul_1H1 = nn.Parameter(torch.full(size=(1, self.num_heads, 1, 1), fill_value=4.0).log(), requires_grad=True)
            self.max_scale_mul = torch.log(torch.tensor(100)).item()
        else:
            self.scale = 1 / math.sqrt(self.head_dim)
        
        if for_attn_pool:
            q = torch.empty(1, self.num_heads, self.head_dim)
            nn.init.trunc_normal_(q, mean=0, std=math.sqrt(1 / embed_dim / 3))
            self.mat_q = nn.Parameter(q)
        else:
            self.mat_q = nn.Linear(embed_dim, embed_dim, bias=True)
        # [10-27 17:33:59] (infinity/models/basic.py, line 474)=> self.mat_kv.weight.shape: torch.Size([8388608])
        self.mat_kv = nn.Linear(kv_dim, embed_dim*2, bias=False)
        self.v_bias = nn.Parameter(torch.zeros(embed_dim))
        self.register_buffer('zero_k_bias', torch.zeros(embed_dim))
        
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = get_dropout_layer(proj_drop)
    
    def forward(self, q, ca_kv):
        """
        :param q: shaped as (batch, seq_len, Q_dim)
        :param ca_kv: contains several vectors, each of which is shaped as (len_i, KV_dim). We have [len_1xKV_dim, len_2xKV_dim, len_3xKV_dim, ...] and lens == [len_1, len_2, len_3, ...]
            - kv_compact: shaped as (sum(lens), KV_dim)
            - cu_seqlens_k: cumulated sum of lens
            - max_seqlen_k: int, max(lens)
        NOTE: seq_len (num of Qs) can reach 10k;  but len_i (num of KVs) must <= 256
        
        :return: shaped as (batch, seq_len, Q_dim)
        """
        kv_compact, cu_seqlens_k, max_seqlen_k = ca_kv
        N = kv_compact.shape[0]
        
        # print(f"self.mat_kv.weight.shape: {self.mat_kv.weight.shape}")
        

        # bias = torch.cat((self.zero_k_bias, self.v_bias))
        # out = self.mat_kv(kv_compact)
        # kv_compact = (out + bias).view(N, 2, self.num_heads, self.head_dim)

        # w = self.mat_kv.weight
        # if w.ndim==1:
        #     w = w.view(self.mat_kv.out_features, self.mat_kv.in_features)

        # print(f"w.shape: {w.shape}")

        # kv_compact: [256, 2, 16, 128] ,[n_tokens, 2(k,v), n_heads, head_dim]
        kv_compact = F.linear(kv_compact, 
                              weight=self.mat_kv.weight, 
                              bias=torch.cat((self.zero_k_bias, self.v_bias))).view(N, 2, self.num_heads, self.head_dim) # NC => N2Hc
        # attn_bias = xformers.ops.fmha.BlockDiagonalMask.from_seqlens
        # exit()

        # mat_q = self.mat_q
        # if mat_q.ndim==1:
        #     mat_q = mat_q.view(1, self.num_heads, self.head_dim)


        if not self.for_attn_pool:
            B, Lq = q.shape[:2]
            q_compact = self.mat_q(q).view(-1, self.num_heads, self.head_dim)
        else:
            B = cu_seqlens_k.shape[0] - 1
            Lq = 1
            q_compact = self.mat_q.repeat(B, 1, 1).to(dtype=kv_compact.dtype)
        if self.cos_attn:   # always False
            scale_mul = self.scale_mul_1H1.clamp_max(self.max_scale_mul).exp()
            k, v = kv_compact.unbind(dim=1)
            q_compact = F.normalize(q_compact, dim=-1).mul(scale_mul)
            k = F.normalize(k, dim=-1)
            kv_compact = torch.stack((k, v), dim=1)
        
        q_compact = q_compact.contiguous()
        kv_compact = kv_compact.contiguous()
        
        cu_seqlens_q = torch.arange(0, Lq * (B+1), Lq, dtype=torch.int32, device=q_compact.device)
        if q_compact.dtype == torch.float32:    # todo: fp16 or bf16?
            oup = flash_attn_varlen_kvpacked_func(q=q_compact.to(dtype=torch.bfloat16), kv=kv_compact.to(dtype=torch.bfloat16), cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, max_seqlen_q=Lq, max_seqlen_k=max_seqlen_k, dropout_p=0, softmax_scale=self.scale).reshape(B, Lq, -1)
            oup = oup.float()
        else:
            oup = flash_attn_varlen_kvpacked_func(q=q_compact, kv=kv_compact, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, max_seqlen_q=Lq, max_seqlen_k=max_seqlen_k, dropout_p=0, softmax_scale=self.scale).reshape(B, Lq, -1)
        
        return self.proj_drop(self.proj(oup))
    
    def extra_repr(self) -> str:
        return f'Cq={self.embed_dim}, Ckv={self.kv_dim}, cos_attn={self.cos_attn}'


class CrossAttentionPrope(nn.Module):
    def __init__(
        self, 
        embed_dim=768, 
        kv_dim=4096, 
        num_heads=12, 
        proj_drop=0., 
        cos_attn=False,
        N_views_src=1,
    ):

        super().__init__()
        self.embed_dim = embed_dim
        self.kv_dim = kv_dim
        assert embed_dim % num_heads == 0
        self.num_heads, self.head_dim = num_heads, embed_dim // num_heads  # =64

        self.scale = 1 / math.sqrt(self.head_dim)

        self.mat_q = nn.Linear(embed_dim, embed_dim, bias=True)
        self.mat_kv = nn.Linear(kv_dim, embed_dim*2, bias=False)
        self.v_bias = nn.Parameter(torch.zeros(embed_dim))
        self.register_buffer('zero_k_bias', torch.zeros(embed_dim))
        
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = get_dropout_layer(proj_drop)

        self.N_views_src = N_views_src


    def forward(self,
                q, # [B, N_views_tgt*N_tokens, D]
                ca_kv, # [kv_compact, cu_seqlens_k, max_seqlen_k], kv_compact: [B*N_views_src*N_tokens_per_view, D]
                poses,
                poses_src,
                intrs,
                intrs_src,
                scale_schedule,
                scale_ind,
                **kwargs): #[B*N_views_src*N_tokens_src, D]

        B, L, C = q.shape

        kv_compact, cu_seqlens_k, max_seqlen_k = ca_kv
        kv_compact = kv_compact.reshape(B,-1,C)

        viewmats = torch.linalg.inv(poses)
        n_cameras_tgt = viewmats.shape[1] 

        patches_per_view_src = kv_compact.shape[1] // self.N_views_src
        patches_x_src = patches_y_src = int(np.sqrt(patches_per_view_src))
        device=viewmats.device

        coeffs_x_src: Tuple[torch.Tensor, torch.Tensor] = _rope_precompute_coeffs(
            torch.tile(torch.arange(patches_x_src,device=device), (patches_y_src*self.N_views_src,)),
            freq_base=100.0,
            freq_scale=1.0,
            feat_dim=self.head_dim // 4,
        )

        # pos_y_scale = torch.tile(torch.repeat_interleave(torch.arange(py, device=device), px), (n_cameras_tgt,))
        coeffs_y_src: Tuple[torch.Tensor, torch.Tensor] = _rope_precompute_coeffs(
            torch.tile(torch.repeat_interleave(torch.arange(patches_y_src, device=device), patches_x_src), (self.N_views_src,)),
            freq_base=100.0,
            freq_scale=1.0,
            feat_dim=self.head_dim // 4,
        )

        viewmats_src = torch.linalg.inv(poses_src)

        _, apply_fn_kv_src, _ = _prepare_apply_fns(
                head_dim=self.head_dim,
                viewmats=viewmats_src,
                Ks=intrs_src,
                patches_x=patches_x_src,
                patches_y=patches_y_src,
                coeffs_x=coeffs_x_src,
                coeffs_y=coeffs_y_src
            )


        # [SAP] k_chunk.shape: torch.Size([1, 16, 2, 128]) [B, N_HEADS, N_views*N_tokens, D//N_Heads]
        kv_compact = F.linear(kv_compact, 
                              weight=self.mat_kv.weight, 
                              bias=torch.cat((self.zero_k_bias, self.v_bias)))#.view(B,N, 2, self.num_heads, self.head_dim)

        # split to k,v [B,N_views_src*N_patches_src, 2, D]
        kv_compact = kv_compact.view(B, self.N_views_src*patches_per_view_src, 2, -1)
        k,v = kv_compact.unbind(dim=2)

        k = k.contiguous()
        v = v.contiguous()

        k = k.reshape(B, self.N_views_src*patches_per_view_src, self.num_heads, self.head_dim).permute(0,2,1,3)
        v = v.reshape(B, self.N_views_src*patches_per_view_src, self.num_heads, self.head_dim).permute(0,2,1,3)
        
        #q: [B,L,C] -> [B,N_heads, L, D//N_heads]
        q = self.mat_q(q).view(B, L, self.num_heads, self.head_dim).contiguous().permute(0,2,1,3)
        # q = q.view(B, self.num_heads, L, self.head_dim).contiguous()

        k = apply_fn_kv_src(k)
        v = apply_fn_kv_src(v)

        patches_xs = [px for _,px,py in scale_schedule]
        patches_ys = [py for _,px,py in scale_schedule]

        device=viewmats.device
        pos_x_list = []
        for px, py in zip(patches_xs, patches_ys):
            pos_x_scale = torch.tile(torch.arange(px, device=device), (py * n_cameras_tgt,))
            pos_x_list.append(pos_x_scale)
        pos_x_total = torch.cat(pos_x_list, dim=0)
        coeffs_x = _rope_precompute_coeffs(pos_x_total, freq_base=100.0, freq_scale=1.0, feat_dim=self.head_dim // 4)

        pos_y_list = []
        for px, py in zip(patches_xs, patches_ys):
            pos_y_scale = torch.tile(torch.repeat_interleave(torch.arange(py, device=device), px), (n_cameras_tgt,))
            pos_y_list.append(pos_y_scale)
        pos_y_total = torch.cat(pos_y_list, dim=0)
        coeffs_y = _rope_precompute_coeffs(pos_y_total, freq_base=100.0, freq_scale=1.0, feat_dim=self.head_dim // 4)

        if scale_ind is not None:
            scale_schedule = [scale_schedule[scale_ind]]

        outs = []
        tok_start = 0
        for i in range(len(scale_schedule)):
            
            patches_x_tgt = scale_schedule[i][-1]
            patches_y_tgt = scale_schedule[i][-2]
            tok_end = tok_start + n_cameras_tgt*patches_x_tgt*patches_y_tgt

            apply_fn_q, _, apply_fn_o = _prepare_apply_fns(
                head_dim=self.head_dim,
                viewmats=viewmats,
                Ks=intrs,
                patches_x=patches_x_tgt,
                patches_y=patches_y_tgt,
                coeffs_x=(coeffs_x[0][:,:, tok_start:tok_end], coeffs_x[1][:,:, tok_start:tok_end]),
                coeffs_y=(coeffs_y[0][:,:, tok_start:tok_end], coeffs_y[1][:,:, tok_start:tok_end])
            )

            q_chunk = apply_fn_q(q[:, :, tok_start:tok_end])

            out_i = F.scaled_dot_product_attention(
                query=q_chunk,
                key=k,
                value=v,
                # scale=self.scale,
                # attn_mask=attn_mask_i,
            )
            out_i = apply_fn_o(out_i)
            outs.append(out_i)

            tok_start = tok_end

        outs = torch.cat(outs, dim=2) if len(outs)>1 else outs[0]
        outs = outs.transpose(1,2).reshape(B,L,C)
   
        return self.proj_drop(self.proj(outs))

class CrossAttentionPropeOptimized(nn.Module):

    def __init__(
        self, 
        embed_dim=768, 
        kv_dim=4096, 
        num_heads=12, 
        proj_drop=0., 
        cos_attn=False,
        N_views_src=1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kv_dim = kv_dim
        assert embed_dim % num_heads == 0
        self.num_heads, self.head_dim = num_heads, embed_dim // num_heads

        self.scale = 1 / math.sqrt(self.head_dim)

        self.mat_q = nn.Linear(embed_dim, embed_dim, bias=True)
        self.mat_kv = nn.Linear(kv_dim, embed_dim*2, bias=False)
        
        self.v_bias = nn.Parameter(torch.zeros(embed_dim))
        self.register_buffer('zero_k_bias', torch.zeros(embed_dim))

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0 else nn.Identity()

        self.N_views_src = N_views_src

    def forward(self, q, ca_kv, 
                poses, poses_src, 
                intrs, intrs_src, 
                scale_schedule, 
                scale_ind=None,
                rope2d_freqs_grid=None):
        B, L, C = q.shape
        device = q.device

        N_views_src = poses_src.shape[1]
        N_views_tgt = poses.shape[1]

        # [1] PREPARE SOURCE (K, V) ---
        # kv_compact = ca_kv
        kv_compact, _, _ = ca_kv
        kv_compact = kv_compact.reshape(B, -1, self.kv_dim)
        patches_per_view_src = kv_compact.shape[1] // N_views_src

        # Apply Linear + Bias
        # [B, L_src, 2*D]
        bias_kv = torch.cat((self.zero_k_bias, self.v_bias))
        kv_projected = F.linear(kv_compact, self.mat_kv.weight, bias=bias_kv)
        
        # Split into K, V and reshape to [B, H, L_src, D]
        kv_projected = kv_projected.view(B, -1, 2, self.embed_dim)
        k, v = kv_projected.unbind(dim=2)
        k = k.reshape(B, N_views_src*patches_per_view_src, self.num_heads, self.head_dim).permute(0,2,1,3)
        v = v.reshape(B, N_views_src*patches_per_view_src, self.num_heads, self.head_dim).permute(0,2,1,3)        

        src_cache_key = "src_full"
        tgt_cache_key = f"tgt_scale_{scale_ind}" if scale_ind is not None else "tgt_full"
        cached_src = None
        if rope2d_freqs_grid is not None:
             cached_src = rope2d_freqs_grid.get(src_cache_key)


        if cached_src is None:

            # [2] PREPARE SOURCE MATRICES [Proj, rope_x, rope_y]
            P_src, P_T_src, P_inv_src = get_prope_matrices(poses_c2w=poses_src, intrs=intrs_src)

            # pos_x_src = []
            # pos_y_src = []
            # P_T_expanded_src = []
            # P_inv_expanded_src = []
            # P_expanded_src = []
        
            # prepare matrices for src tokens(single scale)
            _, px_last, py_last = scale_schedule[-1]
            px_last,py_last = int(math.sqrt(patches_per_view_src)), int(math.sqrt(patches_per_view_src))

            num_repeats = px_last * py_last
            x_grid = torch.arange(px_last, device=device).repeat(py_last * N_views_src)
            y_grid = torch.arange(py_last, device=device).repeat_interleave(px_last).repeat(N_views_src)

            # x_grid = (x_grid+0.5) * px_last / px_last - px_last /2
            # y_grid = (y_grid+0.5) * py_last / py_last - py_last /2

            pos_x_src = x_grid
            pos_y_src = y_grid
            # pos_x_src.append(x_grid)
            # pos_y_src.append(y_grid)

            # P_T_expanded_src.append(P_T_src.repeat_interleave(num_repeats, dim=1))
            # P_inv_expanded_src.append(P_inv_src.repeat_interleave(num_repeats, dim=1))
            # P_expanded_src.append(P_src.repeat_interleave(num_repeats, dim=1))
            # P_T_expanded_src = torch.cat(P_T_expanded_src, dim=1) 
            # P_inv_expanded_src = torch.cat(P_inv_expanded_src, dim=1)
            # P_expanded_src = torch.cat(P_expanded_src, dim=1)

            # pos_x_src = torch.cat(pos_x_src, dim=0)
            # pos_y_src = torch.cat(pos_y_src, dim=0)
            # P_inv_expanded_src = P_inv_src

            coeffs_x_src = self._rope_precompute_coeffs(pos_x_src, 100.0, 1.0, self.head_dim // 4)
            coeffs_y_src = self._rope_precompute_coeffs(pos_y_src, 100.0, 1.0, self.head_dim // 4)

            cached_src = (P_inv_src, coeffs_x_src, coeffs_y_src)
            if rope2d_freqs_grid is not None:
                rope2d_freqs_grid[src_cache_key] = cached_src

        P_inv_src, coeffs_x_src, coeffs_y_src = cached_src
    
        # print(f"[CA] k.shape: {k.shape}")
        # print(f"[CA] P_inv_src.shape: {P_inv_src.shape}")

        # [3] apply transform to source KV
        k = self._apply_transform(k, P_inv_src, coeffs_x_src, coeffs_y_src, inverse_rope=False)
        v = self._apply_transform(v, P_inv_src, coeffs_x_src, coeffs_y_src, inverse_rope=False)

        # [4] Q projection
        # Project Q: [B, L, C] -> [B, H, L, D]
        q = self.mat_q(q).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        

        cached_tgt = None
        if rope2d_freqs_grid is not None:
            cached_tgt = rope2d_freqs_grid.get(tgt_cache_key)

            # print(f"rope2d_freqs_grid.keys(): {rope2d_freqs_grid.keys()}")

        if cached_tgt is None:
            if scale_ind is not None:
                current_schedule = [scale_schedule[scale_ind]]
            else:
                current_schedule = scale_schedule

            # prepare tgt matrices
            P_tgt, P_T_tgt, P_inv_tgt = get_prope_matrices(poses_c2w=poses, intrs=intrs)


            is_single_scale = (len(current_schedule) == 1)
            pos_x_tgt, pos_y_tgt = [], []
            P_T_list, P_list = [], [] # We only need P_T (for Q) and P (for Out) in CA

            # P_expanded_tgt = []
            # P_T_expanded_tgt = []
            # P_inv_expanded_tgt = []
            # pos_x_tgt = []
            # pos_y_tgt = []

            # Build Multi-scale Target Sequences
            for _, px, py in current_schedule:
                num_repeats = px * py

                x_grid = torch.arange(px, device=device).repeat(py * N_views_tgt)
                y_grid = torch.arange(py, device=device).repeat_interleave(px).repeat(N_views_tgt)

                # x_grid = (x_grid+0.5) * px_last / px - px_last /2
                # y_grid = (y_grid+0.5) * py_last / py - py_last /2


                pos_x_tgt.append(x_grid)
                pos_y_tgt.append(y_grid)

                if not is_single_scale:
                    P_T_list.append(P_T_tgt.repeat_interleave(num_repeats, dim=1))
                    P_list.append(P_tgt.repeat_interleave(num_repeats, dim=1))

                # P_T_expanded_tgt.append(P_T_tgt.repeat_interleave(num_repeats, dim=1))
                # P_inv_expanded_tgt.append(P_inv_tgt.repeat_interleave(num_repeats, dim=1))
                # P_expanded_tgt.append(P_tgt.repeat_interleave(num_repeats, dim=1))

            pos_x_tgt = torch.cat(pos_x_tgt, dim=0)
            pos_y_tgt = torch.cat(pos_y_tgt, dim=0)
            coeffs_x_tgt = self._rope_precompute_coeffs(pos_x_tgt, 100.0, 1.0, self.head_dim // 4)
            coeffs_y_tgt = self._rope_precompute_coeffs(pos_y_tgt, 100.0, 1.0, self.head_dim // 4)

            if is_single_scale:
                # FAST PATH
                P_T_expanded_tgt = P_T_tgt
                P_expanded_tgt = P_tgt
            else:
                # SLOW PATH
                P_T_expanded_tgt = torch.cat(P_T_list, dim=1) 
                P_expanded_tgt = torch.cat(P_list, dim=1)


            cached_tgt = (P_T_expanded_tgt, None, P_expanded_tgt, coeffs_x_tgt, coeffs_y_tgt)
            if rope2d_freqs_grid is not None:
                rope2d_freqs_grid[tgt_cache_key] = cached_tgt

        P_T_expanded_tgt, _, P_expanded_tgt, coeffs_x_tgt, coeffs_y_tgt = cached_tgt

        # if len(cached_tgt) == 5: # It came from SA (P_T, P_inv, P, cx, cy)
        #      P_T_expanded_tgt, _, P_expanded_tgt, coeffs_x_tgt, coeffs_y_tgt = cached_tgt
        # else: # It came from CA local compute (P_T, P, cx, cy)
        #      P_T_expanded_tgt, P_expanded_tgt, coeffs_x_tgt, coeffs_y_tgt = cached_tgt

        # if scale_ind is not None:
        #     scale_schedule = [scale_schedule[scale_ind]]
            
        q = self._apply_transform(q, P_T_expanded_tgt, coeffs_x_tgt, coeffs_y_tgt, inverse_rope=False)

        from torch.nn.attention import SDPBackend, sdpa_kernel
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
            )
        # --- 3. ATTENTION ---
        ##NOTE: comment out for CA visualization 
        # out = F.scaled_dot_product_attention(
        #     query=q,
        #     key=k,
        #     value=v,
            # scale=self.scale
        # )

        ##NOTE: uncomment for CA visualization 
        # scale = 1.0 / math.sqrt(q.size(-1))
        # attn_weight = q @ k.transpose(-2, -1) * scale
        # attn_weight = torch.softmax(attn_weight, dim=-1)        
        # global GLOBAL_SAVED_CROSS_ATTN
        # GLOBAL_SAVED_CROSS_ATTN.append(attn_weight.detach().cpu())
        # out = attn_weight @ v

        # --- 4. OUTPUT TRANSFORM ---
        out = self._apply_transform(out, P_expanded_tgt, coeffs_x_tgt, coeffs_y_tgt, inverse_rope=True)

        out = out.transpose(1, 2).reshape(B, L, C)
        return self.proj_drop(self.proj(out))

    # --- Helper Methods ---

    def _apply_transform(self, x, mat_seq_or_cameras, coeffs_x, coeffs_y, inverse_rope=False):
        """
        x: (B, H, L, D)
        mat: (B, L, 4, 4) or (B, Cams, 4, 4)
        Applies M @ x (Matrix-Vector product). 
        """
        B, H, L, D = x.shape
        d_half = D // 2
        d_quart = D // 4

        x_proj, x_rope_x, x_rope_y = torch.split(x, [d_half, d_quart, d_quart], dim=-1)

        
        # --- Projection block ---
        # Efficient Tiling check: if mat is per-camera, use broadcast einsum
        if mat_seq_or_cameras.dim() == 4 and mat_seq_or_cameras.shape[1] <= L and L % mat_seq_or_cameras.shape[1] == 0:
            
            mat = mat_seq_or_cameras 
            cameras = mat.shape[1]
            patches_per_camera = L // cameras
            
            x_proj_c = x_proj.contiguous()
            x_proj_reshaped = x_proj_c.view(B, H, cameras, patches_per_camera, d_half // 4, 4)
            
            # einsum: ...ij, ...kj -> ...ki (Standard Matrix-Vector: M @ x)
            # mat: (B, Cams, 4, 4) -> ij
            # x:   (..., 4) -> kj (treated as column vector j)
            proj_out = torch.einsum("bcij,bncpkj->bncpki", mat, x_proj_reshaped)
            x_proj_out = proj_out.reshape(B, H, L, d_half).contiguous()
        else:
            # print(f"[CA] FULL")
            mat = mat_seq_or_cameras
            x_proj_c = x_proj.contiguous()
            x_proj_reshaped = x_proj_c.view(B, H, L, -1, 4)
            
            # einsum: ...ij, ...kj -> ...ki (Standard Matrix-Vector: M @ x)
            x_proj_out = torch.einsum("blij, bhlkj -> bhlki", mat, x_proj_reshaped).reshape(B, H, L, d_half).contiguous()

        # --- RoPE blocks ---
        x_rope_x_out = self._rope_apply_coeffs(x_rope_x, coeffs_x, inverse=inverse_rope)
        x_rope_y_out = self._rope_apply_coeffs(x_rope_y, coeffs_y, inverse=inverse_rope)

        return torch.cat([x_proj_out, x_rope_x_out, x_rope_y_out], dim=-1).contiguous()

    @staticmethod
    def _rope_precompute_coeffs(positions, freq_base, freq_scale, feat_dim):
        num_freqs = feat_dim // 2
        freqs = freq_scale * (freq_base ** (-torch.arange(num_freqs, device=positions.device) / num_freqs))
        angles = positions[:, None] * freqs[None, :]
        angles = angles.view(1, 1, positions.shape[0], num_freqs)
        return torch.cos(angles), torch.sin(angles)

    @staticmethod
    def _rope_apply_coeffs(feats, coeffs, inverse=False):
        cos, sin = coeffs
        if cos.shape[2] != feats.shape[2]:
            pass

        x_in = feats[..., : feats.shape[-1] // 2]
        y_in = feats[..., feats.shape[-1] // 2 :]
        if not inverse:
            return torch.cat((cos * x_in + sin * y_in, -sin * x_in + cos * y_in), dim=-1)
        else:
            return torch.cat((cos * x_in - sin * y_in, sin * x_in + cos * y_in), dim=-1)

class CrossAttentionPropeOptimized2(nn.Module):
    """
    # How to use PRoPE attention for cross-attention: 
        #  
        #    attn_src = PropeDotProductAttention(...) 
        #    attn_tgt = PropeDotProductAttention(...) 
        #    attn_src._precompute_and_cache_apply_fns(viewmats_src, Ks_src) 
        #    attn_tgt._precompute_and_cache_apply_fns(viewmats_tgt, Ks_tgt) 
        #    q_src = attn_src._apply_to_q(q_src) P_T_src
        #    k_tgt = attn_tgt._apply_to_kv(k_tgt) P_inv_tgt
        #    v_tgt = attn_tgt._apply_to_kv(v_tgt) P_inv_tgt
        #    o_src = F.scaled_dot_product_attention(q_src, k_tgt, v_tgt, **kwargs) 
        #    o_src = attn_src._apply_to_o(o_src) P_src
    """
    def __init__(
        self, 
        embed_dim=768, 
        kv_dim=4096, 
        num_heads=12, 
        proj_drop=0., 
        cos_attn=False,
        N_views_src=1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kv_dim = kv_dim
        assert embed_dim % num_heads == 0
        self.num_heads, self.head_dim = num_heads, embed_dim // num_heads

        self.scale = 1 / math.sqrt(self.head_dim)

        self.mat_q = nn.Linear(embed_dim, embed_dim, bias=True)
        self.mat_kv = nn.Linear(kv_dim, embed_dim*2, bias=False)
        
        self.v_bias = nn.Parameter(torch.zeros(embed_dim))
        self.register_buffer('zero_k_bias', torch.zeros(embed_dim))

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0 else nn.Identity()

        self.N_views_src = N_views_src

    def forward(self, q, ca_kv, 
                poses, poses_src, 
                intrs, intrs_src, 
                scale_schedule, 
                scale_ind=None,
                rope2d_freqs_grid=None):
        B, L, C = q.shape
        device = q.device

        N_views_src = poses_src.shape[1]
        N_views_tgt = poses.shape[1]

        # [1] PREPARE SOURCE (K, V) ---
        # kv_compact = ca_kv
        kv_compact, _, _ = ca_kv
        kv_compact = kv_compact.reshape(B, -1, self.kv_dim)
        patches_per_view_src = kv_compact.shape[1] // N_views_src

        # Apply Linear + Bias
        # [B, L_src, 2*D]
        bias_kv = torch.cat((self.zero_k_bias, self.v_bias))
        kv_projected = F.linear(kv_compact, self.mat_kv.weight, bias=bias_kv)
        
        # Split into K, V and reshape to [B, H, L_src, D]
        kv_projected = kv_projected.view(B, -1, 2, self.embed_dim)
        k, v = kv_projected.unbind(dim=2)
        # print(f"self.N_views_src: {self.N_views_src}")
        # print(f"patches_per_view_src: {patches_per_view_src}")
        # print(f"k.shape: {k.shape}")
        k = k.reshape(B, N_views_src*patches_per_view_src, self.num_heads, self.head_dim).permute(0,2,1,3)
        v = v.reshape(B, N_views_src*patches_per_view_src, self.num_heads, self.head_dim).permute(0,2,1,3)        

        src_cache_key = "src_full"
        tgt_cache_key = f"tgt_scale_{scale_ind}" if scale_ind is not None else "tgt_full"
        cached_src = None
        if rope2d_freqs_grid is not None:
             cached_src = rope2d_freqs_grid.get(src_cache_key)


            #  print(f"rope2d_freqs_grid.keys(): {rope2d_freqs_grid.keys()}")

        if cached_src is None:


            # [2] PREPARE SOURCE MATRICES [Proj, rope_x, rope_y]
            P_src, P_T_src, P_inv_src = get_prope_matrices(poses_c2w=poses_src, intrs=None)

            pos_x_src = []
            pos_y_src = []
            P_T_expanded_src = []
            P_inv_expanded_src = []
            P_expanded_src = []
        
            # prepare matrices for src tokens(single scale)
            _, px_last, py_last = scale_schedule[-1]
            px_last,py_last = int(math.sqrt(patches_per_view_src)), int(math.sqrt(patches_per_view_src))

            num_repeats = px_last * py_last
            x_grid = torch.arange(px_last, device=device).repeat(py_last * N_views_src)
            y_grid = torch.arange(py_last, device=device).repeat_interleave(px_last).repeat(N_views_src)

            # x_grid = (x_grid+0.5) * px_last / px_last - px_last /2
            # y_grid = (y_grid+0.5) * py_last / py_last - py_last /2

            pos_x_src.append(x_grid)
            pos_y_src.append(y_grid)
            P_T_expanded_src.append(P_T_src.repeat_interleave(num_repeats, dim=1))
            P_inv_expanded_src.append(P_inv_src.repeat_interleave(num_repeats, dim=1))
            P_expanded_src.append(P_src.repeat_interleave(num_repeats, dim=1))

            pos_x_src = torch.cat(pos_x_src, dim=0)
            pos_y_src = torch.cat(pos_y_src, dim=0)

            P_T_expanded_src = torch.cat(P_T_expanded_src, dim=1) 
            P_inv_expanded_src = torch.cat(P_inv_expanded_src, dim=1)
            P_expanded_src = torch.cat(P_expanded_src, dim=1)

            coeffs_x_src = self._rope_precompute_coeffs(pos_x_src, 100.0, 1.0, self.head_dim // 4)
            coeffs_y_src = self._rope_precompute_coeffs(pos_y_src, 100.0, 1.0, self.head_dim // 4)

            cached_src = (P_inv_expanded_src, coeffs_x_src, coeffs_y_src)
            if rope2d_freqs_grid is not None:
                rope2d_freqs_grid[src_cache_key] = cached_src

        P_inv_expanded_src, coeffs_x_src, coeffs_y_src = cached_src

        # [3] apply transform to source KV
        k = self._apply_transform(k, P_inv_expanded_src, coeffs_x_src, coeffs_y_src, inverse_rope=False)
        v = self._apply_transform(v, P_inv_expanded_src, coeffs_x_src, coeffs_y_src, inverse_rope=False)

        # [4] Q projection
        # Project Q: [B, L, C] -> [B, H, L, D]
        q = self.mat_q(q).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        

        cached_tgt = None
        if rope2d_freqs_grid is not None:
            cached_tgt = rope2d_freqs_grid.get(tgt_cache_key)

            # print(f"rope2d_freqs_grid.keys(): {rope2d_freqs_grid.keys()}")

        
        if cached_tgt is None:
            if scale_ind is not None:
                current_schedule = [scale_schedule[scale_ind]]
            else:
                current_schedule = scale_schedule

            # prepare tgt matrices
            P_tgt, P_T_tgt, P_inv_tgt = get_prope_matrices(poses_c2w=poses, intrs=None)

            P_expanded_tgt = []
            P_T_expanded_tgt = []
            P_inv_expanded_tgt = []
            pos_x_tgt = []
            pos_y_tgt = []

            # Build Multi-scale Target Sequences
            for _, px, py in current_schedule:
                num_repeats = px * py

                x_grid = torch.arange(px, device=device).repeat(py * N_views_tgt)
                y_grid = torch.arange(py, device=device).repeat_interleave(px).repeat(N_views_tgt)

                # x_grid = (x_grid+0.5) * px_last / px - px_last /2
                # y_grid = (y_grid+0.5) * py_last / py - py_last /2


                pos_x_tgt.append(x_grid)
                pos_y_tgt.append(y_grid)

                P_T_expanded_tgt.append(P_T_tgt.repeat_interleave(num_repeats, dim=1))
                P_inv_expanded_tgt.append(P_inv_tgt.repeat_interleave(num_repeats, dim=1))
                P_expanded_tgt.append(P_tgt.repeat_interleave(num_repeats, dim=1))

            pos_x_tgt = torch.cat(pos_x_tgt, dim=0)
            pos_y_tgt = torch.cat(pos_y_tgt, dim=0)
            P_T_expanded_tgt = torch.cat(P_T_expanded_tgt, dim=1) 
            P_inv_expanded_tgt = torch.cat(P_inv_expanded_tgt, dim=1)
            P_expanded_tgt = torch.cat(P_expanded_tgt, dim=1)

            coeffs_x_tgt = self._rope_precompute_coeffs(pos_x_tgt, 100.0, 1.0, self.head_dim // 4)
            coeffs_y_tgt = self._rope_precompute_coeffs(pos_y_tgt, 100.0, 1.0, self.head_dim // 4)

            cached_tgt = (P_T_expanded_tgt, P_expanded_tgt, coeffs_x_tgt, coeffs_y_tgt)
            if rope2d_freqs_grid is not None:
                rope2d_freqs_grid[tgt_cache_key] = cached_tgt


        if len(cached_tgt) == 5: # It came from SA (P_T, P_inv, P, cx, cy)
             P_T_expanded_tgt, _, P_expanded_tgt, coeffs_x_tgt, coeffs_y_tgt = cached_tgt
        else: # It came from CA local compute (P_T, P, cx, cy)
             P_T_expanded_tgt, P_expanded_tgt, coeffs_x_tgt, coeffs_y_tgt = cached_tgt

        if scale_ind is not None:
            scale_schedule = [scale_schedule[scale_ind]]
            
            
        
        

        q = self._apply_transform(q, P_T_expanded_tgt, coeffs_x_tgt, coeffs_y_tgt, inverse_rope=False)

        # --- 3. ATTENTION ---
        out = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            # scale=self.scale
        )

        # --- 4. OUTPUT TRANSFORM ---
        out = self._apply_transform(out, P_expanded_tgt, coeffs_x_tgt, coeffs_y_tgt, inverse_rope=True)

        out = out.transpose(1, 2).reshape(B, L, C)
        return self.proj_drop(self.proj(out))

    # --- Helper Methods ---

    def _apply_transform(self, x, mat_seq_or_cameras, coeffs_x, coeffs_y, inverse_rope=False):
        """
        x: (B, H, L, D)
        mat: (B, L, 4, 4) or (B, Cams, 4, 4)
        Applies M @ x (Matrix-Vector product). 
        """
        B, H, L, D = x.shape
        d_half = D // 2
        d_quart = D // 4

        x_proj, x_rope_x, x_rope_y = torch.split(x, [d_half, d_quart, d_quart], dim=-1)

        # --- Projection block ---
        # Efficient Tiling check: if mat is per-camera, use broadcast einsum
        if mat_seq_or_cameras.dim() == 4 and mat_seq_or_cameras.shape[1] <= L and L % mat_seq_or_cameras.shape[1] == 0:
            mat = mat_seq_or_cameras 
            cameras = mat.shape[1]
            patches_per_camera = L // cameras
            
            x_proj_c = x_proj.contiguous()
            x_proj_reshaped = x_proj_c.view(B, H, cameras, patches_per_camera, d_half // 4, 4)
            
            # einsum: ...ij, ...kj -> ...ki (Standard Matrix-Vector: M @ x)
            # mat: (B, Cams, 4, 4) -> ij
            # x:   (..., 4) -> kj (treated as column vector j)
            proj_out = torch.einsum("bcij,bncpkj->bncpki", mat, x_proj_reshaped)
            x_proj_out = proj_out.reshape(B, H, L, d_half).contiguous()
        else:
            mat = mat_seq_or_cameras
            x_proj_c = x_proj.contiguous()
            x_proj_reshaped = x_proj_c.view(B, H, L, -1, 4)
            
            # einsum: ...ij, ...kj -> ...ki (Standard Matrix-Vector: M @ x)
            x_proj_out = torch.einsum("blij, bhlkj -> bhlki", mat, x_proj_reshaped).reshape(B, H, L, d_half).contiguous()

        # --- RoPE blocks ---
        x_rope_x_out = self._rope_apply_coeffs(x_rope_x, coeffs_x, inverse=inverse_rope)
        x_rope_y_out = self._rope_apply_coeffs(x_rope_y, coeffs_y, inverse=inverse_rope)

        return torch.cat([x_proj_out, x_rope_x_out, x_rope_y_out], dim=-1).contiguous()

    @staticmethod
    def _rope_precompute_coeffs(positions, freq_base, freq_scale, feat_dim):
        num_freqs = feat_dim // 2
        freqs = freq_scale * (freq_base ** (-torch.arange(num_freqs, device=positions.device) / num_freqs))
        angles = positions[:, None] * freqs[None, :]
        angles = angles.view(1, 1, positions.shape[0], num_freqs)
        return torch.cos(angles), torch.sin(angles)

    @staticmethod
    def _rope_apply_coeffs(feats, coeffs, inverse=False):
        cos, sin = coeffs
        if cos.shape[2] != feats.shape[2]:
            pass

        x_in = feats[..., : feats.shape[-1] // 2]
        y_in = feats[..., feats.shape[-1] // 2 :]
        if not inverse:
            return torch.cat((cos * x_in + sin * y_in, -sin * x_in + cos * y_in), dim=-1)
        else:
            return torch.cat((cos * x_in - sin * y_in, sin * x_in + cos * y_in), dim=-1)

class CrossAttentionPropeMultiscale(nn.Module):
    def __init__(
        self, 
        embed_dim=768, 
        kv_dim=4096, 
        num_heads=12, 
        proj_drop=0., 
        cos_attn=False,
        N_views_src=1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kv_dim = kv_dim
        assert embed_dim % num_heads == 0
        self.num_heads, self.head_dim = num_heads, embed_dim // num_heads

        self.scale = 1 / math.sqrt(self.head_dim)

        self.mat_q = nn.Linear(embed_dim, embed_dim, bias=True)
        self.mat_kv = nn.Linear(kv_dim, embed_dim*2, bias=False)
        
        self.v_bias = nn.Parameter(torch.zeros(embed_dim))
        self.register_buffer('zero_k_bias', torch.zeros(embed_dim))

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0 else nn.Identity()

        self.N_views_src = N_views_src

    def forward(self, 
                q, # [B, N_tokens_tgt, D]
                ca_kv, # [B,N_views_src*N_tokens_src, D] 
                poses, poses_src, # [B, N_views, 4, 4]
                intrs, intrs_src, # [B, N_views, 3, 3]
                scale_schedule, scale_ind=None):
        B, L, C = q.shape
        device = q.device

        # --- 1. PREPARE SOURCE (K, V) ---        
        patches_per_view_src = ca_kv.shape[1] // self.N_views_src

        # Apply projection
        # [B, L_src, 2*D]
        bias_kv = torch.cat((self.zero_k_bias, self.v_bias))
        kv_projected = F.linear(ca_kv, self.mat_kv.weight, bias=bias_kv)
        
        # Split into K, V and reshape to [B, H, L_src, D]
        kv_projected = kv_projected.view(B, -1, 2, self.embed_dim)
        k, v = kv_projected.unbind(dim=2)
        k = k.reshape(B, self.N_views_src*patches_per_view_src, self.num_heads, self.head_dim).permute(0,2,1,3)
        v = v.reshape(B, self.N_views_src*patches_per_view_src, self.num_heads, self.head_dim).permute(0,2,1,3)

        # calc P_src 
        P_src, P_T_src, P_inv_src = self._get_camera_matrices(poses_src, intrs_src)

        P_inv_src_expanded = []
        pos_x_expanded = []
        pos_y_expanded = []

        for _, px, py in scale_schedule:
            num_repeats = px*py
        
            # create (x,y) indices for all scales
            x_grid = torch.arange(px, device=device).repeat(py*self.N_views_src)
            y_grid = torch.arange(py, device=device).repeat_interleave(px).repeat(self.N_views_src)

            pos_x_expanded.append(x_grid)
            pos_y_expanded.append(y_grid)

            P_inv_src_expanded.append(P_inv_src.repeat_interleave(num_repeats, dim=1))

        # P_inv_src_expanded: [B,N_tokens, 4,4]
        P_inv_src_expanded = torch.cat(P_inv_src_expanded, dim=1)
        # pos_x_expanded: [N_tokens]
        pos_x_expanded = torch.cat(pos_x_expanded, dim=0)
        pos_y_expanded = torch.cat(pos_y_expanded, dim=0)
        
        # RoPE Coeffs Source []
        coeffs_x_src = self._rope_precompute_coeffs(pos_x_expanded, 100.0, 1.0, self.head_dim // 4)
        coeffs_y_src = self._rope_precompute_coeffs(pos_y_expanded, 100.0, 1.0, self.head_dim // 4)

        k = self._apply_transform(k, P_inv_src_expanded, coeffs_x_src, coeffs_y_src, inverse_rope=False)
        v = self._apply_transform(v, P_inv_src_expanded, coeffs_x_src, coeffs_y_src, inverse_rope=False)


        # --- 2. PREPARE TARGET (Q) ---
        # Project Q: [B, L, C] -> [B, H, L, D]
        q = self.mat_q(q).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        if scale_ind is not None:
            scale_schedule = [scale_schedule[scale_ind]]
            
        P_tgt, P_T_tgt, P_inv_tgt = self._get_camera_matrices(poses, intrs)
        
        P_tgt_expanded = []
        pos_x_expanded = []
        pos_y_expanded = []
        N_views_tgt = poses.shape[1]

        # Build Multi-scale Target Sequences
        for _, px, py in scale_schedule:
            num_repeats = px * py
            
            # Build grid indices
            x_grid = torch.arange(px, device=device).repeat(py * N_views_tgt)
            y_grid = torch.arange(py, device=device).repeat_interleave(px).repeat(N_views_tgt)
            
            pos_x_expanded.append(x_grid)
            pos_y_expanded.append(y_grid)

            # Expand matrices
            P_tgt_expanded.append(P_tgt.repeat_interleave(num_repeats, dim=1))

        # P_tgt_expanded: [B,N_tokens, 4,4]
        P_tgt_expanded = torch.cat(P_tgt_expanded, dim=1)
        P_tgt_T_expanded = P_tgt_expanded.transpose(-1, -2)
        # pos_x_expanded: [N_tokens]
        pos_x_expanded = torch.cat(pos_x_expanded, dim=0)
        pos_y_expanded = torch.cat(pos_y_expanded, dim=0)
        
        coeffs_x_tgt = self._rope_precompute_coeffs(pos_x_expanded, 100.0, 1.0, self.head_dim // 4)
        coeffs_y_tgt = self._rope_precompute_coeffs(pos_y_expanded, 100.0, 1.0, self.head_dim // 4)

        # Apply P_tgt 
        q = self._apply_transform(q, P_tgt_T_expanded, coeffs_x_tgt, coeffs_y_tgt, inverse_rope=False)

        # print(f"[ca] q.shape: {q.shape}")
        # print(f"[ca] k.shape: {k.shape}")
        # print(f"[ca] v.shape: {v.shape}")
        # --- 3. ATTENTION ---
        out = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            scale=self.scale
        )

        # --- 4. OUTPUT TRANSFORM ---
        out = self._apply_transform(out, P_tgt_expanded, coeffs_x_tgt, coeffs_y_tgt, inverse_rope=True)

        out = out.transpose(1, 2).reshape(B, L, C)
        return self.proj_drop(self.proj(out))

    # --- Helper Methods ---

    def _get_camera_matrices(self, poses_c2w, intrs):
        """
        Calculates P and P_inv.
        Matches Original: input `poses` is converted to `inv(poses)` (w2c) logic.
        P = Lift(K) @ inv(poses)  (World -> Screen)
        P_inv = poses @ Lift(K)^-1 (Screen -> World)
        """
        poses_w2c = self._invert_SE3(poses_c2w) # World-to-Camera (assuming poses is Camera-to-World)

        if intrs is not None:
            Ks_norm = intrs.clone()
            Ks_norm[..., 0, 2] -= 0.5
            Ks_norm[..., 1, 2] -= 0.5
            
            lifted_K = self._lift_K(Ks_norm)
            # P = K @ w2c
            P = torch.einsum("...ij,...jk->...ik", lifted_K, poses_w2c)

            # P_inv
            # view_inv = self._invert_SE3(pose_inv)
            lifted_K_inv = self._lift_K(self._invert_K(Ks_norm))
            
            # P_inv = c2w @ K_inv
            P_inv = torch.einsum("...ij,...jk->...ik", poses_c2w, lifted_K_inv)

            P_T = P.transpose(-1,-2)
        else:
            P = poses_w2c
            P_inv = poses_c2w
            P_T = P.transpose(-1,-2)
            
        return P, P_T, P_inv

    def _apply_transform(self, x, mat_seq_or_cameras, coeffs_x, coeffs_y, inverse_rope=False):
        """
        x: (B, H, L, D)
        mat: (B, L, 4, 4) or (B, Cams, 4, 4)
        Applies M @ x (Matrix-Vector product). 
        """
        B, H, L, D = x.shape
        d_half = D // 2
        d_quart = D // 4

        x_proj, x_rope_x, x_rope_y = torch.split(x, [d_half, d_quart, d_quart], dim=-1)

        # --- Projection block ---
        # Efficient Tiling check: if mat is per-camera, use broadcast einsum
        if mat_seq_or_cameras.dim() == 4 and mat_seq_or_cameras.shape[1] <= L and L % mat_seq_or_cameras.shape[1] == 0:
            mat = mat_seq_or_cameras 
            cameras = mat.shape[1]
            patches_per_camera = L // cameras
            
            x_proj_c = x_proj.contiguous()
            x_proj_reshaped = x_proj_c.view(B, H, cameras, patches_per_camera, d_half // 4, 4)
            
            # einsum: ...ij, ...kj -> ...ki (Standard Matrix-Vector: M @ x)
            # mat: (B, Cams, 4, 4) -> ij
            # x:   (..., 4) -> kj (treated as column vector j)
            proj_out = torch.einsum("bcij,bncpkj->bncpki", mat, x_proj_reshaped)
            x_proj_out = proj_out.reshape(B, H, L, d_half).contiguous()
        else:
            mat = mat_seq_or_cameras
            x_proj_c = x_proj.contiguous()
            x_proj_reshaped = x_proj_c.view(B, H, L, -1, 4)
            
            # einsum: ...ij, ...kj -> ...ki (Standard Matrix-Vector: M @ x)
            x_proj_out = torch.einsum("blij, bhlkj -> bhlki", mat, x_proj_reshaped).reshape(B, H, L, d_half).contiguous()

        # --- RoPE blocks ---
        x_rope_x_out = self._rope_apply_coeffs(x_rope_x, coeffs_x, inverse=inverse_rope)
        x_rope_y_out = self._rope_apply_coeffs(x_rope_y, coeffs_y, inverse=inverse_rope)

        return torch.cat([x_proj_out, x_rope_x_out, x_rope_y_out], dim=-1).contiguous()

    @staticmethod
    def _lift_K(Ks):
        out = torch.zeros(Ks.shape[:-2] + (4, 4), device=Ks.device, dtype=Ks.dtype)
        out[..., :3, :3] = Ks
        out[..., 3, 3] = 1.0
        return out

    @staticmethod
    def _invert_K(Ks):
        out = torch.zeros_like(Ks)
        out[..., 0, 0] = 1.0 / Ks[..., 0, 0]
        out[..., 1, 1] = 1.0 / Ks[..., 1, 1]
        out[..., 0, 2] = -Ks[..., 0, 2] / Ks[..., 0, 0]
        out[..., 1, 2] = -Ks[..., 1, 2] / Ks[..., 1, 1]
        out[..., 2, 2] = 1.0
        return out

    @staticmethod
    def _invert_SE3(transforms):
        Rinv = transforms[..., :3, :3].transpose(-1, -2)
        out = torch.zeros_like(transforms)
        out[..., :3, :3] = Rinv
        out[..., :3, 3] = -torch.einsum("...ij,...j->...i", Rinv, transforms[..., :3, 3])
        out[..., 3, 3] = 1.0
        return out

    @staticmethod
    def _rope_precompute_coeffs(positions, freq_base, freq_scale, feat_dim):
        num_freqs = feat_dim // 2
        freqs = freq_scale * (freq_base ** (-torch.arange(num_freqs, device=positions.device) / num_freqs))
        angles = positions[:, None] * freqs[None, :]
        angles = angles.view(1, 1, positions.shape[0], num_freqs)
        return torch.cos(angles), torch.sin(angles)

    @staticmethod
    def _rope_apply_coeffs(feats, coeffs, inverse=False):
        cos, sin = coeffs
        if cos.shape[2] != feats.shape[2]:
            n_repeats = feats.shape[2] // cos.shape[2]
            cos = cos.repeat(1, 1, n_repeats, 1)
            sin = sin.repeat(1, 1, n_repeats, 1)
        x_in = feats[..., : feats.shape[-1] // 2]
        y_in = feats[..., feats.shape[-1] // 2 :]
        if not inverse:
            return torch.cat((cos * x_in + sin * y_in, -sin * x_in + cos * y_in), dim=-1)
        else:
            return torch.cat((cos * x_in - sin * y_in, sin * x_in + cos * y_in), dim=-1)

class CrossAttentionRayRopeVectorized(nn.Module):
    def __init__(
        self, 
        embed_dim=768, 
        kv_dim=4096, 
        num_heads=12,
        proj_drop=0., 
        tau=1, 
        cos_attn=False, 
        num_rays_per_patch=3,
        depth_type='predict_dsig', 
        freq_base=3.0,
        apply_vo=True,
        **kwargs,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kv_dim = kv_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # --- Matrices ---
        self.mat_q = nn.Linear(embed_dim, embed_dim, bias=True)
        self.mat_kv = nn.Linear(kv_dim, embed_dim * 2, bias=False)
        self.v_bias = nn.Parameter(torch.zeros(embed_dim))
        self.register_buffer('zero_k_bias', torch.zeros(embed_dim))
        
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # --- RayRoPE Config ---
        self.depth_type = depth_type
        self.num_rays_per_patch = num_rays_per_patch
        self.freq_base = freq_base
        self.apply_vo = apply_vo
        
        self.rope_coord_dim = 3 + (self.num_rays_per_patch * 3)
        self.rope_mat_dim = 2 * self.rope_coord_dim
        self.num_rope_freqs = self.head_dim // self.rope_mat_dim

        if self.num_rays_per_patch == 3:
            self.offsets = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        elif self.num_rays_per_patch == 2:
            self.offsets = [[0.0, 0.0], [1.0, 1.0]]
        else:
            self.offsets = [[0.5, 0.5]]

        self.cos_attn = cos_attn
        if self.cos_attn:
            self.scale_mul = nn.Parameter(torch.tensor(0.0), requires_grad=True)
            self.max_scale_mul = math.log(100.0)

        if 'predict_dsig' in depth_type:
            self.d_proj_weight = nn.Parameter(torch.zeros((2, embed_dim)))
            self.d_proj_bias = nn.Parameter(torch.tensor([0.0, 3.0]))
        else:
            self.d_proj_weight = None

    # -----------------------------------------------------------------------
    # GRID GENERATION
    # -----------------------------------------------------------------------
    def _precompute_target_grid(self, scale_schedule, num_cameras, device):
        all_coords_list = []
        cam_id_list = []
        
        for item in scale_schedule:
            px, py = item[-1], item[-2]
            num_patches = px * py
            
            y_base = torch.arange(py, device=device, dtype=torch.float32)
            x_base = torch.arange(px, device=device, dtype=torch.float32)
            grid_y, grid_x = torch.meshgrid(y_base, x_base, indexing='ij')
            grid_x = grid_x.flatten()
            grid_y = grid_y.flatten()

            scale_coords = []
            for (ox, oy) in self.offsets:
                u = ((grid_x.unsqueeze(-1) + ox) / px) - 0.5
                v = ((grid_y.unsqueeze(-1) + oy) / py) - 0.5
                scale_coords.append(torch.stack([u, v], dim=-1))
                # print(f"x_base: {x_base} | y_base: {y_base}")
                # print(f"[_precompute_target_grid] scale_coords[-1]: {scale_coords[-1]}")
            
            scale_coords = torch.cat(scale_coords, dim=1) 
            scale_coords_expanded = scale_coords.repeat(num_cameras, 1, 1)
            all_coords_list.append(scale_coords_expanded)
            
            c_ids = torch.arange(num_cameras, device=device).repeat_interleave(num_patches)
            cam_id_list.append(c_ids)

        all_coords_norm = torch.cat(all_coords_list, dim=0)
        cam_ids = torch.cat(cam_id_list, dim=0)
        return all_coords_norm, cam_ids

    def _precompute_source_grid(self, px, py, num_cameras_src, device):
        num_patches = px * py
        y_base = torch.arange(py, device=device, dtype=torch.float32)
        x_base = torch.arange(px, device=device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(y_base, x_base, indexing='ij')
        grid_x = grid_x.flatten()
        grid_y = grid_y.flatten()

        scale_coords = []
        for (ox, oy) in self.offsets:
            u = ((grid_x.unsqueeze(-1) + ox) / px) - 0.5
            v = ((grid_y.unsqueeze(-1) + oy) / py) - 0.5
            scale_coords.append(torch.stack([u, v], dim=-1))
        
        scale_coords = torch.cat(scale_coords, dim=1) 
        all_coords_norm = scale_coords.repeat(num_cameras_src, 1, 1)
        cam_ids = torch.arange(num_cameras_src, device=device).repeat_interleave(num_patches)
        
        return all_coords_norm, cam_ids

    # -----------------------------------------------------------------------
    # WORLD COORDS & ROPE HELPERS
    # -----------------------------------------------------------------------
    def _compute_world_coords_vectorized(self, P_inv_seq, all_coords_norm, predicted_d):
        if predicted_d.ndim == 4:
             logd = predicted_d[..., 0:1]
             sigma = predicted_d[..., 1:2]
        else: 
             logd = predicted_d[..., 0:1].unsqueeze(-2)
             sigma = predicted_d[..., 1:2].unsqueeze(-2)

        B, L = logd.shape[0], logd.shape[1]
        R = all_coords_norm.shape[1]
        
        d1 = torch.exp(torch.clamp(logd - sigma, max=3.0))
        d2 = torch.exp(torch.clamp(logd + sigma, max=3.0))
        depths = torch.stack([d1, d2], dim=0).expand(-1, -1, -1, R, -1)
        disparity = 1.0 / torch.clamp(depths, min=1e-4, max=100.0)

        coords_input = all_coords_norm.unsqueeze(0).unsqueeze(0).expand(2, B, -1, -1, -1)
        ones = torch.ones_like(disparity)
        coords_4d = torch.cat([coords_input, ones, disparity], dim=-1)
        
        P_inv_broad = P_inv_seq.unsqueeze(0).unsqueeze(3)
        pd_world = torch.einsum("...ij,...j->...i", P_inv_broad, coords_4d)
        
        return pd_world

    def _get_rope_fns_batched(self, P, w2c, p0, pd, seq_len):
        p0_3d = _transform_to_query_frame(p0, P, w2c, transform_type='3d', denc_type='d').squeeze(-2)
        pd_dir, pd_d = _transform_to_query_frame(pd, P, w2c, transform_type='pj', denc_type='d')
        
        pd_dir = pd_dir.flatten(0, 1)[..., :2].flatten(start_dim=-2, end_dim=-1)
        pd_d = pd_d.flatten(0, 1).flatten(start_dim=-2, end_dim=-1)
        
        positions = {'p0': p0_3d, 'pd_dir': pd_dir, 'pd_depth': pd_d}
        cos, sin = _prepare_rope_coeff_uniformd(
            positions, self.num_rope_freqs, self.freq_base, 
            batch=P.shape[0], num_cameras=1, num_patches=seq_len
        )
        
        target = self.head_dim // 2
        if cos.shape[-1] < target:
            pad = target - cos.shape[-1]
            cos = F.pad(cos, (0, pad), value=1.0)
            sin = F.pad(sin, (0, pad), value=0.0)
            
        return cos, sin

    # -----------------------------------------------------------------------
    # MAIN FORWARD
    # -----------------------------------------------------------------------
    def forward(self, 
                q,
                ca_kv,
                poses, # Target Poses (C2W)
                poses_src, # Source Poses (C2W)
                intrs, # Target Intrinsics
                intrs_src, # Source Intrinsics
                scale_schedule=None,
                scale_ind=None,
                **kwargs):
        
        print(f"q.shape: {q.shape}")
        print(f"ca_kv.shape: {ca_kv.shape}")

        B, L_tgt, C = q.shape
        L_src = ca_kv.shape[1]
        device = q.device
        
        num_cams_tgt = poses.shape[1]
        num_cams_src = poses_src.shape[1]
        
        

        # 1. Projections
        q_feat = self.mat_q(q).view(B, L_tgt, self.num_heads, self.head_dim).permute(0, 2, 1, 3) 
        raw_d_q = F.linear(q, self.d_proj_weight, self.d_proj_bias).view(B, L_tgt, 1, 2)
        
        kv_feat = F.linear(ca_kv, self.mat_kv.weight, torch.cat((self.zero_k_bias, self.v_bias)))
        kv_feat = kv_feat.view(B, L_src, 2, self.num_heads, self.head_dim)
        k_feat = kv_feat[:, :, 0].permute(0, 2, 1, 3)
        v_feat = kv_feat[:, :, 1].permute(0, 2, 1, 3)
        
        raw_d_kv = F.linear(ca_kv, self.d_proj_weight, self.d_proj_bias).view(B, L_src, 1, 2)

        if self.cos_attn:
            scale_mul = self.scale_mul.clamp_max(self.max_scale_mul).exp()
            q_feat = F.normalize(q_feat, dim=-1, eps=1e-12).mul(scale_mul)
            k_feat = F.normalize(k_feat, dim=-1, eps=1e-12)

        # ---------------------------
        # 2. Matrices & Grids (EXACT REPLICATION OF ORIGINAL LOGIC)
        # ---------------------------
        # Instead of using 'poses' (C2W) directly, we perform the exact inversion chain
        # to match numerical errors in the reference implementation.
        
        # TARGET CAMS
        # Original: w2cs = torch.inverse(poses) -> c2ws = _invert_SE3(w2cs)
        w2cs_t = torch.inverse(poses) 
        c2ws_t = torch.inverse(w2cs_t) # We use this re-derived C2W for p0 to match
        
        # c2ws_t = poses
        # w2cs_t = torch.inverse(c2ws_t)


        Ks_t = intrs.clone(); 
        Ks_t[...,:2,2] -= 0.5
        P_inv_all_t = torch.einsum("...ij,...jk->...ik", c2ws_t, _lift_K(_invert_K(Ks_t)))

        # SOURCE CAMS
        # c2ws_s = poses_src
        # w2cs_s = torch.inverse(c2ws_s)
        w2cs_s = torch.inverse(poses_src)
        c2ws_s = torch.inverse(w2cs_s) # Re-derived C2W
        
        Ks_s = intrs_src.clone(); 
        Ks_s[...,:2,2] -= 0.5
        P_inv_all_s = torch.einsum("...ij,...jk->...ik", c2ws_s, _lift_K(_invert_K(Ks_s)))

        px_src, py_src = scale_schedule[-1][-1], scale_schedule[-1][-2]
        coords_kv, cam_ids_kv = self._precompute_source_grid(px_src, py_src, num_cams_src, device)

        

        if scale_ind is not None:
            scale_schedule = [scale_schedule[scale_ind]]

        # Grids
        coords_q, cam_ids_q = self._precompute_target_grid(scale_schedule, num_cams_tgt, device)



        # ---------------------------
        # 3. World Coordinates
        # ---------------------------
        P_inv_seq_q = P_inv_all_t[:, cam_ids_q]
        pd_world_q = self._compute_world_coords_vectorized(P_inv_seq_q, coords_q, raw_d_q)
        p0_world_q = c2ws_t[:, cam_ids_q][..., :, 3] # Using the re-derived C2W

        P_inv_seq_kv = P_inv_all_s[:, cam_ids_kv]
        pd_world_kv = self._compute_world_coords_vectorized(P_inv_seq_kv, coords_kv, raw_d_kv)
        p0_world_kv = c2ws_s[:, cam_ids_kv][..., :, 3] # Using the re-derived C2W

        # ---------------------------
        # 4. View Expansion
        # ---------------------------
        BN = B * num_cams_tgt
        
        w2cs_flat = w2cs_t.view(BN, 4, 4)
        Ps_flat = torch.einsum("...ij,...jk->...ik", _lift_K(Ks_t), w2cs_t).view(BN, 4, 4)

        # Q Geometry
        p0_q_exp = p0_world_q.unsqueeze(1).repeat(1, num_cams_tgt, 1, 1)
        pd_q_exp = pd_world_q.unsqueeze(2).repeat(1, 1, num_cams_tgt, 1, 1, 1)
        
        p0_in_q = p0_q_exp.view(BN, L_tgt, 1, 4).unsqueeze(1)
        pd_in_q = pd_q_exp.view(2, BN, L_tgt, self.num_rays_per_patch, 4).unsqueeze(2)
        
        # KV Geometry
        p0_kv_exp = p0_world_kv.unsqueeze(1).repeat(1, num_cams_tgt, 1, 1)
        pd_kv_exp = pd_world_kv.unsqueeze(2).repeat(1, 1, num_cams_tgt, 1, 1, 1)

        p0_in_kv = p0_kv_exp.view(BN, L_src, 1, 4).unsqueeze(1)
        pd_in_kv = pd_kv_exp.view(2, BN, L_src, self.num_rays_per_patch, 4).unsqueeze(2)

        # ---------------------------
        # 5. RoPE Calculation
        # ---------------------------
        cos_q, sin_q = self._get_rope_fns_batched(Ps_flat, w2cs_flat, p0_in_q, pd_in_q, L_tgt)
        cos_k, sin_k = self._get_rope_fns_batched(Ps_flat, w2cs_flat, p0_in_kv, pd_in_kv, L_src)

        # ---------------------------
        # 6. Apply RoPE & Attend
        # ---------------------------
        q_exp = q_feat.repeat_interleave(num_cams_tgt, dim=0)
        k_exp = k_feat.repeat_interleave(num_cams_tgt, dim=0)
        v_exp = v_feat.repeat_interleave(num_cams_tgt, dim=0)

        q_rot = self._apply_rope_coeffs(q_exp, cos_q, sin_q, inverse=True)
        k_rot = self._apply_rope_coeffs(k_exp, cos_k, sin_k, inverse=True)
        v_rot = self._apply_rope_coeffs(v_exp, cos_k, sin_k, inverse=True) if self.apply_vo else v_exp
            
        out = F.scaled_dot_product_attention(
            q_rot, k_rot, v_rot, 
            dropout_p=self.proj_drop.p if self.training else 0.0
        )
        
        if self.apply_vo:
            out = self._apply_rope_coeffs(out, cos_q, sin_q, inverse=False)

        # ---------------------------
        # 7. Gather
        # ---------------------------
        out = out.view(B, num_cams_tgt, self.num_heads, L_tgt, self.head_dim)
        gather_ids = cam_ids_q.view(1, 1, 1, L_tgt, 1).expand(B, 1, self.num_heads, L_tgt, self.head_dim)
        final_out = torch.gather(out, 1, gather_ids).squeeze(1)
        
        final_out = final_out.transpose(1, 2).reshape(B, L_tgt, C)
        return self.proj_drop(self.proj(final_out))

    @staticmethod
    def _apply_rope_coeffs(feats, cos, sin, inverse=False):
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        x1, x2 = torch.chunk(feats, 2, dim=-1)
        if inverse:
            return torch.cat([x1 * cos + x2 * sin, -x1 * sin + x2 * cos], dim=-1)
        else:
            return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class SelfAttnBlock(nn.Module):
    def __init__(
        self, embed_dim, kv_dim, cross_attn_layer_scale, cond_dim, act: bool, shared_aln: bool, norm_layer: partial,
        num_heads, mlp_ratio=4., drop=0., drop_path=0., tau=1, cos_attn=False,
        swiglu=False, customized_flash_attn=False, fused_mlp=False, fused_norm_func=None, checkpointing_sa_only=False,
        batch_size=1,
        rope2d_normalized_by_hw=False,
        N_views_src=1,
        N_views_tgt=1,
        use_prope=False
    ):
        super(SelfAttnBlock, self).__init__()
        self.C, self.D = embed_dim, cond_dim
        self.drop_path_rate = drop_path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if use_prope:
            self.attn = SelfAttentionPropePrefixed(
                embed_dim=embed_dim, 
                num_heads=num_heads, 
                proj_drop=drop, 
                tau=tau, 
                cos_attn=cos_attn, 
                customized_flash_attn=customized_flash_attn,
                batch_size=batch_size, 
                rope2d_normalized_by_hw=rope2d_normalized_by_hw,
                N_views=N_views_tgt
            )
        else:
            self.attn = SelfAttentionPrefixed(
                embed_dim=embed_dim, 
                num_heads=num_heads, 
                proj_drop=drop, 
                tau=tau, 
                cos_attn=cos_attn, 
                customized_flash_attn=customized_flash_attn,
                batch_size=batch_size, 
                rope2d_normalized_by_hw=rope2d_normalized_by_hw,
                N_views=N_views_tgt
            )

        self.using_swiglu = swiglu
        self.ffn = (FFNSwiGLU if swiglu else FFN)(in_features=embed_dim, hidden_features=round(embed_dim * mlp_ratio / 256) * 256, drop=drop, fused_mlp=fused_mlp)
        
        self.ln_wo_grad = norm_layer(embed_dim, elementwise_affine=False)
        self.fused_norm_func = fused_norm_func
        self.norm_eps = norm_layer.keywords.get('eps', 1e-6)
        
        self.shared_aln = shared_aln
        if self.shared_aln:
            self.ada_gss = nn.Parameter(torch.randn(1, 1, 6, embed_dim) / embed_dim**0.5)
        else:
            lin = nn.Linear(cond_dim, 6*embed_dim)
            self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), lin) if act else nn.Sequential(lin)

        self.fused_ada_norm=None

        self.use_prope = use_prope
        
    # NOTE: attn_bias_or_two_vector is None during inference
    def forward(self, 
                x, 
                cond_BD, 
                attn_bias_or_two_vector,
                attn_fn=None, 
                scale_schedule=None,
                rope2d_freqs_grid=None,
                scale_ind=None,
                poses=None,
                intrs=None,
                poses_src=None,
                intrs_src=None,
                input_size=None):  # todo: minGPT and vqgan also uses pre-norm, just like this, while MaskGiT uses post-norm
        
        with torch.cuda.amp.autocast(enabled=False):
            if self.shared_aln: # always True;                   (1, 1, 6, C)  + (B, 1, 6, C)
                gamma1, gamma2, scale1, scale2, shift1, shift2 = (self.ada_gss + cond_BD).unbind(2) # 116C + B16C =unbind(2)=> 6 B1C
            else:
                gamma1, gamma2, scale1, scale2, shift1, shift2 = self.ada_lin(cond_BD).view(-1, 1, 6, self.C).unbind(2)
        
        if self.use_prope:
            assert poses is not None
            assert intrs is not None

            if self.fused_ada_norm is None:
                x_sa = self.ln_wo_grad(x.float()).mul(scale1.add(1)).add_(shift1)
                x_sa = self.attn(x_sa,
                                attn_bias_or_two_vector=attn_bias_or_two_vector,
                                attn_fn=attn_fn,
                                scale_schedule=scale_schedule,
                                scale_ind=scale_ind,
                                poses=poses,
                                intrs=intrs,
                                poses_src=poses_src,
                                intrs_src=intrs_src,
                                input_size=input_size)
                
                x = x + self.drop_path(x_sa.mul_(gamma1))
                x = x + self.drop_path(self.ffn( 
                    self.ln_wo_grad(x.float()).mul(scale2.add(1)).add_(shift2)
                ).mul(gamma2))

                # x = x + self.drop_path(self.attn( 
                #     self.ln_wo_grad(x.float()).mul(scale1.add(1)).add_(shift1), 
                #     attn_bias_or_two_vector=attn_bias_or_two_vector,
                #     scale_schedule=scale_schedule,
                #     rope2d_freqs_grid=rope2d_freqs_grid).mul_(gamma1))
                # x = x + self.drop_path(self.ffn( 
                #     self.ln_wo_grad(x.float()).mul(scale2.add(1)).add_(shift2) ).mul(gamma2)) # this mul(gamma2) cannot be in-placed cuz we possibly use FusedMLP
            else:
                x = x + self.drop_path(self.attn(self.fused_ada_norm(C=self.C, eps=self.norm_eps, x=x, scale=scale1, shift=shift1), attn_bias_or_two_vector=attn_bias_or_two_vector).mul_(gamma1))
                x = x + self.drop_path(self.ffn(self.fused_ada_norm(C=self.C, eps=self.norm_eps, x=x, scale=scale2, shift=shift2)).mul(gamma2)) # this mul(gamma2) cannot be in-placed cuz we possibly use FusedMLP
        else:
            if self.fused_ada_norm is None:
                x_sa = self.ln_wo_grad(x.float()).mul(scale1.add(1)).add_(shift1)
                x_sa = self.attn(x_sa,
                                attn_bias_or_two_vector,
                                attn_fn,
                                scale_schedule)
                
                x = x + self.drop_path(x_sa.mul_(gamma1))
                x = x + self.drop_path(self.ffn( 
                    self.ln_wo_grad(x.float()).mul(scale2.add(1)).add_(shift2)
                ).mul(gamma2))

                # x = x + self.drop_path(self.attn( 
                #     self.ln_wo_grad(x.float()).mul(scale1.add(1)).add_(shift1), 
                #     attn_bias_or_two_vector=attn_bias_or_two_vector,
                #     scale_schedule=scale_schedule,
                #     rope2d_freqs_grid=rope2d_freqs_grid).mul_(gamma1))
                # x = x + self.drop_path(self.ffn( 
                #     self.ln_wo_grad(x.float()).mul(scale2.add(1)).add_(shift2) ).mul(gamma2)) # this mul(gamma2) cannot be in-placed cuz we possibly use FusedMLP
            else:
                x = x + self.drop_path(self.attn(self.fused_ada_norm(C=self.C, eps=self.norm_eps, x=x, scale=scale1, shift=shift1), attn_bias_or_two_vector=attn_bias_or_two_vector).mul_(gamma1))
                x = x + self.drop_path(self.ffn(self.fused_ada_norm(C=self.C, eps=self.norm_eps, x=x, scale=scale2, shift=shift2)).mul(gamma2)) # this mul(gamma2) cannot be in-placed cuz we possibly use FusedMLP
        return x

    def extra_repr(self) -> str:
        return f'shared_aln={self.shared_aln}, fused_norm={self.fused_norm_func is not None}'


class CrossAttnBlock(nn.Module):
    def __init__(
        self,
        embed_dim, kv_dim, cross_attn_layer_scale, cond_dim, act: bool, shared_aln: bool, norm_layer: partial,
        num_heads, mlp_ratio=4., drop=0., drop_path=0., tau=1, cos_attn=False,
        swiglu=False, customized_flash_attn=False, fused_mlp=False, fused_norm_func=None, checkpointing_sa_only=False,
        use_flex_attn=False, batch_size=2, pad_to_multiplier=1, apply_rope2d=False, rope2d_normalized_by_hw=False,
        N_views_src=1, N_views_tgt=1, use_prope=False, use_rayrope=False, multiscale_context=False,
        switti_attn_backend="sdpa",
    ):
        super(CrossAttnBlock, self).__init__()
        self.C, self.D = embed_dim, cond_dim
        self.drop_path_rate = drop_path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if use_prope:
            # self.sa = SelfAttentionPrope(
            #     embed_dim=embed_dim, num_heads=num_heads, proj_drop=drop, tau=tau, cos_attn=cos_attn, customized_flash_attn=customized_flash_attn,
            #     use_flex_attn=use_flex_attn, batch_size=batch_size, pad_to_multiplier=pad_to_multiplier, rope2d_normalized_by_hw=rope2d_normalized_by_hw,
            #     N_views=N_views_tgt
            # )

            self.sa = SelfAttentionPropeOptimized(
                embed_dim=embed_dim, num_heads=num_heads, proj_drop=drop, tau=tau, cos_attn=cos_attn, customized_flash_attn=customized_flash_attn,
                use_flex_attn=use_flex_attn, batch_size=batch_size, pad_to_multiplier=pad_to_multiplier, rope2d_normalized_by_hw=rope2d_normalized_by_hw,
                N_views=N_views_tgt, switti_attn_backend=switti_attn_backend,
            )

            # self.sa = SelfAttentionPropeOptimized2(
            #     embed_dim=embed_dim, num_heads=num_heads, proj_drop=drop, tau=tau, cos_attn=cos_attn, customized_flash_attn=customized_flash_attn,
            #     use_flex_attn=use_flex_attn, batch_size=batch_size, pad_to_multiplier=pad_to_multiplier, rope2d_normalized_by_hw=rope2d_normalized_by_hw,
            #     N_views=N_views_tgt
            # )
        elif use_rayrope:
            self.sa = SelfAttentionRayRopeUnified(
                embed_dim=embed_dim, num_heads=num_heads, proj_drop=drop, tau=tau, cos_attn=cos_attn, customized_flash_attn=customized_flash_attn,
                use_flex_attn=use_flex_attn, batch_size=batch_size, pad_to_multiplier=pad_to_multiplier, rope2d_normalized_by_hw=rope2d_normalized_by_hw,
                N_views=N_views_tgt
            )

        else:
            self.sa = SelfAttention(
                embed_dim=embed_dim, num_heads=num_heads, proj_drop=drop, tau=tau, cos_attn=cos_attn, customized_flash_attn=customized_flash_attn,
                use_flex_attn=use_flex_attn, batch_size=batch_size, pad_to_multiplier=pad_to_multiplier, rope2d_normalized_by_hw=rope2d_normalized_by_hw,
                N_views=N_views_tgt
            )



        if use_prope:

            if multiscale_context:
                self.ca = CrossAttentionPropeMultiscale(
                    embed_dim=embed_dim, 
                    kv_dim=kv_dim, 
                    num_heads=num_heads, 
                    proj_drop=drop, 
                    cos_attn=cos_attn,
                    N_views_src=N_views_src
                )
            else:
                self.ca = CrossAttentionPropeOptimized(
                    embed_dim=embed_dim, 
                    kv_dim=kv_dim, 
                    num_heads=num_heads, 
                    proj_drop=drop, 
                    cos_attn=cos_attn,
                    N_views_src=N_views_src
                )

                # self.ca = CrossAttentionPrope(
                #     embed_dim=embed_dim, 
                #     kv_dim=kv_dim, 
                #     num_heads=num_heads, 
                #     proj_drop=drop, 
                #     cos_attn=cos_attn,
                #     N_views_src=N_views_src
                # )
        elif use_rayrope:

            self.ca = CrossAttentionRayRopeVectorized(
                    embed_dim=embed_dim, 
                    kv_dim=kv_dim, 
                    num_heads=num_heads, 
                    proj_drop=drop, 
                    cos_attn=cos_attn,
                    N_views_src=N_views_src
                )

        else:
            self.ca = CrossAttention(embed_dim=embed_dim, kv_dim=kv_dim, num_heads=num_heads, proj_drop=drop, cos_attn=cos_attn)
        
        self.using_swiglu = swiglu
        self.ffn = (FFNSwiGLU if swiglu else FFN)(in_features=embed_dim, hidden_features=round(embed_dim * mlp_ratio / 256) * 256, drop=drop, fused_mlp=fused_mlp)
        
        self.ln_wo_grad = norm_layer(embed_dim, elementwise_affine=False)
        self.fused_norm_func = fused_norm_func
        self.norm_eps = norm_layer.keywords.get('eps', 1e-6)
        self.ca_norm = norm_layer(embed_dim, elementwise_affine=True)
        
        self.shared_aln = shared_aln
        if self.shared_aln: # always True
            self.ada_gss = nn.Parameter(torch.randn(1, 1, 6, embed_dim) / embed_dim**0.5)
        else:
            lin = nn.Linear(cond_dim, 6*embed_dim)
            self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), lin) if act else nn.Sequential(lin)
        
        if cross_attn_layer_scale >= 0:
            self.ca_gamma = nn.Parameter(cross_attn_layer_scale * torch.ones(embed_dim), requires_grad=True)
        else:
            self.ca_gamma = 1
        
        self.checkpointing_sa_only = checkpointing_sa_only
        self.use_prope = use_prope
        self.use_rayrope = use_rayrope
    
    # NOTE: attn_bias_or_two_vector is None during inference
    def forward(self, 
                x, 
                cond_BD, 
                ca_kv, #kv_compact, cu_seqlens_k, max_seqlen_k, kv_compact: [B*N_tokens_reference_view, D]
                attn_bias_or_two_vector, 
                attn_fn=None, 
                scale_schedule=None, 
                rope2d_freqs_grid=None,
                scale_ind=None,
                poses=None,
                intrs=None,
                poses_src=None,
                intrs_src=None,
                input_size=None):    # todo: minGPT and vqgan also uses pre-norm, just like this, while MaskGiT uses post-norm
                
        with torch.amp.autocast("cuda",enabled=False):    # disable half precision
            if self.shared_aln: # always True;                   (1, 1, 6, C)  + (B, 1, 6, C)
                gamma1, gamma2, scale1, scale2, shift1, shift2 = (self.ada_gss + cond_BD).unbind(2) # 116C + B16C =unbind(2)=> 6 B1C
            else:
                gamma1, gamma2, scale1, scale2, shift1, shift2 = self.ada_lin(cond_BD).view(-1, 1, 6, self.C).unbind(2)
        
        if self.use_prope or self.use_rayrope:

            assert poses is not None
            assert intrs is not None

            if self.fused_norm_func is None:
                x_sa = self.ln_wo_grad(x.float()).mul(scale1.add(1)).add_(shift1)
                if self.checkpointing_sa_only and self.training:
                    x_sa = checkpoint(self.sa, 
                                      x_sa, 
                                      attn_bias_or_two_vector, 
                                      attn_fn, 
                                      scale_schedule, 
                                      rope2d_freqs_grid,
                                      scale_ind=scale_ind,
                                      use_reentrant=False,
                                      poses=poses,
                                      intrs=intrs,
                                      input_size=input_size)
                else:
                    x_sa = self.sa(x_sa, 
                                   attn_bias_or_two_vector, 
                                   attn_fn, 
                                   scale_schedule, 
                                   rope2d_freqs_grid, 
                                   scale_ind=scale_ind,
                                   poses=poses,
                                   intrs=intrs,
                                   input_size=input_size)
                x = x + self.drop_path(x_sa.mul_(gamma1))
                
                x_ca = self.ca(self.ca_norm(x), 
                                ca_kv, 
                                poses=poses, 
                                poses_src=poses_src,
                                intrs=intrs,
                                intrs_src=intrs_src,
                                scale_schedule=scale_schedule,
                                scale_ind=scale_ind,
                                rope2d_freqs_grid=rope2d_freqs_grid).float().mul_(self.ca_gamma)
                
                x = x + self.drop_path(x_ca)

                # x = x + self.ca(self.ca_norm(x), 
                #                 ca_kv, 
                #                 poses=poses, 
                #                 poses_src=poses_src,
                #                 intrs=intrs,
                #                 intrs_src=intrs_src,
                #                 scale_schedule=scale_schedule,
                #                 scale_ind=scale_ind,
                #                 rope2d_freqs_grid=rope2d_freqs_grid).float().mul_(self.ca_gamma)
                
                x = x + self.drop_path(self.ffn( self.ln_wo_grad(x.float()).mul(scale2.add(1)).add_(shift2) ).mul(gamma2)) # this mul(gamma2) cannot be in-placed cuz we possibly use FusedMLP
            else:
                x_sa = self.fused_norm_func(C=self.C, eps=self.norm_eps, x=x, scale=scale1, shift=shift1)
                if self.checkpointing_sa_only and self.training:
                    x_sa = checkpoint(self.sa, 
                                      x_sa, 
                                      attn_bias_or_two_vector, 
                                      attn_fn, 
                                      scale_schedule, 
                                      rope2d_freqs_grid, 
                                      scale_ind=scale_ind,
                                      use_reentrant=False,
                                      poses=poses,
                                      intrs=intrs,
                                      input_size=input_size)
                else:
                    x_sa = self.sa(x_sa, 
                                   attn_bias_or_two_vector, 
                                   attn_fn, 
                                   scale_schedule, 
                                   rope2d_freqs_grid, 
                                   scale_ind=scale_ind,
                                   poses=poses,
                                   intrs=intrs,
                                   input_size=input_size)
                x = x + self.drop_path(x_sa.mul_(gamma1))
                x_ca = self.ca(self.ca_norm(x),
                                ca_kv,
                                poses=poses,
                                poses_src=poses_src,
                                intrs=intrs,
                                intrs_src=intrs_src,
                                scale_schedule=scale_schedule,
                                scale_ind=scale_ind,
                                rope2d_freqs_grid=rope2d_freqs_grid).float().mul_(self.ca_gamma)
                x = x + self.drop_path(x_ca)
                x = x + self.drop_path(self.ffn(self.fused_norm_func(C=self.C, eps=self.norm_eps, x=x, scale=scale2, shift=shift2)).mul(gamma2)) # this mul(gamma2) cannot be in-placed cuz we possibly use FusedMLP
        
        else:

            if self.fused_norm_func is None:
                x_sa = self.ln_wo_grad(x.float()).mul(scale1.add(1)).add_(shift1)
                if self.checkpointing_sa_only and self.training:
                    x_sa = checkpoint(self.sa, x_sa, attn_bias_or_two_vector, attn_fn, scale_schedule, rope2d_freqs_grid, use_reentrant=False)
                else:
                    x_sa = self.sa(x_sa, attn_bias_or_two_vector, attn_fn, scale_schedule, rope2d_freqs_grid, scale_ind=scale_ind)
                x = x + self.drop_path(x_sa.mul_(gamma1))
                x = x + self.ca(self.ca_norm(x), ca_kv).float().mul_(self.ca_gamma)
                x = x + self.drop_path(self.ffn( self.ln_wo_grad(x.float()).mul(scale2.add(1)).add_(shift2) ).mul(gamma2)) # this mul(gamma2) cannot be in-placed cuz we possibly use FusedMLP
            else:
                x_sa = self.fused_norm_func(C=self.C, eps=self.norm_eps, x=x, scale=scale1, shift=shift1)
                if self.checkpointing_sa_only and self.training:
                    x_sa = checkpoint(self.sa, x_sa, attn_bias_or_two_vector, attn_fn, scale_schedule, rope2d_freqs_grid, use_reentrant=False)
                else:
                    x_sa = self.sa(x_sa, attn_bias_or_two_vector, attn_fn, scale_schedule, rope2d_freqs_grid, scale_ind=scale_ind)
                x = x + self.drop_path(x_sa.mul_(gamma1))
                x = x + self.ca(self.ca_norm(x), ca_kv).float().mul_(self.ca_gamma)
                x = x + self.drop_path(self.ffn(self.fused_norm_func(C=self.C, eps=self.norm_eps, x=x, scale=scale2, shift=shift2)).mul(gamma2)) # this mul(gamma2) cannot be in-placed cuz we possibly use FusedMLP
        return x
    
    def extra_repr(self) -> str:
        return f'shared_aln={self.shared_aln}, fused_norm={self.fused_norm_func is not None}, ca_gamma={"<learnable>" if isinstance(self.ca_gamma, nn.Parameter) else self.ca_gamma}'


class AdaLNBeforeHead(nn.Module):
    def __init__(self, C, D, act: bool, norm_layer: partial, fused_norm_func=None):   # C: embed_dim, D: cond_dim
        super().__init__()
        self.C, self.D = C, D
        self.ln_wo_grad = norm_layer(C, elementwise_affine=False)
        self.fused_norm_func = fused_norm_func
        self.norm_eps = norm_layer.keywords.get('eps', 1e-6)
        lin = nn.Linear(D, 2*C)
        self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), lin) if act else nn.Sequential(lin)
    
    def forward(self, x_BLC: torch.Tensor, cond_BD: Optional[torch.Tensor]):
        scale, shift = self.ada_lin(cond_BD).view(-1, 1, 2, self.C).unbind(2)
        if self.fused_norm_func is None:
            return self.ln_wo_grad(x_BLC).mul(scale.add(1)).add_(shift)
        else:
            return self.fused_norm_func(C=self.C, eps=self.norm_eps, x=x_BLC, scale=scale, shift=shift)


def eval_cross_attention(dev, scale_schedule):

    # --- Hyperparams ---
    B = 1
    H = 16
    D = 2048  # Embed dim
    D_kv = 2048 # KV dim
    N_views_src = 2
    N_views_tgt = 3

    # Calculate lengths
    # Target length = Sum(N_views_tgt * px * py) for all scales used
    L_tgt_total = 0
    for _, px, py in scale_schedule:
        L_tgt_total += N_views_tgt * px * py
        
    # Source length = N_views_src * patch_src * patch_src
    patch_src_dim = 16
    L_src_total = N_views_src * patch_src_dim * patch_src_dim
    
    # --- Inputs ---
    q = torch.randn(B, L_tgt_total, D, device=dev, requires_grad=True)
    
    # KV is passed as (kv_compact, cu_seqlens, max_seqlen)
    kv_data = torch.randn(B, L_src_total, D_kv, device=dev, requires_grad=True)
    # Dummy cu_seqlens (assuming full batch is one block for simplicity here, or just unused)
    cu_seqlens_k = torch.tensor([0, L_src_total], device=dev, dtype=torch.int32)
    max_seqlen_k = L_src_total
    
    # Poses (B, N, 4, 4) - Random SE3-ish
    poses_tgt = torch.eye(4, device=dev).repeat(B, N_views_tgt, 1, 1)
    poses_tgt[..., :3, 3] = torch.randn(B, N_views_tgt, 3, device=dev) # Random translation
    
    poses_src = torch.eye(4, device=dev).repeat(B, N_views_src, 1, 1)
    poses_src[..., :3, 3] = torch.randn(B, N_views_src, 3, device=dev)
    
    # Intrinsics (B, N, 3, 3)
    intrs_tgt = torch.eye(3, device=dev).repeat(B, N_views_tgt, 1, 1)
    intrs_src = torch.eye(3, device=dev).repeat(B, N_views_src, 1, 1)
    
    intrs_src = None
    intrs_tgt = None

    # --- Models ---
    model_old = CrossAttentionPrope(embed_dim=D, kv_dim=D_kv, num_heads=H, N_views_src=N_views_src).to(dev)
    model_new = CrossAttentionPropeOptimized(embed_dim=D, kv_dim=D_kv, num_heads=H, N_views_src=N_views_src).to(dev)
    
    # Copy weights to ensure identical initialization
    model_new.load_state_dict(model_old.state_dict())
    
    # --- Forward Pass ---
    print(f"Running Forward... len(scale_schedule): {scale_schedule}")
    
    # Old
    out_old = model_old(
        q, (kv_data, cu_seqlens_k, max_seqlen_k), 
        poses_tgt, poses_src, intrs_tgt, intrs_src, 
        scale_schedule, scale_ind=None
    )
    
    # New
    out_new = model_new(
        q, (kv_data, cu_seqlens_k, max_seqlen_k), 
        poses_tgt, poses_src, intrs_tgt, intrs_src, 
        scale_schedule, scale_ind=None
    )
    
    # Verify Output
    # Floating point arithmetic order changes slightly (einsum vs loop), allow small tolerance
    diff = (out_old - out_new).abs().max().item()
    print(f"Forward Max Difference: {diff:.8f}")
    if diff < 1e-5:
        print("✅ Forward pass matches!")
    else:
        print("❌ Forward pass mismatch!")

    # --- Backward Pass ---
    print("\nRunning Backward...")
    
    loss_old = out_old.mean()
    loss_new = out_new.mean()
    
    # We need to retain grad on inputs to check them
    q.retain_grad()
    kv_data.retain_grad()
    
    # Zero grads
    model_old.zero_grad()
    loss_old.backward()
    grad_q_old = q.grad.clone()
    grad_kv_old = kv_data.grad.clone()
    grad_w_old = model_old.mat_q.weight.grad.clone()
    
    # Reset input grads
    q.grad = None
    kv_data.grad = None
    
    model_new.zero_grad()
    loss_new.backward()
    grad_q_new = q.grad.clone()
    grad_kv_new = kv_data.grad.clone()
    grad_w_new = model_new.mat_q.weight.grad.clone()
    
    # Verify Gradients
    diff_q = (grad_q_old - grad_q_new).abs().max().item()
    diff_kv = (grad_kv_old - grad_kv_new).abs().max().item()
    diff_w = (grad_w_old - grad_w_new).abs().max().item()
    
    print(f"Grad Q Diff: {diff_q:.8f}")
    print(f"Grad KV Diff: {diff_kv:.8f}")
    print(f"Grad Weight Diff: {diff_w:.8f}")
    
    if diff_q < 1e-5 and diff_kv < 1e-5:
        print("✅ Backward pass matches!")
    else:
        print("❌ Backward pass mismatch!")

    import time
    times1 = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for i in range(1000):
        
        torch.cuda.synchronize()
        start_event.record()
        out_old = model_old(
            q, (kv_data, cu_seqlens_k, max_seqlen_k), 
            poses_tgt, poses_src, intrs_tgt, intrs_src, 
            scale_schedule, scale_ind=None
            )
        end_event.record()
        torch.cuda.synchronize()

        cost_ms = start_event.elapsed_time(end_event)
        cost_s = cost_ms / 1000.0

        if i>50:
            times1.append(cost_s)


    times2 = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for i in range(1000):
        torch.cuda.synchronize()
        start_event.record()

        out_new = model_new(
            q, (kv_data, cu_seqlens_k, max_seqlen_k), 
            poses_tgt, poses_src, intrs_tgt, intrs_src, 
            scale_schedule, scale_ind=None
        )

        end_event.record()
        torch.cuda.synchronize()

        cost_ms = start_event.elapsed_time(end_event)
        cost_s = cost_ms / 1000.0

        if i>50:
            times2.append(cost_s)

    med_old = np.median(times1)
    med_new = np.median(times2)

    print(f"Old Model med Time: {med_old:.6f} s")
    print(f"New Model med Time: {med_new:.6f} s")
    
    print(f"Speedup: {med_old / med_new:.2f}x")



def eval_self_attention(dev, scale_schedule, benchmark=False):

    # --- Hyperparams ---
    B = 1
    H = 16
    D = 2048
    N_views = 3

    # Calculate Total Sequence Length
    # Logic: Sum(N_views * px * py) for all scales
    L_total = 0
    for _, px, py in scale_schedule:
        L_total += N_views * px * py
        
    print(f"Total Sequence Length (L): {L_total}")

    # --- Inputs ---
    # Input X (B, L, D)
    x = torch.randn(B, L_total, D, device=dev, requires_grad=True)


    l_end = np.sum([i*j*k for i,j,k in scale_schedule])
    print(f"l_end: {l_end}")

    d: torch.Tensor = torch.cat([torch.full((pn[0]*pn[1]*pn[2],), i) for i, pn in enumerate(scale_schedule)]).view(1, l_end, 1)
    l_end_nviews = l_end * N_views
    d = d.repeat_interleave(N_views).view(1, l_end_nviews, 1)
    
    dT = d.transpose(1, 2)    # dT: 11L
    attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, l_end_nviews, l_end_nviews)
    attn_bias = attn_bias_for_masking[:, :, :l_end_nviews, :l_end_nviews].contiguous()   # attn_bias: 11LL
    attn_bias_or_two_vector = attn_bias.type_as(x).to(x.device)
    
    # Poses (B, N, 4, 4) - Self-attn expects N_views for projection logic
    poses = torch.eye(4, device=dev).repeat(B, N_views, 1, 1)
    poses[..., :3, 3] = torch.randn(B, N_views, 3, device=dev)
    
    # Intrinsics (B, N, 3, 3)
    intrs = torch.eye(3, device=dev).repeat(B, N_views, 1, 1)
    intrs = None

    # --- Models ---
    # Initialize both models
    model_old = SelfAttentionPropeOptimized2(embed_dim=D, num_heads=H, customized_flash_attn=False).to(dev)
    model_new = SelfAttentionPropeOptimized(embed_dim=D, num_heads=H, customized_flash_attn=False).to(dev)
    
    # Ensure identical initialization
    model_new.load_state_dict(model_old.state_dict())
    
    # --- Forward Pass ---
    print("\nRunning Forward...")
    
    # Note: attn_bias_or_two_vector is a required positional arg, passing None
    out_old = model_old(x, attn_bias_or_two_vector, poses=poses, intrs=intrs, scale_schedule=scale_schedule)
    out_new = model_new(x, attn_bias_or_two_vector, poses=poses, intrs=intrs, scale_schedule=scale_schedule)

    diff = (out_old - out_new).abs().max().item()
    print(f"Forward Max Difference: {diff:.8f}")
    
    # Tolerance slightly higher for accumulation differences in RoPE
    if diff < 1e-4:
        print("✅ Forward pass matches!")
    else:
        print("❌ Forward pass mismatch!")

    # --- Backward Pass ---
    print("\nRunning Backward...")
    
    loss_old = out_old.mean()
    loss_new = out_new.mean()
    
    # Retain grad on input to verify backprop through the complex RoPE/Projection logic
    x.retain_grad()
    
    # Old Backward
    model_old.zero_grad()
    loss_old.backward()
    grad_x_old = x.grad.clone()
    grad_w_old = model_old.mat_qkv.weight.grad.clone()
    
    # Reset input grad
    x.grad = None
    
    # New Backward
    model_new.zero_grad()
    loss_new.backward()
    grad_x_new = x.grad.clone()
    grad_w_new = model_new.mat_qkv.weight.grad.clone()

    
    diff_x = (grad_x_old - grad_x_new).abs().max().item()
    diff_w = (grad_w_old - grad_w_new).abs().max().item()
    
    print(f"Grad Input X Diff: {diff_x:.8f}")
    print(f"Grad Weight Diff:  {diff_w:.8f}")
    
    if diff_x < 1e-4:
        print("✅ Backward pass matches!")
    else:
        print("❌ Backward pass mismatch!")

    if benchmark:

        # --- Timing ---
        print("\nRunning Timing (1000 iters)...")
        
        # Time Old
        times1 = []
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        for i in range(1000):
            
            torch.cuda.synchronize()
            start_event.record()
            _ = model_old(x, attn_bias_or_two_vector, 
                        poses=poses, intrs=intrs, 
                        scale_schedule=scale_schedule)

            end_event.record()
            torch.cuda.synchronize()

            cost_ms = start_event.elapsed_time(end_event)
            cost_s = cost_ms / 1000.0

            if i>50:
                times1.append(cost_s)

        # Time New
        times2 = []
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        for i in range(1000):
            
            torch.cuda.synchronize()
            start_event.record()
            _ = model_new(x, attn_bias_or_two_vector, 
                        poses=poses, intrs=intrs, 
                        scale_schedule=scale_schedule)

            end_event.record()
            torch.cuda.synchronize()

            cost_ms = start_event.elapsed_time(end_event)
            cost_s = cost_ms / 1000.0

            if i>50:
                times2.append(cost_s)

        med_old = np.median(times1)
        med_new = np.median(times2)

        print(f"Old Model med Time: {med_old:.6f} s")
        print(f"New Model med Time: {med_new:.6f} s")
        
        print(f"Speedup: {med_old / med_new:.2f}x")
    


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    rng = torch.Generator(device=dev)
    rng.manual_seed(42)
    
    
    # Scale schedule: [(idx, px, py), ...]
    scale_schedule = [
        (1, 1, 1),  # Scale 0
        (1, 2, 2),   # Scale 1
        (1, 4, 4),
        (1, 6, 6),
        (1, 8, 8),
        (1, 12, 12),
        (1, 16, 16)
    ]
    scale_ind=5


    # 2.32, 1.21
    # 2.29, 1.17
    # 2.35, 1.16
    # 2.48, 1.21
    # 2.22, 1.21

    # eval_cross_attention(dev, scale_schedule)
    # eval_cross_attention(dev, [scale_schedule[scale_ind]])
    # 1: 2.21, 1.06
    # 2: 2.10, 1.05
    # 3: 2.14, 1.11
    # 4: 2.22, 1.03
    # 5: 2.14, 1.11
    
    
    

    # 2.56, 1.23
    # 2.63, 1.27
    # 2.47, 1.03
    # 2.66, 1.03
    # 2.22, 1.04
    # 2.72, 1.05

    eval_self_attention(dev, scale_schedule, False)
    eval_self_attention(dev, [scale_schedule[scale_ind]], False)
    # 2.56, 1.03
    # 2.57, 1.13
    # 2.51 0.81
    # 2.65, 1.05
    # 2.68, 1.19
    # 2.67, 0.99
    exit()
    



if __name__ == '__main__':
    main()
