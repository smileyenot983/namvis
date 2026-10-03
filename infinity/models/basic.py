"""
Definitions of blocks of VAR transformer model.
"""

import math
from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import DropPath
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


MAX_D_F = 10.0
MAX_ASINH_D_F = math.asinh(MAX_D_F)


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
            out = F.scaled_dot_product_attention(
                query=q_attention,
                key=k_attention,
                value=v_attention,
                attn_mask=attn_mask,
                **kwargs
            )


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
            q = q.contiguous()      # bf16
            k = k.contiguous()      # bf16
            v = v.contiguous()      # bf16                                             # bf16
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
    cosine_list = []
    sine_list = []
    for pos_name, pos in positions.items():
        if pos_name in ['p0', 'pd_3d']:
            max_period = 1.0 * 4
        elif pos_name in ['pinf_dir', 'pd_dir', 'p0_dir']:
            max_period = 2.0 * 4
        elif pos_name in ['pd_disparity', 'p0_disparity']:
            max_period = 20.0 * 4
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


        rope_angle = rope_angle.to(torch.float32)

        if rope_angle.shape[0] == batch * 2:
            rope_angles1 = rope_angle[:batch]
            rope_angles2 = rope_angle[batch:]

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


        # pd_dir [2*B*N_views, 1, L, 6(R*xy)] z removed
        pd_dir = pd_dir.flatten(0, 1)[..., :2].flatten(start_dim=-2, end_dim=-1)
        # pd_d: [2*B*N_Views, 1, L, R]
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
            # B. Gather P_inv & Compute World Coords
            P_inv_seq = P_inv_all[:, cam_ids]

            # [2(d1,d2),B,N_views*N_tokens_total,R,4]
            pd_world = self._compute_world_coords_vectorized(P_inv_seq, all_coords_norm, predicted_d)
            # [B,N_views*N_tokens_total,4]
            p0_world = c2ws[:, cam_ids][..., :, 3]

            # C. View Expansion (Parallelize N views)
            p0_world_exp = p0_world.unsqueeze(1).repeat(1, num_cameras, 1, 1).unsqueeze(-2)
            pd_world_exp = pd_world.unsqueeze(2).repeat(1, 1, num_cameras, 1, 1, 1)

            BN = B * num_cameras
            w2cs_flat = w2cs.view(BN, 4, 4)
            Ps_flat = torch.einsum("...ij,...jk->...ik", _lift_K(Ks_norm), w2cs).view(BN, 4, 4)

            p0_in = p0_world_exp.view(BN, L, 1, 4).unsqueeze(1)
            pd_in = pd_world_exp.view(2, BN, L, self.num_rays_per_patch, 4).unsqueeze(2)


            # D. RoPE & Attention
            cos_full, sin_full = self._get_rope_fns_batched(Ps_flat, w2cs_flat, p0_in, pd_in, L)

            q_exp = q.repeat_interleave(num_cameras, dim=0)
            k_exp = k.repeat_interleave(num_cameras, dim=0)
            v_exp = v.repeat_interleave(num_cameras, dim=0)

            q_rot = self._apply_rope_coeffs(q_exp, cos_full, sin_full, inverse=True)
            k_rot = self._apply_rope_coeffs(k_exp, cos_full, sin_full, inverse=True)
            v_rot = self._apply_rope_coeffs(v_exp, cos_full, sin_full, inverse=True) if self.apply_vo else v_exp


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

            # A. Chunk Grid & World Coords
            # chunk_coords: [N_patches, N_rays, 2(u,v)]
            chunk_coords = self._compute_chunk_coords(px, py, device) # (P, R, 2)


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


            # P: K*w2cs
            # cos/sin_chunk: [B*N_views, R, head_dim]
            cos_chunk, sin_chunk = self._get_rope_fns_batched(Ps_flat,
                                                              w2cs_flat,
                                                              p0_in,
                                                              pd_in,
                                                              chunk_len)


            # Rotate Chunk for all views
            q_exp = q.repeat_interleave(num_cameras, dim=0)
            k_exp = k.repeat_interleave(num_cameras, dim=0)
            v_exp = v.repeat_interleave(num_cameras, dim=0)


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


            x_grid = ((x_grid+0.5) / px)
            y_grid = ((y_grid+0.5) / py)

            pos_x.append(x_grid)
            pos_y.append(y_grid)


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
                k = self.cached_k = torch.cat([self.cached_k, k], dim=2)
                v = self.cached_v = torch.cat([self.cached_v, v], dim=2)


        out = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=attn_mask,
            **kwargs
        )

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


        # kv_compact: [256, 2, 16, 128] ,[n_tokens, 2(k,v), n_heads, head_dim]
        kv_compact = F.linear(kv_compact,
                              weight=self.mat_kv.weight,
                              bias=torch.cat((self.zero_k_bias, self.v_bias))).view(N, 2, self.num_heads, self.head_dim) # NC => N2Hc


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


            # prepare matrices for src tokens(single scale)
            _, px_last, py_last = scale_schedule[-1]
            px_last,py_last = int(math.sqrt(patches_per_view_src)), int(math.sqrt(patches_per_view_src))

            num_repeats = px_last * py_last
            x_grid = torch.arange(px_last, device=device).repeat(py_last * N_views_src)
            y_grid = torch.arange(py_last, device=device).repeat_interleave(px_last).repeat(N_views_src)


            pos_x_src = x_grid
            pos_y_src = y_grid

            coeffs_x_src = self._rope_precompute_coeffs(pos_x_src, 100.0, 1.0, self.head_dim // 4)
            coeffs_y_src = self._rope_precompute_coeffs(pos_y_src, 100.0, 1.0, self.head_dim // 4)

            cached_src = (P_inv_src, coeffs_x_src, coeffs_y_src)
            if rope2d_freqs_grid is not None:
                rope2d_freqs_grid[src_cache_key] = cached_src

        P_inv_src, coeffs_x_src, coeffs_y_src = cached_src


        # [3] apply transform to source KV
        k = self._apply_transform(k, P_inv_src, coeffs_x_src, coeffs_y_src, inverse_rope=False)
        v = self._apply_transform(v, P_inv_src, coeffs_x_src, coeffs_y_src, inverse_rope=False)

        # [4] Q projection
        # Project Q: [B, L, C] -> [B, H, L, D]
        q = self.mat_q(q).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)


        cached_tgt = None
        if rope2d_freqs_grid is not None:
            cached_tgt = rope2d_freqs_grid.get(tgt_cache_key)


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


            # Build Multi-scale Target Sequences
            for _, px, py in current_schedule:
                num_repeats = px * py

                x_grid = torch.arange(px, device=device).repeat(py * N_views_tgt)
                y_grid = torch.arange(py, device=device).repeat_interleave(px).repeat(N_views_tgt)


                pos_x_tgt.append(x_grid)
                pos_y_tgt.append(y_grid)

                if not is_single_scale:
                    P_T_list.append(P_T_tgt.repeat_interleave(num_repeats, dim=1))
                    P_list.append(P_tgt.repeat_interleave(num_repeats, dim=1))


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


        q = self._apply_transform(q, P_T_expanded_tgt, coeffs_x_tgt, coeffs_y_tgt, inverse_rope=False)

        from torch.nn.attention import SDPBackend, sdpa_kernel
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
            )
        # --- 3. ATTENTION ---

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


        Ks_t = intrs.clone();
        Ks_t[...,:2,2] -= 0.5
        P_inv_all_t = torch.einsum("...ij,...jk->...ik", c2ws_t, _lift_K(_invert_K(Ks_t)))

        # SOURCE CAMS
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

            self.sa = SelfAttentionPropeOptimized(
                embed_dim=embed_dim, num_heads=num_heads, proj_drop=drop, tau=tau, cos_attn=cos_attn, customized_flash_attn=customized_flash_attn,
                use_flex_attn=use_flex_attn, batch_size=batch_size, pad_to_multiplier=pad_to_multiplier, rope2d_normalized_by_hw=rope2d_normalized_by_hw,
                N_views=N_views_tgt, switti_attn_backend=switti_attn_backend,
            )

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
