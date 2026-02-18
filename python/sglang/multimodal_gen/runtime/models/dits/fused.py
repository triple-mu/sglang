import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.multimodal_gen.runtime.layers.layernorm import FP32LayerNorm, RMSNorm


@triton.jit
def _qk_rms_norm_rope_inplace(
    q_ptr,  # [b, s, n, d]
    k_ptr,  # [b, s, n, d]
    qw_ptr,  # [n, d]
    kw_ptr,  # [n, d]
    cos_ptr,  # [s, d//2]
    sin_ptr,  # [s, d//2]
    qk_stride_b,
    qk_stride_s,
    qk_stride_n,
    qk_stride_d,
    qkw_stride_n,
    qkw_stride_d,
    emb_stride_s,
    emb_stride_d,
    num_heads,
    head_dim,
    half_head_dim,
    eps,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_HALF_D: tl.constexpr,
    INTERLEAVE: tl.constexpr,
    WITH_WEIGHT: tl.constexpr,
):
    BLOCK_SIZE_D: tl.constexpr = BLOCK_SIZE_HALF_D * 2
    s_id = tl.program_id(0)
    b_id = tl.program_id(1)

    offset = b_id * qk_stride_b + s_id * qk_stride_s
    q_ptr += offset
    k_ptr += offset

    offset = s_id * emb_stride_s
    cos_ptr += offset
    sin_ptr += offset

    n_offset = tl.arange(0, BLOCK_SIZE_N)[:, None]
    d_offset = tl.arange(0, BLOCK_SIZE_D)[None, :]
    half_d_offset = tl.arange(0, BLOCK_SIZE_HALF_D)[None, :]
    n_mask = n_offset < num_heads
    d_mask = d_offset < head_dim
    half_d_mask = half_d_offset < half_head_dim
    nd_mask = n_mask & d_mask
    nd_half_mask = n_mask & half_d_mask

    # [BLOCK_SIZE_HALF_D]
    cos = tl.load(
        cos_ptr + half_d_offset * emb_stride_d, mask=half_d_mask, other=0.0
    ).to(tl.float32)
    sin = tl.load(
        sin_ptr + half_d_offset * emb_stride_d, mask=half_d_mask, other=0.0
    ).to(tl.float32)

    if INTERLEAVE:
        q_block_ptr = q_ptr + n_offset * qk_stride_n + d_offset * qk_stride_d
        k_block_ptr = k_ptr + n_offset * qk_stride_n + d_offset * qk_stride_d

        # [BLOCK_SIZE_N, BLOCK_SIZE_D]
        q = tl.load(q_block_ptr, mask=nd_mask, other=0.0).to(tl.float32)
        k = tl.load(k_block_ptr, mask=nd_mask, other=0.0).to(tl.float32)

        if WITH_WEIGHT:
            q_w_block_ptr = qw_ptr + n_offset * qkw_stride_n + d_offset * qkw_stride_d
            k_w_block_ptr = kw_ptr + n_offset * qkw_stride_n + d_offset * qkw_stride_d

            # [BLOCK_SIZE_N, BLOCK_SIZE_D]
            q_w = tl.load(q_w_block_ptr, mask=nd_mask).to(tl.float32)
            k_w = tl.load(k_w_block_ptr, mask=nd_mask).to(tl.float32)
        else:
            q_w = 1.0
            k_w = 1.0

        q_mean_sq = tl.sum(q * q, keep_dims=True) / (num_heads * head_dim)
        k_mean_sq = tl.sum(k * k, keep_dims=True) / (num_heads * head_dim)
        q_rstd = tl.math.rsqrt(q_mean_sq + eps)
        k_rstd = tl.math.rsqrt(k_mean_sq + eps)

        # [BLOCK_SIZE_N, BLOCK_SIZE_D]
        q = q * q_rstd * q_w
        k = k * k_rstd * k_w

        q1, q2 = tl.split(tl.reshape(q, [BLOCK_SIZE_N, BLOCK_SIZE_HALF_D, 2]))
        k1, k2 = tl.split(tl.reshape(k, [BLOCK_SIZE_N, BLOCK_SIZE_HALF_D, 2]))

        # [BLOCK_SIZE_N, BLOCK_SIZE_HALF_D]
        qo1 = tl.fma(-q2, sin, q1 * cos)
        qo2 = tl.fma(q1, sin, q2 * cos)
        ko1 = tl.fma(-k2, sin, k1 * cos)
        ko2 = tl.fma(k1, sin, k2 * cos)

        # [BLOCK_SIZE_N, BLOCK_SIZE_D]
        q = tl.interleave(qo1, qo2)
        k = tl.interleave(ko1, ko2)

        tl.store(q_block_ptr, q, mask=nd_mask)
        tl.store(k_block_ptr, k, mask=nd_mask)
    else:
        q1_block_ptr = q_ptr + n_offset * qk_stride_n + half_d_offset * qk_stride_d
        q2_block_ptr = (
            q_ptr
            + n_offset * qk_stride_n
            + (half_d_offset + half_head_dim) * qk_stride_d
        )
        k1_block_ptr = k_ptr + n_offset * qk_stride_n + half_d_offset * qk_stride_d
        k2_block_ptr = (
            k_ptr
            + n_offset * qk_stride_n
            + (half_d_offset + half_head_dim) * qk_stride_d
        )

        # [BLOCK_SIZE_N, BLOCK_SIZE_HALF_D]
        q1 = tl.load(q1_block_ptr, mask=nd_half_mask, other=0.0).to(tl.float32)
        q2 = tl.load(q2_block_ptr, mask=nd_half_mask, other=0.0).to(tl.float32)
        k1 = tl.load(k1_block_ptr, mask=nd_half_mask, other=0.0).to(tl.float32)
        k2 = tl.load(k2_block_ptr, mask=nd_half_mask, other=0.0).to(tl.float32)

        if WITH_WEIGHT:
            q_w1_block_ptr = (
                qw_ptr + n_offset * qkw_stride_n + half_d_offset * qkw_stride_d
            )
            q_w2_block_ptr = (
                qw_ptr
                + n_offset * qkw_stride_n
                + (half_d_offset + half_head_dim) * qkw_stride_d
            )
            k_w1_block_ptr = (
                kw_ptr + n_offset * qkw_stride_n + half_d_offset * qkw_stride_d
            )
            k_w2_block_ptr = (
                kw_ptr
                + n_offset * qkw_stride_n
                + (half_d_offset + half_head_dim) * qkw_stride_d
            )
            # [BLOCK_SIZE_N, BLOCK_SIZE_HALF_D]
            q_w1 = tl.load(q_w1_block_ptr, mask=nd_half_mask).to(tl.float32)
            q_w2 = tl.load(q_w2_block_ptr, mask=nd_half_mask).to(tl.float32)
            k_w1 = tl.load(k_w1_block_ptr, mask=nd_half_mask).to(tl.float32)
            k_w2 = tl.load(k_w2_block_ptr, mask=nd_half_mask).to(tl.float32)
        else:
            q_w1 = 1.0
            q_w2 = 1.0
            k_w1 = 1.0
            k_w2 = 1.0

        q_mean_sq = tl.sum(q1 * q1 + q2 * q2, keep_dims=True) / (num_heads * head_dim)
        k_mean_sq = tl.sum(k1 * k1 + k2 * k2, keep_dims=True) / (num_heads * head_dim)
        q_rstd = tl.math.rsqrt(q_mean_sq + eps)
        k_rstd = tl.math.rsqrt(k_mean_sq + eps)

        # [BLOCK_SIZE_N, BLOCK_SIZE_HALF_D]
        q1 = q1 * q_rstd * q_w1
        q2 = q2 * q_rstd * q_w2
        k1 = k1 * k_rstd * k_w1
        k2 = k2 * k_rstd * k_w2

        qo1 = tl.fma(-q2, sin, q1 * cos)
        qo2 = tl.fma(q1, sin, q2 * cos)
        ko1 = tl.fma(-k2, sin, k1 * cos)
        ko2 = tl.fma(k1, sin, k2 * cos)

        tl.store(q1_block_ptr, qo1, mask=nd_half_mask)
        tl.store(q2_block_ptr, qo2, mask=nd_half_mask)
        tl.store(k1_block_ptr, ko1, mask=nd_half_mask)
        tl.store(k2_block_ptr, ko2, mask=nd_half_mask)


def qk_rms_norm_rope_triton(
    q: torch.Tensor,  # [b, s, n, d]
    k: torch.Tensor,  # [b, s, n, d]
    qw: torch.Tensor | None,  # [n, d]
    kw: torch.Tensor | None,  # [n, d]
    cos: torch.Tensor,  # [s, d//2]
    sin: torch.Tensor,  # [s, d//2]
    eps: float = 1e-6,
    interleave: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    q = q.contiguous()
    k = k.contiguous()

    b, s, n, d = q.shape

    WITH_WEIGHT = isinstance(qw, torch.Tensor) and isinstance(kw, torch.Tensor)

    if isinstance(qw, torch.Tensor):
        qw = qw.contiguous()
        qw = qw.view(n, d)
    if isinstance(kw, torch.Tensor):
        kw = kw.contiguous()
        kw = kw.view(n, d)

    cos = cos.contiguous()
    sin = sin.contiguous()

    cos = cos.view(s, d // 2)
    sin = sin.view(s, d // 2)

    BLOCK_SIZE_N = triton.next_power_of_2(n)
    BLOCK_SIZE_HALF_D = triton.next_power_of_2(d // 2)

    grid = (s, b)

    with torch.cuda.device(q.device):
        _qk_rms_norm_rope_inplace[grid](
            q,
            k,
            qw,
            kw,
            cos,
            sin,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            qw.stride(0) if WITH_WEIGHT else None,
            qw.stride(1) if WITH_WEIGHT else None,
            cos.stride(0),
            cos.stride(1),
            num_heads=n,
            head_dim=d,
            half_head_dim=d // 2,
            eps=eps,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_HALF_D=BLOCK_SIZE_HALF_D,
            INTERLEAVE=interleave,
            WITH_WEIGHT=WITH_WEIGHT,
            num_warps=8,
        )

    return q, k


def qk_rms_norm_rope_naive(
    q: torch.Tensor,  # [b, s, n, d]
    k: torch.Tensor,  # [b, s, n, d]
    qw: torch.Tensor,  # [n, d]
    kw: torch.Tensor,  # [n, d]
    cos: torch.Tensor,  # [s, d//2]
    sin: torch.Tensor,  # [s, d//2]
    eps: float = 1e-6,
    interleave: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    shape = q.shape
    q = q.flatten(2)
    k = k.flatten(2)

    q = F.rms_norm(q, q.shape[-1:], qw, eps=eps)
    k = F.rms_norm(k, k.shape[-1:], kw, eps=eps)

    q = q.reshape(shape)
    k = k.reshape(shape)

    if interleave:
        q1, q2 = q[..., 0::2], q[..., 1::2]
        k1, k2 = k[..., 0::2], k[..., 1::2]
    else:
        q1, q2 = q.chunk(2, dim=-1)
        k1, k2 = k.chunk(2, dim=-1)

    cos, sin = cos[:, None, :], sin[:, None, :]

    qo1 = (q1 * cos - q2 * sin).to(dtype=q.dtype)
    qo2 = (q2 * cos + q1 * sin).to(dtype=q.dtype)
    ko1 = (k1 * cos - k2 * sin).to(dtype=k.dtype)
    ko2 = (k2 * cos + k1 * sin).to(dtype=k.dtype)

    if interleave:
        q = torch.stack([qo1, qo2], dim=-1).flatten(-2)
        k = torch.stack([ko1, ko2], dim=-1).flatten(-2)
    else:
        q = torch.cat([qo1, qo2], dim=-1)
        k = torch.cat([ko1, ko2], dim=-1)
    return q, k


def broadcast_tensor_for_bsh(tensor, B: int, S: int, H: int):
    if not isinstance(tensor, torch.Tensor):
        return tensor

    if tensor.ndim == 1:
        if tensor.numel() == 1:
            return tensor.view(1, 1, 1).expand(B, S, H)
        else:
            return tensor.view(1, 1, H).expand(B, S, H)
    if tensor.ndim == 2:
        return tensor.view(B, 1, H).expand(B, S, H)
    if tensor.ndim == 3:
        return tensor.expand(B, S, H)
    raise ValueError(f"BSH broadcast: unsupported tensor ndim: {tensor.ndim}.")


@triton.jit
def _scale_shift(
    x_out_ptr,  # [b, s, h]
    x_ptr,  # [b, s, h]
    scale_ptr,  # [b, s, h]
    shift_ptr,  # [b, s, h]
    x_stride_b,
    x_stride_s,
    x_stride_h,
    scale_stride_b,
    scale_stride_s,
    scale_stride_h,
    shift_stride_b,
    shift_stride_s,
    shift_stride_h,
    h,
    BLOCK_SIZE: tl.constexpr,
):
    """
    y = x * scale + shift
    """
    s_id = tl.program_id(0)
    b_id = tl.program_id(1)

    offset = b_id * x_stride_b + s_id * x_stride_s
    x_out_ptr += offset
    x_ptr += offset

    scale_ptr += b_id * scale_stride_b + s_id * scale_stride_s
    shift_ptr += b_id * shift_stride_b + s_id * shift_stride_s

    h_offset = tl.arange(0, BLOCK_SIZE)
    h_mask = h_offset < h

    x = tl.load(x_ptr + h_offset * x_stride_h, mask=h_mask, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr + h_offset * scale_stride_h, mask=h_mask, other=0.0).to(
        tl.float32
    )
    shift = tl.load(shift_ptr + h_offset * shift_stride_h, mask=h_mask, other=0.0).to(
        tl.float32
    )
    x = x * scale + shift

    tl.store(x_out_ptr + h_offset * x_stride_h, x, mask=h_mask)


def scale_shift(
    x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor
) -> torch.Tensor:
    x = x.contiguous()
    y = torch.empty_like(x)

    bsz, *s, h = x.shape
    s = math.prod(s)

    x_view = x.view(bsz, s, h)
    y_view = y.view(bsz, s, h)

    scale = scale.contiguous()
    shift = shift.contiguous()

    scale = broadcast_tensor_for_bsh(scale, bsz, s, h)
    shift = broadcast_tensor_for_bsh(shift, bsz, s, h)

    BLOCK_SIZE = triton.next_power_of_2(h)

    with torch.cuda.device(x.device):
        _scale_shift[(s, bsz)](
            y_view,
            x_view,
            scale,
            shift,
            x_view.stride(0),
            x_view.stride(1),
            x_view.stride(2),
            scale.stride(0),
            scale.stride(1),
            scale.stride(2),
            shift.stride(0),
            shift.stride(1),
            shift.stride(2),
            h,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=8,
        )
    return y


@triton.jit
def _scale_residual_norm_scale_shift(
    resi_out_ptr,  # [b, s, h]
    x_out_ptr,  # [b, s, h]
    resi_ptr,  # [b, s, h]
    x_ptr,  # [b, s, h]
    weight_ptr,  # [h]
    bias_ptr,  # [h]
    gate_ptr,  # [b, s, h]
    scale_ptr,  # [b, s, h]
    shift_ptr,  # [b, s, h]
    x_stride_b,
    x_stride_s,
    x_stride_h,
    res_stride_b,
    res_stride_s,
    res_stride_h,
    gate_stride_b,
    gate_stride_s,
    gate_stride_h,
    scale_stride_b,
    scale_stride_s,
    scale_stride_h,
    shift_stride_b,
    shift_stride_s,
    shift_stride_h,
    h,
    eps,
    NORM_TYPE: tl.constexpr,
    WITH_WEIGHT: tl.constexpr,
    WITH_BIAS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    1. residual_out = residual + gate * x
    2. normed = layernorm(residual_out) or rmsnorm(residual_out)
    3. out = normed * (1 + scale) + shift
    """
    s_id = tl.program_id(0)
    b_id = tl.program_id(1)

    resi_out_ptr += b_id * res_stride_b + s_id * res_stride_s
    x_out_ptr += b_id * x_stride_b + s_id * x_stride_s
    resi_ptr += b_id * res_stride_b + s_id * res_stride_s
    x_ptr += b_id * x_stride_b + s_id * x_stride_s

    gate_ptr += b_id * gate_stride_b + s_id * gate_stride_s
    scale_ptr += b_id * scale_stride_b + s_id * scale_stride_s
    shift_ptr += b_id * shift_stride_b + s_id * shift_stride_s

    h_offset = tl.arange(0, BLOCK_SIZE)
    h_mask = h_offset < h

    x = tl.load(x_ptr + h_offset * x_stride_h, mask=h_mask, other=0.0).to(tl.float32)

    gate = tl.load(gate_ptr + h_offset * gate_stride_h, mask=h_mask, other=0.0).to(
        tl.float32
    )
    resi = tl.load(resi_ptr + h_offset * res_stride_h, mask=h_mask, other=0.0).to(
        tl.float32
    )
    x = resi + gate * x
    tl.store(resi_out_ptr + h_offset * res_stride_h, x, mask=h_mask)

    if WITH_WEIGHT:
        weight = tl.load(weight_ptr + h_offset, mask=h_mask, other=0.0).to(tl.float32)
    else:
        weight = 1.0

    if WITH_BIAS:
        bias = tl.load(bias_ptr + h_offset, mask=h_mask, other=0.0).to(tl.float32)
    else:
        bias = 0.0

    if NORM_TYPE == "rms":
        mean_sq = tl.sum(x * x, keep_dims=True) / h
        rstd = tl.math.rsqrt(mean_sq + eps)
        norm_x = x * rstd * weight
    elif NORM_TYPE == "layer":
        mean_x = tl.sum(x, keep_dims=True) / h
        norm_x = x - mean_x
        norm_x = tl.where(h_mask, norm_x, 0.0)
        var_x = tl.sum(norm_x * norm_x, keep_dims=True) / h
        rstd = tl.math.rsqrt(var_x + eps)
        norm_x = norm_x * rstd * weight + bias
    else:
        tl.static_assert(f"{NORM_TYPE=} is not supported!")
        norm_x = 0.0

    scale = tl.load(scale_ptr + h_offset * scale_stride_h, mask=h_mask, other=0.0).to(
        tl.float32
    )
    shift = tl.load(shift_ptr + h_offset * shift_stride_h, mask=h_mask, other=0.0).to(
        tl.float32
    )
    norm_x = norm_x * (1 + scale) + shift

    tl.store(x_out_ptr + h_offset * x_stride_h, norm_x, mask=h_mask)


def scale_residual_norm_scale_shift(
    residual: torch.Tensor,
    x: torch.Tensor,
    gate: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    w: torch.Tensor | None = None,
    b: torch.Tensor | None = None,
    norm_type: str = "layer",
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    WITH_WEIGHT = isinstance(w, torch.Tensor)
    WITH_BIAS = isinstance(b, torch.Tensor)

    residual = residual.contiguous()
    x = x.contiguous()
    w = w.contiguous() if WITH_WEIGHT else None
    b = b.contiguous() if WITH_BIAS else None

    residual_out = torch.empty_like(residual)
    y = torch.empty_like(x)

    bsz, *s, h = x.shape
    s = math.prod(s)

    residual_out_view = residual_out.view(bsz, s, h)
    y_view = y.view(bsz, s, h)
    residual_view = residual.view(bsz, s, h)
    x_view = x.view(bsz, s, h)

    gate = gate.contiguous()
    scale = scale.contiguous()
    shift = shift.contiguous()

    gate = broadcast_tensor_for_bsh(gate, bsz, s, h)
    scale = broadcast_tensor_for_bsh(scale, bsz, s, h)
    shift = broadcast_tensor_for_bsh(shift, bsz, s, h)

    BLOCK_SIZE = triton.next_power_of_2(h)

    with torch.cuda.device(x.device):
        _scale_residual_norm_scale_shift[(s, bsz)](
            residual_out_view,
            y_view,
            residual_view,
            x_view,
            w,
            b,
            gate,
            scale,
            shift,
            x_view.stride(0),
            x_view.stride(1),
            x_view.stride(2),
            residual_view.stride(0),
            residual_view.stride(1),
            residual_view.stride(2),
            gate.stride(0),
            gate.stride(1),
            gate.stride(2),
            scale.stride(0),
            scale.stride(1),
            scale.stride(2),
            shift.stride(0),
            shift.stride(1),
            shift.stride(2),
            h,
            eps,
            WITH_WEIGHT=WITH_WEIGHT,
            WITH_BIAS=WITH_BIAS,
            NORM_TYPE=norm_type,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=8,
        )
    return y, residual_out


class ScaleResidualNormScaleShift(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        elementwise_affine: bool = False,
        dtype: torch.dtype = torch.float32,
        norm_type: str = "layer",
        prefix: str = "",
    ):
        super().__init__()
        self.eps = eps
        self.dtype = dtype
        self.norm_type = norm_type
        if self.norm_type == "rms":
            self.norm = RMSNorm(hidden_size, eps=eps, dtype=dtype)
        elif self.norm_type == "layer":
            self.norm = FP32LayerNorm(
                hidden_size, elementwise_affine=elementwise_affine, eps=eps, dtype=dtype
            )
        else:
            raise NotImplementedError(f"Norm type {self.norm_type} not implemented")

    def forward(
        self,
        residual: torch.Tensor,
        x: torch.Tensor,
        gate: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return scale_residual_norm_scale_shift(
            residual=residual,
            x=x,
            gate=gate,
            scale=scale,
            shift=shift,
            w=getattr(self.norm, "weight", None),
            b=getattr(self.norm, "bias", None),
            norm_type=self.norm_type,
            eps=self.eps,
        )
