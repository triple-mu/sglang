from typing import Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


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
        q_w_block_ptr = qw_ptr + n_offset * qkw_stride_n + d_offset * qkw_stride_d
        k_w_block_ptr = kw_ptr + n_offset * qkw_stride_n + d_offset * qkw_stride_d

        # [BLOCK_SIZE_N, BLOCK_SIZE_D]
        q = tl.load(q_block_ptr, mask=nd_mask, other=0.0).to(tl.float32)
        k = tl.load(k_block_ptr, mask=nd_mask, other=0.0).to(tl.float32)

        # [BLOCK_SIZE_N, BLOCK_SIZE_D]
        q_w = tl.load(q_w_block_ptr, mask=nd_mask).to(tl.float32)
        k_w = tl.load(k_w_block_ptr, mask=nd_mask).to(tl.float32)

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

        q_w1_block_ptr = qw_ptr + n_offset * qkw_stride_n + half_d_offset * qkw_stride_d
        q_w2_block_ptr = (
            qw_ptr
            + n_offset * qkw_stride_n
            + (half_d_offset + half_head_dim) * qkw_stride_d
        )
        k_w1_block_ptr = kw_ptr + n_offset * qkw_stride_n + half_d_offset * qkw_stride_d
        k_w2_block_ptr = (
            kw_ptr
            + n_offset * qkw_stride_n
            + (half_d_offset + half_head_dim) * qkw_stride_d
        )

        # [BLOCK_SIZE_N, BLOCK_SIZE_HALF_D]
        q1 = tl.load(q1_block_ptr, mask=nd_half_mask, other=0.0).to(tl.float32)
        q2 = tl.load(q2_block_ptr, mask=nd_half_mask, other=0.0).to(tl.float32)
        k1 = tl.load(k1_block_ptr, mask=nd_half_mask, other=0.0).to(tl.float32)
        k2 = tl.load(k2_block_ptr, mask=nd_half_mask, other=0.0).to(tl.float32)

        # [BLOCK_SIZE_N, BLOCK_SIZE_HALF_D]
        q_w1 = tl.load(q_w1_block_ptr, mask=nd_half_mask).to(tl.float32)
        q_w2 = tl.load(q_w2_block_ptr, mask=nd_half_mask).to(tl.float32)
        k_w1 = tl.load(k_w1_block_ptr, mask=nd_half_mask).to(tl.float32)
        k_w2 = tl.load(k_w2_block_ptr, mask=nd_half_mask).to(tl.float32)

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
    qw: torch.Tensor,  # [n, d]
    kw: torch.Tensor,  # [n, d]
    cos: torch.Tensor,  # [s, d//2]
    sin: torch.Tensor,  # [s, d//2]
    eps: float = 1e-6,
    interleave: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    q = q.contiguous()
    k = k.contiguous()
    qw = qw.contiguous()
    kw = kw.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()

    b, s, n, d = q.shape
    qw = qw.view(n, d)
    kw = kw.view(n, d)
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
            qw.stride(0),
            qw.stride(1),
            cos.stride(0),
            cos.stride(1),
            num_heads=n,
            head_dim=d,
            half_head_dim=d // 2,
            eps=eps,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_HALF_D=BLOCK_SIZE_HALF_D,
            INTERLEAVE=interleave,
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
