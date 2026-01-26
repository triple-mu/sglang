import itertools
from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps, num_stages in itertools.product([2, 4, 8, 16], [2, 3, 4, 5])
    ],
    key=["world_size", "num_heads", "half_head_dim", "head_dim"],
)
@triton.jit
def _rmsnorm_rope_quant_permute(
    q_out_ptr,
    k_out_ptr,
    v_out_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    cos_ptr,
    sin_ptr,
    q_w_ptr,
    k_w_ptr,
    world_size,
    num_heads,
    half_head_dim,
    head_dim,
    stride_out_p,
    stride_out_s,
    stride_out_b,
    stride_out_n,
    stride_out_d,
    stride_in_b,
    stride_in_s,
    stride_in_p,
    stride_in_n,
    stride_in_d,
    stride_w_p,
    cos_stride_s,
    sin_stride_s,
    eps,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_HALF_D: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    INTERLEAVE: tl.constexpr,
):
    s_id = tl.program_id(0)
    b_id = tl.program_id(1)

    q_out_ptr += s_id * stride_out_s + b_id * stride_out_b
    k_out_ptr += s_id * stride_out_s + b_id * stride_out_b
    v_out_ptr += s_id * stride_out_s + b_id * stride_out_b

    q_ptr += s_id * stride_in_s + b_id * stride_in_b
    k_ptr += s_id * stride_in_s + b_id * stride_in_b
    v_ptr += s_id * stride_in_s + b_id * stride_in_b

    cos_ptr += s_id * cos_stride_s
    sin_ptr += s_id * sin_stride_s

    n_offset = tl.arange(0, BLOCK_SIZE_N)[:, None]
    n_mask = n_offset < num_heads

    half_d_offset = tl.arange(0, BLOCK_SIZE_HALF_D)[None, :]
    half_d_mask = half_d_offset < half_head_dim

    d_offset = tl.arange(0, BLOCK_SIZE_D)[None, :]
    d_mask = d_offset < head_dim

    iw_nd_offset = n_offset * stride_in_n + d_offset * stride_in_d
    o_nd_offset = n_offset * stride_out_n + d_offset * stride_out_d
    nd_mask = n_mask & d_mask

    cos = tl.load(cos_ptr + half_d_offset, mask=half_d_mask, other=0.0).to(tl.float32)
    sin = tl.load(sin_ptr + half_d_offset, mask=half_d_mask, other=0.0).to(tl.float32)

    q_sum_sq = tl.zeros([1, 1], dtype=tl.float32)
    k_sum_sq = tl.zeros([1, 1], dtype=tl.float32)

    for p in tl.range(0, world_size):
        q_head_ptr = q_ptr + p * stride_in_p
        k_head_ptr = k_ptr + p * stride_in_p
        q = tl.load(
            q_head_ptr + iw_nd_offset,
            mask=nd_mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        k = tl.load(
            k_head_ptr + iw_nd_offset,
            mask=nd_mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        q_sum_sq += tl.sum(q * q, keep_dims=True)
        k_sum_sq += tl.sum(k * k, keep_dims=True)

    q_mean_sq = q_sum_sq / (world_size * num_heads * head_dim)
    k_mean_sq = k_sum_sq / (world_size * num_heads * head_dim)
    q_rstd = tl.math.rsqrt(q_mean_sq + eps)
    k_rstd = tl.math.rsqrt(k_mean_sq + eps)

    for p in tl.range(0, world_size):

        q_head_ptr = q_ptr + p * stride_in_p
        k_head_ptr = k_ptr + p * stride_in_p
        v_head_ptr = v_ptr + p * stride_in_p
        q_w_head_ptr = q_w_ptr + p * stride_w_p
        k_w_head_ptr = k_w_ptr + p * stride_w_p

        q = tl.load(
            q_head_ptr + iw_nd_offset,
            mask=nd_mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        k = tl.load(
            k_head_ptr + iw_nd_offset,
            mask=nd_mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        v = tl.load(
            v_head_ptr + iw_nd_offset,
            mask=nd_mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        q_w = tl.load(
            q_w_head_ptr + iw_nd_offset,
            mask=nd_mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        k_w = tl.load(
            k_w_head_ptr + iw_nd_offset,
            mask=nd_mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)

        q_out = q * q_rstd * q_w
        k_out = k * k_rstd * k_w
        v_out = v

        if INTERLEAVE:
            q_out_reshape = tl.reshape(q_out, [BLOCK_SIZE_N, BLOCK_SIZE_D // 2, 2])
            q_out_a, q_out_b = tl.split(q_out_reshape)
            q_out_ab = tl.interleave(
                q_out_a * cos - q_out_b * sin, q_out_a * sin + q_out_b * cos
            )
            k_out_reshape = tl.reshape(k_out, [BLOCK_SIZE_N, BLOCK_SIZE_D // 2, 2])
            k_out_a, k_out_b = tl.split(k_out_reshape)
            k_out_ab = tl.interleave(
                k_out_a * cos - k_out_b * sin, k_out_a * sin + k_out_b * cos
            )

        else:
            q_out_reshape = tl.reshape(q_out, [BLOCK_SIZE_N, 2, BLOCK_SIZE_D // 2])
            q_out_a, q_out_b = tl.split(tl.trans(q_out_reshape, 0, 2, 1))
            q_out_cat = tl.join(
                q_out_a * cos - q_out_b * sin, q_out_a * sin + q_out_b * cos
            )
            q_out_ab = tl.reshape(
                tl.trans(q_out_cat, 0, 2, 1), BLOCK_SIZE_N, BLOCK_SIZE_D
            )

            k_out_reshape = tl.reshape(k_out, [BLOCK_SIZE_N, 2, BLOCK_SIZE_D // 2])
            k_out_a, k_out_b = tl.split(tl.trans(k_out_reshape, 0, 2, 1))
            k_out_cat = tl.join(
                k_out_a * cos - k_out_b * sin, k_out_a * sin + k_out_b * cos
            )
            k_out_ab = tl.reshape(
                tl.trans(k_out_cat, 0, 2, 1), BLOCK_SIZE_N, BLOCK_SIZE_D
            )

        q_scale = tl.max(tl.abs(q_out_ab), axis=1, keep_dims=True) / 448.0
        k_scale = tl.max(tl.abs(k_out_ab), axis=1, keep_dims=True) / 448.0
        v_scale = tl.max(tl.abs(v_out), axis=1, keep_dims=True) / 448.0
        q_scale = tl.maximum(q_scale, eps)
        k_scale = tl.maximum(k_scale, eps)
        v_scale = tl.maximum(v_scale, eps)
        quant_q_out = tl.clamp(q_out_ab / q_scale, -448.0, 448.0).to(tl.float8e4nv)
        quant_k_out = tl.clamp(k_out_ab / k_scale, -448.0, 448.0).to(tl.float8e4nv)
        quant_v_out = tl.clamp(v_out / v_scale, -448.0, 448.0).to(tl.float8e4nv)

        q_out_head_ptr = q_out_ptr + p * stride_out_p
        k_out_head_ptr = k_out_ptr + p * stride_out_p
        v_out_head_ptr = v_out_ptr + p * stride_out_p

        tl.store(q_out_head_ptr + o_nd_offset, quant_q_out, mask=nd_mask)
        tl.store(k_out_head_ptr + o_nd_offset, quant_k_out, mask=nd_mask)
        tl.store(v_out_head_ptr + o_nd_offset, quant_v_out, mask=nd_mask)

        q_out_head_ptr = q_out_head_ptr + n_offset * stride_out_n + head_dim
        k_out_head_ptr = k_out_head_ptr + n_offset * stride_out_n + head_dim
        v_out_head_ptr = v_out_head_ptr + n_offset * stride_out_n + head_dim

        q_out_scale_ptr = q_out_head_ptr.to(tl.pointer_type(tl.float32))
        k_out_scale_ptr = k_out_head_ptr.to(tl.pointer_type(tl.float32))
        v_out_scale_ptr = v_out_head_ptr.to(tl.pointer_type(tl.float32))

        tl.store(q_out_scale_ptr, q_scale, mask=n_mask)
        tl.store(k_out_scale_ptr, k_scale, mask=n_mask)
        tl.store(v_out_scale_ptr, v_scale, mask=n_mask)


def rms_norm_rope_quant_permute(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_w: torch.Tensor,
    k_w: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float = 1e-6,
    interleave: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert q.shape == k.shape == v.shape, f"not {q.shape} == {k.shape} == {v.shape}"
    assert q_w.shape == k_w.shape, f"not {q_w.shape} == {k_w.shape}"
    assert (
        q.device == k.device == v.device == q_w.device == k_w.device
    ), f"not {q.device} == {k.device} == {v.device} == {q_w.device} == {k_w.device}"
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    q_w = q_w.contiguous()
    k_w = k_w.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()
    b, local_s, p, local_n, d = q.shape
    half_d = d // 2
    q_w = q_w.reshape(p, local_n, d)
    k_w = k_w.reshape(p, local_n, d)

    BLOCK_SIZE_N = triton.next_power_of_2(local_n)
    BLOCK_SIZE_HALF_D = triton.next_power_of_2(half_d)
    BLOCK_SIZE_D = BLOCK_SIZE_HALF_D * 2

    q_out = torch.empty(
        (p, local_s, b, local_n, d + 4), dtype=torch.float8_e4m3fn, device=q.device
    )
    k_out = torch.empty(
        (p, local_s, b, local_n, d + 4), dtype=torch.float8_e4m3fn, device=k.device
    )
    v_out = torch.empty(
        (p, local_s, b, local_n, d + 4), dtype=torch.float8_e4m3fn, device=v.device
    )

    grid = (local_s, b)

    with torch.cuda.device(q.device):
        _rmsnorm_rope_quant_permute[grid](
            q_out,
            k_out,
            v_out,
            q,
            k,
            v,
            cos,
            sin,
            q_w,
            k_w,
            p,
            local_n,
            half_d,
            d,
            q_out.stride(0),
            q_out.stride(1),
            q_out.stride(2),
            q_out.stride(3),
            q_out.stride(4),
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            q.stride(4),
            q_w.stride(0),
            cos.stride(1),
            sin.stride(1),
            eps=eps,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_HALF_D=BLOCK_SIZE_HALF_D,
            BLOCK_SIZE_D=BLOCK_SIZE_D,
            INTERLEAVE=interleave,
        )

    return q_out, k_out, v_out


@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [2, 4, 8, 16]],
    key=["num_heads", "head_dim"],
)
@triton.jit
def _dequant_permute(
    x_ptr,
    quant_x_ptr,
    stride_x_b,
    stride_x_s,
    stride_x_n,
    stride_qx_s,
    stride_qx_b,
    stride_qx_n,
    num_heads,
    head_dim,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    s_id = tl.program_id(0)
    b_id = tl.program_id(1)
    quant_x_ptr += s_id * stride_qx_s + b_id * stride_qx_b
    x_ptr += b_id * stride_x_b + s_id * stride_x_s

    n_offset = tl.arange(0, BLOCK_SIZE_N)[None, :]
    n_mask = n_offset < num_heads
    d_offset = tl.arange(0, BLOCK_SIZE_D)[:, None]
    d_mask = d_offset < head_dim
    mask = n_mask & d_mask

    x_blk = x_ptr + n_offset * stride_x_n + d_offset
    qx_blk = quant_x_ptr + n_offset * stride_qx_n + d_offset
    scale_ptr = (quant_x_ptr + n_offset * stride_qx_n + head_dim).to(
        tl.pointer_type(tl.float32)
    )

    qx = tl.load(qx_blk, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr, mask=n_mask, other=1.0)
    tl.store(x_blk, qx * scale, mask=mask)


def dequant_permute(
    quant_x: torch.Tensor, dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    global_s, b, local_n, d_ = quant_x.shape
    d = d_ - 4

    BLOCK_SIZE_N = triton.next_power_of_2(local_n)
    BLOCK_SIZE_D = triton.next_power_of_2(d)

    x = torch.empty((b, global_s, local_n, d), dtype=dtype, device=quant_x.device)

    grid = (global_s, b)

    with torch.cuda.device(x.device):
        _dequant_permute[grid](
            x,
            quant_x,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            quant_x.stride(0),
            quant_x.stride(1),
            quant_x.stride(2),
            local_n,
            d,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_D=BLOCK_SIZE_D,
        )

    return x
