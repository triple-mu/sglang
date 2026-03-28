import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.srt.utils.custom_op import register_custom_op


def qk_rmsnorm_with_rotary_naive(
    q: torch.Tensor,
    k: torch.Tensor,
    qw: torch.Tensor | None,
    kw: torch.Tensor | None,
    cos_sin_cache: torch.Tensor,
    position_offset: int = 0,
    eps: float = 1e-6,
    interleave: bool = True,
) -> None:
    """Naive PyTorch implementation for correctness checking."""
    assert (
        q.is_contiguous()
        and k.is_contiguous()
        and q.ndim == k.ndim == 4
        and q.shape == k.shape
        and cos_sin_cache.ndim == 2
    )
    b, s, n, d = q.shape
    rope_dim = cos_sin_cache.shape[1]
    assert rope_dim <= d and rope_dim % 2 == 0
    if (not interleave) and rope_dim < d:
        raise ValueError("NeoX mode currently requires rope_dim == head_dim")

    device = q.device
    half_rope_dim = rope_dim // 2
    pos = torch.arange(s, device=device, dtype=torch.long) + position_offset
    cache = cos_sin_cache.to(device=device, dtype=torch.float32).contiguous()
    cache_slice = cache.index_select(0, pos)
    cos = cache_slice[:, :half_rope_dim].unsqueeze(0).unsqueeze(2)
    sin = cache_slice[:, half_rope_dim:].unsqueeze(0).unsqueeze(2)

    qf = q.to(device=device, dtype=torch.float32)
    kf = k.to(device=device, dtype=torch.float32)
    qw_f = (
        qw.to(device=device, dtype=torch.float32)
        if isinstance(qw, torch.Tensor)
        else None
    )
    kw_f = (
        kw.to(device=device, dtype=torch.float32)
        if isinstance(kw, torch.Tensor)
        else None
    )
    qf = F.rms_norm(qf, normalized_shape=(d,), weight=qw_f, eps=eps)
    kf = F.rms_norm(kf, normalized_shape=(d,), weight=kw_f, eps=eps)

    q_out = qf.clone()
    k_out = kf.clone()
    q_rot = qf[..., :rope_dim]
    k_rot = kf[..., :rope_dim]

    if interleave:
        q_pair = q_rot.view(b, s, n, half_rope_dim, 2)
        k_pair = k_rot.view(b, s, n, half_rope_dim, 2)
        q0, q1 = q_pair[..., 0], q_pair[..., 1]
        k0, k1 = k_pair[..., 0], k_pair[..., 1]
        q0_new = q0 * cos - q1 * sin
        q1_new = q1 * cos + q0 * sin
        k0_new = k0 * cos - k1 * sin
        k1_new = k1 * cos + k0 * sin
        q_out[..., :rope_dim] = torch.stack((q0_new, q1_new), dim=-1).reshape(
            b, s, n, rope_dim
        )
        k_out[..., :rope_dim] = torch.stack((k0_new, k1_new), dim=-1).reshape(
            b, s, n, rope_dim
        )
    else:
        q0 = q_rot[..., :half_rope_dim]
        q1 = q_rot[..., half_rope_dim:rope_dim]
        k0 = k_rot[..., :half_rope_dim]
        k1 = k_rot[..., half_rope_dim:rope_dim]
        q0_new = q0 * cos - q1 * sin
        q1_new = q1 * cos + q0 * sin
        k0_new = k0 * cos - k1 * sin
        k1_new = k1 * cos + k0 * sin
        q_out[..., :half_rope_dim] = q0_new
        q_out[..., half_rope_dim:rope_dim] = q1_new
        k_out[..., :half_rope_dim] = k0_new
        k_out[..., half_rope_dim:rope_dim] = k1_new

    q.copy_(q_out.to(dtype=q.dtype))
    k.copy_(k_out.to(dtype=k.dtype))


@triton.jit
def _qk_rmsnorm_with_rotary(
    q_ptr,
    k_ptr,
    qw_ptr,
    kw_ptr,
    cos_sin_ptr,
    qk_stride_b,
    qk_stride_s,
    qk_stride_n,
    emb_stride_s,
    num_heads: tl.constexpr,
    head_dim,
    rope_dim,
    half_head_dim,
    half_rope_dim,
    position_offset,
    eps,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_HALF_D: tl.constexpr,
    INTERLEAVE: tl.constexpr,
    WITH_WEIGHT: tl.constexpr,
):
    BLOCK_SIZE_D: tl.constexpr = BLOCK_SIZE_HALF_D * 2
    dtype = q_ptr.dtype.element_ty

    s_id = tl.program_id(0)
    b_id = tl.program_id(1)

    offset_b_s = b_id * qk_stride_b + s_id * qk_stride_s
    q_base = q_ptr + offset_b_s
    k_base = k_ptr + offset_b_s
    cos_sin_ptr += (s_id + position_offset) * emb_stride_s

    d1_offset = tl.arange(0, BLOCK_SIZE_HALF_D)
    d2_offset = half_head_dim + d1_offset
    half_d_mask = d1_offset < half_head_dim
    d_offset = tl.arange(0, BLOCK_SIZE_D)
    d_mask = d_offset < head_dim

    rope_half_mask = d1_offset < half_rope_dim
    cos = tl.load(cos_sin_ptr + d1_offset, mask=rope_half_mask, other=1.0).to(
        tl.float32
    )
    sin = tl.load(
        cos_sin_ptr + half_rope_dim + d1_offset, mask=rope_half_mask, other=0.0
    ).to(tl.float32)

    if WITH_WEIGHT:
        if INTERLEAVE:
            q_w = tl.load(qw_ptr + d_offset, mask=d_mask).to(tl.float32)
            k_w = tl.load(kw_ptr + d_offset, mask=d_mask).to(tl.float32)
        else:
            q_w1 = tl.load(qw_ptr + d1_offset, mask=half_d_mask).to(tl.float32)
            q_w2 = tl.load(qw_ptr + d2_offset, mask=half_d_mask).to(tl.float32)
            k_w1 = tl.load(kw_ptr + d1_offset, mask=half_d_mask).to(tl.float32)
            k_w2 = tl.load(kw_ptr + d2_offset, mask=half_d_mask).to(tl.float32)
    else:
        if INTERLEAVE:
            q_w, k_w = 1.0, 1.0
        else:
            q_w1, q_w2, k_w1, k_w2 = 1.0, 1.0, 1.0, 1.0

    if INTERLEAVE:
        q_block_ptr = tl.make_block_ptr(
            base=q_base,
            shape=(num_heads, head_dim),
            strides=(qk_stride_n, 1),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
            order=(1, 0),
        )
        k_block_ptr = tl.make_block_ptr(
            base=k_base,
            shape=(num_heads, head_dim),
            strides=(qk_stride_n, 1),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_D),
            order=(1, 0),
        )
    else:
        q1_block_ptr = tl.make_block_ptr(
            base=q_base,
            shape=(num_heads, head_dim),
            strides=(qk_stride_n, 1),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_HALF_D),
            order=(1, 0),
        )
        q2_block_ptr = tl.make_block_ptr(
            base=q_base,
            shape=(num_heads, head_dim),
            strides=(qk_stride_n, 1),
            offsets=(0, half_head_dim),
            block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_HALF_D),
            order=(1, 0),
        )
        k1_block_ptr = tl.make_block_ptr(
            base=k_base,
            shape=(num_heads, head_dim),
            strides=(qk_stride_n, 1),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_HALF_D),
            order=(1, 0),
        )
        k2_block_ptr = tl.make_block_ptr(
            base=k_base,
            shape=(num_heads, head_dim),
            strides=(qk_stride_n, 1),
            offsets=(0, half_head_dim),
            block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_HALF_D),
            order=(1, 0),
        )

    for _ in tl.static_range(0, num_heads, BLOCK_SIZE_N):
        if INTERLEAVE:
            q = tl.load(q_block_ptr, boundary_check=(0, 1), padding_option="zero").to(
                tl.float32
            )
            k = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero").to(
                tl.float32
            )

            q_mean_sq = tl.sum(q * q, axis=1, keep_dims=True) / head_dim
            k_mean_sq = tl.sum(k * k, axis=1, keep_dims=True) / head_dim
            q_rstd = tl.math.rsqrt(q_mean_sq + eps)
            k_rstd = tl.math.rsqrt(k_mean_sq + eps)

            q = q * q_rstd * q_w
            k = k * k_rstd * k_w

            q1, q2 = tl.split(tl.reshape(q, [BLOCK_SIZE_N, BLOCK_SIZE_HALF_D, 2]))
            k1, k2 = tl.split(tl.reshape(k, [BLOCK_SIZE_N, BLOCK_SIZE_HALF_D, 2]))

            qo1 = tl.fma(-q2, sin, q1 * cos)
            qo2 = tl.fma(q1, sin, q2 * cos)
            ko1 = tl.fma(-k2, sin, k1 * cos)
            ko2 = tl.fma(k1, sin, k2 * cos)

            q = tl.interleave(qo1, qo2)
            k = tl.interleave(ko1, ko2)

            tl.store(q_block_ptr, q.to(dtype), boundary_check=(0, 1))
            tl.store(k_block_ptr, k.to(dtype), boundary_check=(0, 1))

            q_block_ptr = tl.advance(q_block_ptr, (BLOCK_SIZE_N, 0))
            k_block_ptr = tl.advance(k_block_ptr, (BLOCK_SIZE_N, 0))

        else:
            q1 = tl.load(q1_block_ptr, boundary_check=(0, 1), padding_option="zero").to(
                tl.float32
            )
            q2 = tl.load(q2_block_ptr, boundary_check=(0, 1), padding_option="zero").to(
                tl.float32
            )
            k1 = tl.load(k1_block_ptr, boundary_check=(0, 1), padding_option="zero").to(
                tl.float32
            )
            k2 = tl.load(k2_block_ptr, boundary_check=(0, 1), padding_option="zero").to(
                tl.float32
            )

            q_mean_sq = tl.sum(q1 * q1 + q2 * q2, axis=1, keep_dims=True) / head_dim
            k_mean_sq = tl.sum(k1 * k1 + k2 * k2, axis=1, keep_dims=True) / head_dim
            q_rstd = tl.math.rsqrt(q_mean_sq + eps)
            k_rstd = tl.math.rsqrt(k_mean_sq + eps)

            q1 = q1 * q_rstd * q_w1
            q2 = q2 * q_rstd * q_w2
            k1 = k1 * k_rstd * k_w1
            k2 = k2 * k_rstd * k_w2

            qo1 = tl.fma(-q2, sin, q1 * cos)
            qo2 = tl.fma(q1, sin, q2 * cos)
            ko1 = tl.fma(-k2, sin, k1 * cos)
            ko2 = tl.fma(k1, sin, k2 * cos)

            tl.store(q1_block_ptr, qo1.to(dtype), boundary_check=(0, 1))
            tl.store(q2_block_ptr, qo2.to(dtype), boundary_check=(0, 1))
            tl.store(k1_block_ptr, ko1.to(dtype), boundary_check=(0, 1))
            tl.store(k2_block_ptr, ko2.to(dtype), boundary_check=(0, 1))

            q1_block_ptr = tl.advance(q1_block_ptr, (BLOCK_SIZE_N, 0))
            q2_block_ptr = tl.advance(q2_block_ptr, (BLOCK_SIZE_N, 0))
            k1_block_ptr = tl.advance(k1_block_ptr, (BLOCK_SIZE_N, 0))
            k2_block_ptr = tl.advance(k2_block_ptr, (BLOCK_SIZE_N, 0))


@register_custom_op(op_name="qk_rmsnorm_with_rotary")
def qk_rmsnorm_with_rotary(
    q: torch.Tensor,
    k: torch.Tensor,
    qw: torch.Tensor | None,
    kw: torch.Tensor | None,
    cos_sin_cache: torch.Tensor,
    position_offset: int = 0,
    eps: float = 1e-6,
    interleave: bool = True,
) -> None:
    assert (
        q.is_contiguous()
        and k.is_contiguous()
        and q.ndim == k.ndim == 4
        and cos_sin_cache.ndim == 2
    )
    b, s, n, d = q.shape
    rope_dim = cos_sin_cache.shape[1]
    assert rope_dim <= d and rope_dim % 2 == 0
    if (not interleave) and rope_dim < d:
        raise ValueError("NeoX mode currently requires rope_dim == head_dim")

    WITH_WEIGHT = isinstance(qw, torch.Tensor) and isinstance(kw, torch.Tensor)

    cos_sin_cache = cos_sin_cache.contiguous()

    BLOCK_SIZE_HALF_D = triton.next_power_of_2(d // 2)
    BLOCK_SIZE_N = 1024 // BLOCK_SIZE_HALF_D

    grid = (s, b)

    with torch.cuda.device(q.device):
        _qk_rmsnorm_with_rotary[grid](
            q,
            k,
            qw,
            kw,
            cos_sin_cache,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            cos_sin_cache.stride(0),
            num_heads=n,
            head_dim=d,
            rope_dim=rope_dim,
            half_head_dim=d // 2,
            half_rope_dim=rope_dim // 2,
            position_offset=position_offset,
            eps=eps,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_HALF_D=BLOCK_SIZE_HALF_D,
            INTERLEAVE=interleave,
            WITH_WEIGHT=WITH_WEIGHT,
            num_warps=8,
        )
