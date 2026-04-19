import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore

from sglang.multimodal_gen.runtime.platforms import current_platform


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_HEADS": 1, "BLOCK_HS_HALF": 32}, num_warps=2),
        triton.Config({"BLOCK_HEADS": 2, "BLOCK_HS_HALF": 32}, num_warps=2),
        triton.Config({"BLOCK_HEADS": 4, "BLOCK_HS_HALF": 32}, num_warps=4),
        triton.Config({"BLOCK_HEADS": 4, "BLOCK_HS_HALF": 64}, num_warps=4),
        triton.Config({"BLOCK_HEADS": 8, "BLOCK_HS_HALF": 64}, num_warps=8),
    ],
    key=["num_heads", "head_size"],
)
@triton.jit
def _rotary_embedding_kernel(
    output_ptr,
    x_ptr,
    cos_ptr,
    sin_ptr,
    num_heads,
    head_size,
    num_tokens,
    stride_out_bt,
    stride_out_head,
    stride_x_bt,
    stride_x_head,
    stride_cos_row,
    stride_sin_row,
    BLOCK_HEADS: tl.constexpr,
    BLOCK_HS_HALF: tl.constexpr,
):
    bt_idx = tl.program_id(0)
    head_block_idx = tl.program_id(1)
    token_idx = bt_idx % num_tokens

    cos_row_ptr = cos_ptr + token_idx * stride_cos_row
    sin_row_ptr = sin_ptr + token_idx * stride_sin_row
    head_offsets = head_block_idx * BLOCK_HEADS + tl.arange(0, BLOCK_HEADS)
    head_mask = head_offsets < num_heads

    head_size_half = head_size // 2
    x_row_ptrs = x_ptr + bt_idx * stride_x_bt + head_offsets[:, None] * stride_x_head
    output_row_ptrs = (
        output_ptr + bt_idx * stride_out_bt + head_offsets[:, None] * stride_out_head
    )

    for block_start in range(0, head_size_half, BLOCK_HS_HALF):
        offsets_half = block_start + tl.arange(0, BLOCK_HS_HALF)
        half_mask = offsets_half < head_size_half
        mask = head_mask[:, None] & half_mask[None, :]

        cos_vals = tl.load(cos_row_ptr + offsets_half, mask=half_mask, other=0.0)
        sin_vals = tl.load(sin_row_ptr + offsets_half, mask=half_mask, other=0.0)

        offsets_x1 = 2 * offsets_half
        offsets_x2 = 2 * offsets_half + 1

        x1_vals = tl.load(x_row_ptrs + offsets_x1[None, :], mask=mask, other=0.0)
        x2_vals = tl.load(x_row_ptrs + offsets_x2[None, :], mask=mask, other=0.0)

        x1_fp32 = x1_vals.to(tl.float32)
        x2_fp32 = x2_vals.to(tl.float32)
        cos_fp32 = cos_vals.to(tl.float32)[None, :]
        sin_fp32 = sin_vals.to(tl.float32)[None, :]
        o1_vals = tl.fma(-x2_fp32, sin_fp32, x1_fp32 * cos_fp32)
        o2_vals = tl.fma(x1_fp32, sin_fp32, x2_fp32 * cos_fp32)

        tl.store(
            output_row_ptrs + offsets_x1[None, :],
            o1_vals.to(x1_vals.dtype),
            mask=mask,
        )
        tl.store(
            output_row_ptrs + offsets_x2[None, :],
            o2_vals.to(x2_vals.dtype),
            mask=mask,
        )


def apply_rotary_embedding(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, interleaved: bool = False
) -> torch.Tensor:
    output = torch.empty_like(x)

    if x.dim() > 3:
        bsz, num_tokens, num_heads, head_size = x.shape
    else:
        num_tokens, num_heads, head_size = x.shape
        bsz = 1

    assert head_size % 2 == 0, "head_size must be divisible by 2"

    x_reshaped = x.view(bsz * num_tokens, num_heads, head_size)
    output_reshaped = output.view(bsz * num_tokens, num_heads, head_size)

    if interleaved and cos.shape[-1] == head_size:
        cos = cos[..., ::2].contiguous()
        sin = sin[..., ::2].contiguous()
    else:
        cos = cos.contiguous()
        sin = sin.contiguous()

    _rotary_embedding_kernel[
        lambda META: (bsz * num_tokens, triton.cdiv(num_heads, META["BLOCK_HEADS"]))
    ](
        output_reshaped,
        x_reshaped,
        cos,
        sin,
        num_heads,
        head_size,
        num_tokens,
        output_reshaped.stride(0),
        output_reshaped.stride(1),
        x_reshaped.stride(0),
        x_reshaped.stride(1),
        cos.stride(0),
        sin.stride(0),
    )

    return output


@triton.jit
def _ernie_image_rope_qk_inplace(
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    qk_stride_b,
    qk_stride_s,
    qk_stride_n,
    cos_sin_stride_b,
    cos_sin_stride_s,
    num_heads,
    head_dim,
    NUM_HEADS: tl.constexpr,
    HALF_DIM: tl.constexpr,
):
    s_id = tl.program_id(0)
    b_id = tl.program_id(1)

    half_dim = head_dim // 2
    h_offset = tl.arange(0, HALF_DIM)[None, :]
    h_mask = h_offset < half_dim
    n_offset = tl.arange(0, NUM_HEADS)[:, None]
    n_mask = n_offset < num_heads
    nh_mask = n_mask & h_mask

    q_ptr += b_id * qk_stride_b + s_id * qk_stride_s
    k_ptr += b_id * qk_stride_b + s_id * qk_stride_s
    cos_ptr += b_id * cos_sin_stride_b + s_id * cos_sin_stride_s
    sin_ptr += b_id * cos_sin_stride_b + s_id * cos_sin_stride_s

    # cos/sin are half-size (head_dim // 2).  The full-size pattern is
    # repeat_interleave(val, 2): [v0, v0, v1, v1, ...].  For the first
    # half of q (index j in 0..H-1) the lookup is cos_half[j // 2]; for
    # the second half (index H+j) it is cos_half[H//2 + j // 2].
    cos1 = tl.load(cos_ptr + h_offset // 2, mask=h_mask, other=0.0).to(tl.float32)
    sin1 = tl.load(sin_ptr + h_offset // 2, mask=h_mask, other=0.0).to(tl.float32)
    cos2 = tl.load(cos_ptr + half_dim // 2 + h_offset // 2, mask=h_mask, other=0.0).to(
        tl.float32
    )
    sin2 = tl.load(sin_ptr + half_dim // 2 + h_offset // 2, mask=h_mask, other=0.0).to(
        tl.float32
    )

    q_base = q_ptr + n_offset * qk_stride_n
    k_base = k_ptr + n_offset * qk_stride_n

    q1 = tl.load(q_base + h_offset, mask=nh_mask, other=0.0).to(tl.float32)
    q2 = tl.load(q_base + half_dim + h_offset, mask=nh_mask, other=0.0).to(tl.float32)
    k1 = tl.load(k_base + h_offset, mask=nh_mask, other=0.0).to(tl.float32)
    k2 = tl.load(k_base + half_dim + h_offset, mask=nh_mask, other=0.0).to(tl.float32)

    # First half:  out[j]   = q1[j]*cos1[j] - q2[j]*sin1[j]
    # Second half: out[H+j] = q2[j]*cos2[j] + q1[j]*sin2[j]
    qo1 = tl.fma(-q2, sin1, q1 * cos1)
    qo2 = tl.fma(q1, sin2, q2 * cos2)
    ko1 = tl.fma(-k2, sin1, k1 * cos1)
    ko2 = tl.fma(k1, sin2, k2 * cos2)

    tl.store(q_base + h_offset, qo1, mask=nh_mask)
    tl.store(q_base + half_dim + h_offset, qo2, mask=nh_mask)
    tl.store(k_base + h_offset, ko1, mask=nh_mask)
    tl.store(k_base + half_dim + h_offset, ko2, mask=nh_mask)


def ernie_image_rope_qk_inplace(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply ErnieImage-style rotary embedding to Q and K tensors **in-place**.

    The rotation convention is ``[-x2, x1]`` (half-split, not interleaved),
    matching the original ErnieImage ``_apply_rotary_bshd`` implementation.

    Args:
        q: Query tensor of shape ``(B, S, num_heads, head_dim)``.
        k: Key tensor of shape ``(B, S, num_heads, head_dim)``.
        cos: Cosine cache of shape ``(B, S, head_dim // 2)`` — half-size,
            without the ``repeat_interleave`` expansion.
        sin: Sine cache of shape ``(B, S, head_dim // 2)``.

    Returns:
        ``(q, k)`` with rotary embedding applied in-place.
    """
    if q.ndim != 4:
        raise ValueError(f"q must be 4-D (B, S, H, D), got ndim={q.ndim}")
    if k.shape != q.shape:
        raise ValueError(f"q and k shape mismatch: {q.shape} vs {k.shape}")
    if cos.shape != sin.shape:
        raise ValueError(f"cos and sin shape mismatch: {cos.shape} vs {sin.shape}")
    if cos.ndim != 3:
        raise ValueError(f"cos/sin must be 3-D (B, S, D), got ndim={cos.ndim}")
    if cos.size(0) != q.size(0) or cos.size(1) != q.size(1):
        raise ValueError(
            f"cos/sin batch/seq dims must match q: "
            f"cos {cos.shape[:2]} vs q {q.shape[:2]}"
        )
    if q.size(3) % 2 != 0:
        raise ValueError(f"head_dim must be even, got {q.size(3)}")
    if cos.size(2) != q.size(3) // 2:
        raise ValueError(
            f"cos/sin last dim must be head_dim // 2: "
            f"cos {cos.size(2)} vs head_dim // 2 = {q.size(3) // 2}"
        )
    if not q.is_contiguous() or not k.is_contiguous():
        raise ValueError("q and k must be contiguous")
    if not cos.is_contiguous() or not sin.is_contiguous():
        raise ValueError("cos and sin must be contiguous")

    bsz, seq_len, num_heads, head_dim = q.shape
    NUM_HEADS = triton.next_power_of_2(num_heads)
    HALF_DIM = triton.next_power_of_2(head_dim // 2)
    grid = (seq_len, bsz)
    with torch.cuda.device(q.device):
        _ernie_image_rope_qk_inplace[grid](
            q,
            k,
            cos,
            sin,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            cos.stride(0),
            cos.stride(1),
            num_heads,
            head_dim,
            NUM_HEADS=NUM_HEADS,
            HALF_DIM=HALF_DIM,
            num_warps=8,
        )
    return q, k


if current_platform.is_npu():
    from .npu_fallback import apply_rotary_embedding_native

    apply_rotary_embedding = apply_rotary_embedding_native

if current_platform.is_mps():
    from .mps_fallback import apply_rotary_embedding_native

    apply_rotary_embedding = apply_rotary_embedding_native

if current_platform.is_npu() or current_platform.is_mps():

    def ernie_image_rope_qk_inplace(  # type: ignore[misc]
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pure-PyTorch fallback of :func:`ernie_image_rope_qk_inplace`."""
        # Expand half-size cos/sin to full size
        cos = cos.repeat_interleave(2, dim=-1)[:, :, None, :]
        sin = sin.repeat_interleave(2, dim=-1)[:, :, None, :]

        q1, q2 = q.chunk(2, dim=-1)
        k1, k2 = k.chunk(2, dim=-1)
        q_rot = torch.cat([-q2, q1], dim=-1)
        k_rot = torch.cat([-k2, k1], dim=-1)

        qo = q * cos + q_rot * sin
        ko = k * cos + k_rot * sin
        q.copy_(qo)
        k.copy_(ko)

        return q, k
