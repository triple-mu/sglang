import itertools

import pytest
import torch
import torch.nn.functional as F
import triton


@torch.compile()
def qk_rms_norm_cross_head_with_rope_naive(
    q: torch.Tensor,
    k: torch.Tensor,
    qw: torch.Tensor,
    kw: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float = 1e-6,
    interleave: bool = True,
) -> None:
    shape = q.shape

    ori_q, ori_k = q, k
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
    ori_q.copy_(q)
    ori_k.copy_(k)


BS_LIST = [1]
SEQ_LIST = [1024, 2048, 4096, 8192]
NUM_HEADS_LIST = [16, 24, 32, 40, 48, 64]
HEAD_DIM_LIST = [64, 128, 256]
DEVICE = "cuda"
DTYPE = torch.bfloat16


@pytest.mark.parametrize(
    "batch_size,seq_len,num_heads,head_dim",
    list(itertools.product(BS_LIST, SEQ_LIST, NUM_HEADS_LIST, HEAD_DIM_LIST)),
)
def test_qknorm_with_rotary(
    batch_size: int, seq_len: int, num_heads: int, head_dim: int
) -> None:
    from sglang.jit_kernel.diffusion.triton.rmsnorm_with_rotary import (
        qk_rms_norm_cross_head_with_rope,
    )

    q = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=DEVICE, dtype=DTYPE
    )
    k = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=DEVICE, dtype=DTYPE
    )
    q_weight = torch.randn(num_heads * head_dim, device=DEVICE, dtype=DTYPE)
    k_weight = torch.randn(num_heads * head_dim, device=DEVICE, dtype=DTYPE)
    cos = torch.rand(seq_len, head_dim // 2, device=DEVICE, dtype=torch.float32)
    sin = torch.rand(seq_len, head_dim // 2, device=DEVICE, dtype=torch.float32)

    q_k_triton = (q.clone(), k.clone())
    q_k_naive = (q.clone(), k.clone())

    qk_rms_norm_cross_head_with_rope(
        q_k_triton[0],
        q_k_triton[1],
        q_weight,
        k_weight,
        cos,
        sin,
        eps=1e-6,
        interleave=True,
    )
    qk_rms_norm_cross_head_with_rope_naive(
        q_k_naive[0],
        q_k_naive[1],
        q_weight,
        k_weight,
        cos,
        sin,
        eps=1e-6,
        interleave=True,
    )

    q_k_triton = (q.clone(), k.clone())
    q_k_naive = (q.clone(), k.clone())

    qk_rms_norm_cross_head_with_rope(
        q_k_triton[0],
        q_k_triton[1],
        q_weight,
        k_weight,
        cos,
        sin,
        eps=1e-6,
        interleave=False,
    )
    qk_rms_norm_cross_head_with_rope_naive(
        q_k_naive[0],
        q_k_naive[1],
        q_weight,
        k_weight,
        cos,
        sin,
        eps=1e-6,
        interleave=False,
    )

    triton.testing.assert_close(q_k_triton[0], q_k_naive[0], atol=1e-2, rtol=1e-2)
    triton.testing.assert_close(q_k_triton[1], q_k_naive[1], atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__])
