import itertools
import sys

import pytest
import torch
import triton

from sglang.jit_kernel.diffusion.triton.rotary import ernie_image_rope_qk_inplace
from sglang.jit_kernel.utils import get_ci_test_range
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="stage-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=40, suite="nightly-kernel-1-gpu", nightly=True)

DEVICE = "cuda"
DTYPE = torch.bfloat16
ATOL = 1e-2
RTOL = 1e-2


def ernie_image_rope_qk_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch reference for ernie_image_rope_qk_inplace."""
    q_fp32 = q.float()
    k_fp32 = k.float()
    cos_fp32 = cos[:, :, None, :].float()
    sin_fp32 = sin[:, :, None, :].float()

    q1, q2 = q_fp32.chunk(2, dim=-1)
    k1, k2 = k_fp32.chunk(2, dim=-1)

    qo = q_fp32 * cos_fp32 + torch.cat([-q2, q1], dim=-1) * sin_fp32
    ko = k_fp32 * cos_fp32 + torch.cat([-k2, k1], dim=-1) * sin_fp32
    return qo, ko


BSZ_LIST = [1, 2, 4]
SEQ_LIST = get_ci_test_range([1, 4, 16, 32, 64, 128, 256], [1, 16, 64, 256])
HEADS_LIST = get_ci_test_range([1, 3, 4, 8, 16, 24, 32], [4, 8, 24])
HEAD_DIM_LIST = get_ci_test_range([2, 4, 64, 128], [64, 128])


@pytest.mark.parametrize(
    "batch_size,seq_len,num_heads,head_dim",
    list(
        itertools.product(
            BSZ_LIST,
            SEQ_LIST,
            HEADS_LIST,
            HEAD_DIM_LIST,
        )
    ),
)
def test_ernie_image_rope_qk_inplace(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
) -> None:
    torch.manual_seed(
        batch_size * 1000003 + seq_len * 8191 + num_heads * 127 + head_dim * 17
    )

    q = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=DEVICE, dtype=DTYPE
    )
    k = torch.randn(
        batch_size, seq_len, num_heads, head_dim, device=DEVICE, dtype=DTYPE
    )
    cos = torch.randn(batch_size, seq_len, head_dim, device=DEVICE, dtype=DTYPE)
    sin = torch.randn(batch_size, seq_len, head_dim, device=DEVICE, dtype=DTYPE)

    q_ref, k_ref = ernie_image_rope_qk_reference(q, k, cos, sin)

    q_triton, k_triton = q.clone(), k.clone()
    ernie_image_rope_qk_inplace(q_triton, k_triton, cos, sin)

    triton.testing.assert_close(q_triton.float(), q_ref, atol=ATOL, rtol=RTOL)
    triton.testing.assert_close(k_triton.float(), k_ref, atol=ATOL, rtol=RTOL)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
