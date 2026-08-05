"""Bit-exactness of the fused QKNorm+RoPE+destination-major-pack kernel.

The reference is always the two-kernel chain the fused kernel replaces:
``fused_inplace_qknorm_rope(round_norm_before_rope=True)`` followed by
``pack_qkv_destination_major``. Both are asserted equal bit for bit -- the pack
is a pure permutation and the arithmetic is copied verbatim, so anything short
of ``torch.equal`` would hide a real divergence.
"""

import itertools
import sys

import pytest
import torch

from sglang.kernels.jit.utils import get_ci_test_range
from sglang.kernels.ops.diffusion.qknorm_rope import fused_inplace_qknorm_rope
from sglang.kernels.ops.diffusion.qknorm_rope_pack_qkv import (
    fused_qknorm_rope_pack_qkv,
)
from sglang.kernels.ops.diffusion.triton.ulysses_qkv import (
    pack_qkv_destination_major,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

DEVICE = "cuda"
DTYPE = torch.bfloat16
EPS = 1e-5
ROPE_BASE = 10000.0


def _cos_sin_cache(rope_dim: int, max_position: int) -> torch.Tensor:
    inv_freq = 1.0 / (
        ROPE_BASE
        ** (torch.arange(0, rope_dim, 2, dtype=torch.float32, device=DEVICE) / rope_dim)
    )
    t = torch.arange(max_position, dtype=torch.float32, device=DEVICE)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(DTYPE)


def _make_qkv(
    rows: int, num_heads: int, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """q/k/v as strided views of one fused qkv buffer, as the DiT produces them."""
    inner = num_heads * head_dim
    qkv = torch.randn(rows, 3 * inner, device=DEVICE, dtype=DTYPE)
    return tuple(
        chunk.view(rows, num_heads, head_dim) for chunk in qkv.split(inner, dim=-1)
    )


def _reference_packed(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    world_size: int,
    is_neox: bool,
) -> torch.Tensor:
    q_ref, k_ref = q.clone(), k.clone()
    fused_inplace_qknorm_rope(
        q_ref,
        k_ref,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        is_neox=is_neox,
        eps=EPS,
        head_dim=q.shape[-1],
        rope_dim=cos_sin_cache.shape[-1],
        round_norm_before_rope=True,
    )
    return pack_qkv_destination_major(q_ref, k_ref, v, world_size)


HEADS_WORLD = get_ci_test_range(
    [(14, 1), (14, 2), (14, 7), (14, 14), (56, 1), (56, 2), (56, 4), (56, 8)],
    [(14, 2), (56, 4), (56, 8)],
)
ROWS_LIST = get_ci_test_range([1, 17, 129, 257, 1024, 4097], [1, 257, 4097])
DIMS_LIST = get_ci_test_range(
    [(128, 96), (128, 128), (64, 64), (256, 128)], [(128, 96)]
)
POSITION_DTYPES = [torch.int32, torch.int64]


@pytest.mark.parametrize(
    "heads_world,rows,dims,position_dtype",
    list(
        itertools.product(HEADS_WORLD, ROWS_LIST, DIMS_LIST, POSITION_DTYPES),
    ),
)
def test_qknorm_rope_pack_qkv_bit_exact(
    heads_world: tuple[int, int],
    rows: int,
    dims: tuple[int, int],
    position_dtype: torch.dtype,
) -> None:
    num_heads, world_size = heads_world
    head_dim, rope_dim = dims
    q, k, v = _make_qkv(rows, num_heads, head_dim)
    q_weight = torch.randn(head_dim, device=DEVICE, dtype=DTYPE)
    k_weight = torch.randn(head_dim, device=DEVICE, dtype=DTYPE)
    positions = torch.randint(0, 4096, (rows,), device=DEVICE, dtype=position_dtype)
    cos_sin_cache = _cos_sin_cache(rope_dim, 4096)

    expected = _reference_packed(
        q,
        k,
        v,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        world_size,
        is_neox=True,
    )
    actual = fused_qknorm_rope_pack_qkv(
        q,
        k,
        v,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        world_size=world_size,
        is_neox=True,
        eps=EPS,
        head_dim=head_dim,
        rope_dim=rope_dim,
    )

    assert actual.shape == expected.shape
    assert torch.equal(
        actual, expected
    ), f"max abs diff {(actual.float() - expected.float()).abs().max().item()}"


@pytest.mark.parametrize("is_neox", [True, False])
def test_qknorm_rope_pack_qkv_contiguous_inputs(is_neox: bool) -> None:
    """Independent contiguous q/k/v, non-neox rotation, and the H3 shape."""
    rows, num_heads, head_dim, rope_dim, world_size = 9456, 56, 128, 96, 4
    q = torch.randn(rows, num_heads, head_dim, device=DEVICE, dtype=DTYPE)
    k = torch.randn(rows, num_heads, head_dim, device=DEVICE, dtype=DTYPE)
    v = torch.randn(rows, num_heads, head_dim, device=DEVICE, dtype=DTYPE)
    q_weight = torch.randn(head_dim, device=DEVICE, dtype=DTYPE)
    k_weight = torch.randn(head_dim, device=DEVICE, dtype=DTYPE)
    positions = torch.arange(rows, device=DEVICE, dtype=torch.int64)
    cos_sin_cache = _cos_sin_cache(rope_dim, rows)

    expected = _reference_packed(
        q,
        k,
        v,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        world_size,
        is_neox,
    )
    actual = fused_qknorm_rope_pack_qkv(
        q,
        k,
        v,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        world_size=world_size,
        is_neox=is_neox,
        eps=EPS,
        head_dim=head_dim,
        rope_dim=rope_dim,
    )
    assert torch.equal(actual, expected)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
