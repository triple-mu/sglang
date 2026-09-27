import hashlib
import sys

import pytest
import torch

from sglang.kernels.ops.diffusion import (
    can_use_sage_block_sparse_attn_sm120,
    sage_block_sparse_attn_sm120,
    sage_block_sparse_dense_block_index,
)
from sglang.kernels.ops.diffusion.attention import sage_block_sparse_sm120_cake as cake
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.sage_sm120_reference import (
    K_BLOCK,
    block_sparse_attention_reference,
    dequantize_sage,
    quantize_sage_kv,
    quantize_sage_q,
)

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

requires_sm120 = pytest.mark.skipif(
    not can_use_sage_block_sparse_attn_sm120(),
    reason="vendored Cake kernel targets compute capability 12.0 only",
)


def test_vendored_kernel_hash_matches_manifest():
    """The generated source is a byte-exact copy; nobody edits it in place."""
    digest = hashlib.sha256(cake.vendored_kernel_path().read_bytes()).hexdigest()
    assert digest == cake._VENDOR_KERNEL_SHA256
    cake.verify_vendored_kernel()


def _operands(batch, heads, seqlen_q, seqlen_k, *, seed=0, alloc_q=None, alloc_k=None):
    """Quantised operands for a live ``seqlen`` prefix inside an ``alloc``-row buffer.

    The bf16 sources are drawn at the allocated size so two calls that differ
    only in ``alloc_*`` quantise the same live rows.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    alloc_q = seqlen_q if alloc_q is None else alloc_q
    alloc_k = seqlen_k if alloc_k is None else alloc_k
    q = torch.randn(batch, heads, max(alloc_q, 1024), 128, generator=g)[:, :, :alloc_q]
    k = torch.randn(batch, heads, max(alloc_k, 1024), 128, generator=g)[:, :, :alloc_k]
    v = torch.randn(batch, heads, max(alloc_k, 1024), 128, generator=g)[:, :, :alloc_k]
    q, k, v = (t.to("cuda", torch.bfloat16) for t in (q, k, v))
    q_int8, q_scale = quantize_sage_q(q[:, :, :seqlen_q])
    k_int8, v_fp8, k_scale, v_scale = quantize_sage_kv(
        k[:, :, :seqlen_k], v[:, :, :seqlen_k]
    )
    if alloc_q > seqlen_q:
        pad = torch.zeros(
            batch, heads, alloc_q - seqlen_q, 128, dtype=torch.int8, device="cuda"
        )
        q_int8 = torch.cat([q_int8, pad], dim=2)
    if alloc_k > seqlen_k:
        pad = alloc_k - seqlen_k
        k_int8 = torch.cat(
            [
                k_int8,
                torch.zeros(batch, heads, pad, 128, dtype=torch.int8, device="cuda"),
            ],
            2,
        )
        v_fp8 = torch.cat(
            [
                v_fp8,
                torch.zeros(
                    batch, heads, 128, pad, dtype=torch.float8_e4m3fn, device="cuda"
                ),
            ],
            3,
        )
    return (
        q_int8.contiguous(),
        k_int8.contiguous(),
        v_fp8.contiguous(),
        q_scale,
        k_scale,
        v_scale,
    )


def _run(
    q_int8, k_int8, v_fp8, q_scale, k_scale, v_scale, index, nums, *, seqlen_q, seqlen_k
):
    out = torch.zeros(q_int8.shape, dtype=torch.bfloat16, device="cuda")
    sage_block_sparse_attn_sm120(
        q_int8,
        k_int8,
        v_fp8,
        q_scale,
        k_scale,
        v_scale,
        index,
        nums,
        out=out,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        softmax_scale=128**-0.5,
    )
    return out


def _reference(
    q_int8, k_int8, v_fp8, q_scale, k_scale, v_scale, index, nums, *, seqlen_q, seqlen_k
):
    q, k, v = dequantize_sage(
        q_int8,
        k_int8,
        v_fp8,
        q_scale,
        k_scale,
        v_scale,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
    )
    return block_sparse_attention_reference(
        q, k, v, index, nums, softmax_scale=128**-0.5
    )


def _rel_l2(got, want):
    return (got.float() - want).norm() / want.norm().clamp_min(1e-6)


def _cosine(got, want):
    g, w = got.float().flatten(), want.flatten()
    return torch.dot(g, w) / (g.norm() * w.norm()).clamp_min(1e-12)


# The kernel quantises P to FP8 E4M3 (3 mantissa bits) before the PV MMA and
# stores BF16; the fp32 reference does neither, so ~2-3% relative L2 is the
# expected floor (historically 0.037 against unquantised bf16 SDPA).
REL_L2_TOL = 5e-2
COSINE_MIN = 0.999


@requires_sm120
@pytest.mark.parametrize(
    "batch,heads,seqlen_q,seqlen_k",
    [(1, 2, 128, 128), (2, 4, 640, 1024), (1, 7, 1000, 1088)],
)
def test_dense_index_matches_dequantized_reference(batch, heads, seqlen_q, seqlen_k):
    ops = _operands(batch, heads, seqlen_q, seqlen_k)
    index, nums = sage_block_sparse_dense_block_index(
        batch, heads, seqlen_q, seqlen_k, torch.device("cuda")
    )
    out = _run(*ops, index, nums, seqlen_q=seqlen_q, seqlen_k=seqlen_k)
    want = _reference(*ops, index, nums, seqlen_q=seqlen_q, seqlen_k=seqlen_k)
    assert torch.isfinite(out[:, :, :seqlen_q].float()).all()
    assert _rel_l2(out[:, :, :seqlen_q], want) < REL_L2_TOL
    assert _cosine(out[:, :, :seqlen_q], want) > COSINE_MIN


@requires_sm120
def test_sparse_unordered_rows_with_empty_and_partial_counts():
    batch, heads, seqlen_q, seqlen_k = 1, 3, 512, 1024
    ops = _operands(batch, heads, seqlen_q, seqlen_k, seed=1)
    q_blocks, k_blocks = seqlen_q // K_BLOCK, seqlen_k // K_BLOCK
    g = torch.Generator(device="cpu").manual_seed(2)
    capacity = 6
    index = (
        torch.stack(
            [
                torch.randperm(k_blocks, generator=g)[:capacity]
                for _ in range(batch * heads * q_blocks)
            ]
        )
        .view(batch, heads, q_blocks, capacity)
        .to("cuda", torch.int32)
    )
    nums = torch.randint(1, capacity + 1, (batch, heads, q_blocks), generator=g).to(
        "cuda", torch.int32
    )
    nums[0, 0, 0] = (
        0  # a query block that keeps nothing must produce zeros, not garbage
    )
    nums[0, 1, 3] = capacity
    out = _run(*ops, index, nums, seqlen_q=seqlen_q, seqlen_k=seqlen_k)
    want = _reference(*ops, index, nums, seqlen_q=seqlen_q, seqlen_k=seqlen_k)
    live = nums.repeat_interleave(K_BLOCK, dim=-1) > 0
    assert _rel_l2(out[live], want[live]) < REL_L2_TOL
    assert _cosine(out[live], want[live]) > COSINE_MIN
    assert torch.equal(out[~live].float(), torch.zeros_like(want[~live]))


@requires_sm120
def test_live_prefix_of_a_larger_allocation_matches_exact_sizes():
    """Handing in a resident buffer and computing only ``seqlen`` rows is bit-identical."""
    batch, heads, seqlen_q, seqlen_k = 1, 2, 640, 640
    exact = _operands(batch, heads, seqlen_q, seqlen_k, seed=3)
    over = _operands(
        batch, heads, seqlen_q, seqlen_k, seed=3, alloc_q=1024, alloc_k=1024
    )
    index, nums = sage_block_sparse_dense_block_index(
        batch, heads, seqlen_q, seqlen_k, torch.device("cuda")
    )
    out_exact = _run(*exact, index, nums, seqlen_q=seqlen_q, seqlen_k=seqlen_k)
    out_over = torch.full(
        (batch, heads, 1024, 128), 7.0, dtype=torch.bfloat16, device="cuda"
    )
    sage_block_sparse_attn_sm120(
        *over,
        index,
        nums,
        out=out_over,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        softmax_scale=128**-0.5,
    )
    assert torch.equal(out_over[:, :, :seqlen_q], out_exact)
    assert torch.equal(
        out_over[:, :, seqlen_q:], torch.full_like(out_over[:, :, seqlen_q:], 7.0)
    )


@requires_sm120
def test_descriptor_cache_survives_buffer_changes():
    batch, heads, seqlen = 1, 2, 256
    a = _operands(batch, heads, seqlen, seqlen, seed=4)
    b = _operands(batch, heads, seqlen, seqlen, seed=5)
    index, nums = sage_block_sparse_dense_block_index(
        batch, heads, seqlen, seqlen, torch.device("cuda")
    )
    first_a = _run(*a, index, nums, seqlen_q=seqlen, seqlen_k=seqlen)
    first_b = _run(*b, index, nums, seqlen_q=seqlen, seqlen_k=seqlen)
    again_a = _run(*a, index, nums, seqlen_q=seqlen, seqlen_k=seqlen)
    assert torch.equal(first_a, again_a)
    assert not torch.equal(first_a, first_b)


@requires_sm120
def test_rejects_unaligned_seqlen_k_and_undersized_scales():
    ops = _operands(1, 1, 128, 128)
    index, nums = sage_block_sparse_dense_block_index(
        1, 1, 128, 128, torch.device("cuda")
    )
    with pytest.raises(Exception, match="multiple of 64"):
        _run(*ops, index, nums, seqlen_q=128, seqlen_k=100)
    q_int8, k_int8, v_fp8, q_scale, k_scale, v_scale = ops
    with pytest.raises(Exception):
        _run(
            q_int8,
            k_int8,
            v_fp8,
            q_scale,
            k_scale[:, :, :1],
            v_scale,
            index,
            nums,
            seqlen_q=128,
            seqlen_k=128,
        )


@requires_sm120
def test_minimax_h3_shape_is_deterministic():
    """The serving shape (7 local heads, 37888 tokens, 25% density) runs and repeats bit-exactly."""
    batch, heads, seqlen = 1, 7, 37888
    ops = _operands(batch, heads, seqlen, seqlen, seed=6)
    q_blocks = seqlen // K_BLOCK
    keep = q_blocks // 4
    g = torch.Generator(device="cpu").manual_seed(7)
    index = (
        torch.stack(
            [
                torch.randperm(q_blocks, generator=g)[:keep]
                for _ in range(batch * heads * q_blocks)
            ]
        )
        .view(batch, heads, q_blocks, keep)
        .to("cuda", torch.int32)
    )
    nums = torch.full((batch, heads, q_blocks), keep, dtype=torch.int32, device="cuda")
    first = _run(*ops, index, nums, seqlen_q=seqlen, seqlen_k=seqlen)
    second = _run(*ops, index, nums, seqlen_q=seqlen, seqlen_k=seqlen)
    assert torch.isfinite(first.float()).all()
    assert torch.equal(first, second)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
