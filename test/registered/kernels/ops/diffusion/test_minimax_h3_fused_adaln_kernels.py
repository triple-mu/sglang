# SPDX-License-Identifier: Apache-2.0
"""The MiniMax-H3 fused adaLN kernels against the eager chains they replace.

Bit-exactness is the contract: `rmsnorm_indexed_scale_shift` equals
`indexed_scale_shift_bf16_(nn.RMSNorm(x))`, the gated variant equals
`indexed_gate_bf16` followed by that chain, and `indexed_scale_shift` equals the
Triton kernel (its fp32 output equals the bf16 result widened). Modulation rows
come from a `[groups, 6 * hidden]` slab like the model's adaLN tensors.
"""

import sys

import pytest
import torch
import torch.nn as nn

from sglang.kernels.ops.diffusion import (
    can_use_gate_residual_rmsnorm_indexed_scale_shift,
    can_use_indexed_scale_shift,
    can_use_rmsnorm_indexed_scale_shift,
    gate_residual_rmsnorm_indexed_scale_shift_,
    gate_residual_rmsnorm_indexed_scale_shift_fp8_,
    indexed_gate_bf16,
    indexed_scale_shift,
    indexed_scale_shift_bf16_,
    rmsnorm_indexed_scale_shift,
    rmsnorm_indexed_scale_shift_fp8,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA GPU"
)

HIDDEN = 5376
EPS = 1e-5


def _case(rows: int, groups: int, seed: int, index_dtype=torch.int64):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = (torch.randn(rows, HIDDEN, generator=g, device="cuda") * 2.0).to(torch.bfloat16)
    x[::9] *= 40.0  # rows with a large rms
    slab = (torch.randn(groups, 6 * HIDDEN, generator=g, device="cuda") * 0.4).to(
        torch.bfloat16
    )
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = slab.chunk(
        6, dim=-1
    )
    indices = torch.randint(0, groups, (rows,), generator=g, device="cuda").to(
        index_dtype
    )
    norm = nn.RMSNorm(HIDDEN, eps=EPS, dtype=torch.bfloat16, device="cuda")
    with torch.no_grad():
        norm.weight.copy_(1.0 + 0.2 * torch.randn(HIDDEN, generator=g, device="cuda"))
    return (
        x,
        (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp),
        indices,
        norm,
    )


@pytest.mark.parametrize("rows,groups", [(4736, 8), (1000, 3), (64, 1)])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_plan_a_is_bitwise_the_eager_chain(rows, groups, index_dtype):
    x, (shift, scale, *_), indices, norm = _case(rows, groups, 1, index_dtype)
    assert scale.stride(0) == 6 * HIDDEN  # the slab view, not a copy
    assert can_use_rmsnorm_indexed_scale_shift(x, norm.weight, scale, shift, indices)
    got = rmsnorm_indexed_scale_shift(x, norm.weight, scale, shift, indices, eps=EPS)
    want = indexed_scale_shift_bf16_(norm(x), shift, scale, indices)
    assert torch.equal(got, want)


@pytest.mark.parametrize("rows,groups", [(4736, 8), (1000, 3)])
def test_plan_b_is_bitwise_gate_then_the_eager_chain(rows, groups):
    x, (shift_msa, scale_msa, gate, shift, scale, _), indices, norm = _case(
        rows, groups, 2
    )
    update = rmsnorm_indexed_scale_shift(
        x, norm.weight, scale_msa, shift_msa, indices, eps=EPS
    )
    assert can_use_gate_residual_rmsnorm_indexed_scale_shift(
        x, update, gate, norm.weight, scale, shift, indices
    )
    want_residual = indexed_gate_bf16(x, gate, update, indices)
    want = indexed_scale_shift_bf16_(norm(want_residual), shift, scale, indices)
    residual = x.clone()
    got, got_residual = gate_residual_rmsnorm_indexed_scale_shift_(
        residual, update, gate, norm.weight, scale, shift, indices, eps=EPS
    )
    assert got_residual is residual
    assert torch.equal(got_residual, want_residual)
    assert torch.equal(got, want)


def test_plan_a_rejects_a_copy_free_but_misaligned_slab():
    x, (shift, scale, *_), indices, norm = _case(128, 4, 3)
    odd = scale[:, 1:]  # 2-byte offset breaks the 16-byte alignment
    assert not can_use_rmsnorm_indexed_scale_shift(
        x, norm.weight, odd, shift[:, 1:], indices
    )
    assert not can_use_rmsnorm_indexed_scale_shift(
        x.float(), norm.weight, scale, shift, indices
    )


def test_indexed_scale_shift_matches_triton_and_widens_exactly():
    x, (shift, scale, *_), indices, _ = _case(2000, 5, 4)
    assert can_use_indexed_scale_shift(x, shift, scale, indices)
    want = indexed_scale_shift_bf16_(x.clone(), shift, scale, indices)
    got_bf16 = indexed_scale_shift(
        x.clone(), shift, scale, indices, out_dtype=torch.bfloat16
    )
    got_fp32 = indexed_scale_shift(x, shift, scale, indices, out_dtype=torch.float32)
    assert torch.equal(got_bf16, want)
    assert got_fp32.dtype is torch.float32
    assert torch.equal(got_fp32, want.float())
    with pytest.raises(ValueError):
        indexed_scale_shift(x, shift, scale, indices, out_dtype=torch.float16)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


def _per_token_fp8(rows_bf16: torch.Tensor):
    """Per-token fp8 as the kernel defines it: scale = amax / 448 and q = sat(x / scale),
    both correctly rounded (tensor denominators; torch's scalar division multiplies by a
    reciprocal instead)."""
    x = rows_bf16.float()
    amax = x.abs().amax(dim=-1, keepdim=True)
    scale = amax / torch.full_like(amax, 448.0)
    inv = torch.where(scale == 0, torch.zeros_like(scale), 1.0 / scale)
    q = (x * inv).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return q, scale


@pytest.mark.parametrize("rows,groups", [(4736, 8), (1000, 3)])
def test_fp8_plan_a_quantises_the_bf16_chain_per_token(rows, groups):
    x, params, indices, norm = _case(rows, groups, 21)
    shift_msa, scale_msa = params[0], params[1]
    x[3] = 0
    shift_msa[indices[3]] = 0  # row 3 modulates to all zeros: zero scale, zero codes
    want_rows = rmsnorm_indexed_scale_shift(
        x, norm.weight, scale_msa, shift_msa, indices, eps=EPS
    )
    want_q, want_s = _per_token_fp8(want_rows)
    q, s = rmsnorm_indexed_scale_shift_fp8(
        x, norm.weight, scale_msa, shift_msa, indices, eps=EPS
    )
    assert q.dtype is torch.float8_e4m3fn and tuple(s.shape) == (rows, 1)
    assert torch.equal(s, want_s)
    assert torch.equal(q.view(torch.uint8), want_q.view(torch.uint8))
    assert s[3].item() == 0 and not q[3].view(torch.uint8).any()


@pytest.mark.parametrize("rows,groups", [(4736, 8), (1000, 3)])
def test_fp8_plan_b_quantises_the_gated_chain_per_token(rows, groups):
    x, params, indices, norm = _case(rows, groups, 22)
    shift_msa, scale_msa, gate_msa = params[0], params[1], params[2]
    g = torch.Generator(device="cuda").manual_seed(23)
    update = (torch.randn(rows, HIDDEN, generator=g, device="cuda") * 0.7).to(
        torch.bfloat16
    )
    want_rows, want_residual = gate_residual_rmsnorm_indexed_scale_shift_(
        x.clone(), update, gate_msa, norm.weight, scale_msa, shift_msa, indices, eps=EPS
    )
    want_q, want_s = _per_token_fp8(want_rows)
    (q, s), residual = gate_residual_rmsnorm_indexed_scale_shift_fp8_(
        x.clone(), update, gate_msa, norm.weight, scale_msa, shift_msa, indices, eps=EPS
    )
    assert torch.equal(residual, want_residual)
    assert torch.equal(s, want_s)
    assert torch.equal(q.view(torch.uint8), want_q.view(torch.uint8))


@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0),
    reason="the GEMM quantiser is built with fast math on SM90 (per_token_quant_fp8.py)",
)
def test_fp8_plan_a_matches_the_gemm_quantiser():
    """The pair is what the fp8 GEMM would have computed from the bf16 rows itself."""
    from sglang.kernels.ops.quantization import sgl_per_token_quant_fp8

    x, params, indices, norm = _case(4736, 8, 24)
    shift_msa, scale_msa = params[0], params[1]
    rows = rmsnorm_indexed_scale_shift(
        x, norm.weight, scale_msa, shift_msa, indices, eps=EPS
    )
    want_q = torch.empty_like(rows, dtype=torch.float8_e4m3fn)
    want_s = torch.empty((rows.shape[0], 1), dtype=torch.float32, device="cuda")
    sgl_per_token_quant_fp8(rows, want_q, want_s)
    q, s = rmsnorm_indexed_scale_shift_fp8(
        x, norm.weight, scale_msa, shift_msa, indices, eps=EPS
    )
    assert torch.equal(s, want_s)
    assert torch.equal(q.view(torch.uint8), want_q.view(torch.uint8))
