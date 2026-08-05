"""Bit-exactness tests for the fused indexed AdaLN modulation + RMSNorm kernel.

MiniMax-H3 runs a `quality: lossless` path, so the fused kernel must reproduce
the eager chain (`indexed_gate_bf16_` -> `nn.RMSNorm` -> `indexed_scale_shift_bf16_`)
bit for bit, not merely within a tolerance. Every case below asserts
`torch.equal` against that chain.
"""

import sys

import pytest
import torch
from torch import nn

from sglang.kernels.ops.diffusion.indexed_modulation_norm import (
    indexed_gate_norm_scale_shift,
    indexed_norm_scale_shift,
)
from sglang.kernels.ops.diffusion.triton.indexed_modulation import (
    indexed_gate_bf16_,
    indexed_scale_shift_bf16_,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="1-gpu-large")

_EPS = 1e-6
# MiniMax-H3 hidden size, plus widths that exercise the partial-tail
# (num_vec < 128) and multi-iteration paths of the reduction loop.
_HIDDEN = [5376, 1024, 256]
_ROWS = [1, 7, 129, 1000]


@pytest.fixture(autouse=True)
def cuda_setup():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(0)


def _make_adaln_tables(num_params: int, hidden: int, *, contiguous: bool):
    """Six [P, H] bf16 tables, laid out as in MiniMaxH3AdalnProj.split_output."""
    if contiguous:
        return tuple(
            torch.randn(num_params, hidden, device="cuda", dtype=torch.bfloat16) * 0.5
            for _ in range(6)
        )
    packed = (
        torch.randn(num_params, 6 * hidden, device="cuda", dtype=torch.bfloat16) * 0.5
    )
    return tuple(packed.chunk(6, dim=-1))


def _make_norm(hidden: int) -> nn.RMSNorm:
    norm = nn.RMSNorm(hidden, eps=_EPS, dtype=torch.bfloat16).cuda()
    with torch.no_grad():
        norm.weight.normal_(mean=1.0, std=0.2)
    return norm


def _eager_norm_scale_shift(x, norm, scale, shift, indices):
    return indexed_scale_shift_bf16_(norm(x), shift, scale, indices)


def _eager_gate_norm_scale_shift(residual, update, gate, norm, scale, shift, indices):
    res_out = indexed_gate_bf16_(residual, gate, update, indices)
    return res_out, indexed_scale_shift_bf16_(norm(res_out), shift, scale, indices)


@pytest.mark.parametrize("hidden", _HIDDEN)
@pytest.mark.parametrize("rows", _ROWS)
@pytest.mark.parametrize("contiguous_tables", [True, False])
def test_norm_scale_shift_bit_exact(hidden, rows, contiguous_tables):
    num_params = 3
    x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    _, _, _, shift, scale, _ = _make_adaln_tables(
        num_params, hidden, contiguous=contiguous_tables
    )
    indices = torch.randint(0, num_params, (rows,), device="cuda", dtype=torch.int64)
    norm = _make_norm(hidden)

    expected = _eager_norm_scale_shift(x.clone(), norm, scale, shift, indices)
    actual = indexed_norm_scale_shift(x, norm.weight, scale, shift, indices, _EPS)
    assert torch.equal(actual, expected), (actual - expected).abs().max().item()


@pytest.mark.parametrize("hidden", _HIDDEN)
@pytest.mark.parametrize("rows", _ROWS)
@pytest.mark.parametrize("contiguous_tables", [True, False])
def test_gate_norm_scale_shift_bit_exact(hidden, rows, contiguous_tables):
    num_params = 3
    residual = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    update = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    _, _, gate, shift, scale, _ = _make_adaln_tables(
        num_params, hidden, contiguous=contiguous_tables
    )
    indices = torch.randint(0, num_params, (rows,), device="cuda", dtype=torch.int64)
    norm = _make_norm(hidden)

    expected_res, expected_y = _eager_gate_norm_scale_shift(
        residual.clone(), update, gate, norm, scale, shift, indices
    )
    actual_y = indexed_gate_norm_scale_shift(
        residual, update, gate, norm.weight, scale, shift, indices, _EPS
    )
    # The gated residual lands in `residual`, mirroring the in-place eager gate.
    assert torch.equal(residual, expected_res)
    assert torch.equal(actual_y, expected_y), (actual_y - expected_y).abs().max().item()


def test_production_shape_bit_exact():
    """MiniMax-H3 T2VA 5s per-rank shape with the real AdaLN chunk layout."""
    rows, hidden, num_params = 9456, 5376, 3
    residual = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    update = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    _, _, gate, shift, scale, _ = _make_adaln_tables(
        num_params, hidden, contiguous=False
    )
    indices = torch.randint(0, num_params, (rows,), device="cuda", dtype=torch.int64)
    norm = _make_norm(hidden)

    expected_res, expected_y = _eager_gate_norm_scale_shift(
        residual.clone(), update, gate, norm, scale, shift, indices
    )
    actual_y = indexed_gate_norm_scale_shift(
        residual, update, gate, norm.weight, scale, shift, indices, _EPS
    )
    assert torch.equal(residual, expected_res)
    assert torch.equal(actual_y, expected_y)


def test_custom_op_torch_compile_fullgraph():
    """The ops must survive Dynamo: the JIT module lookup is not traceable."""
    rows, hidden, num_params = 64, 1024, 2
    residual = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    update = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    _, _, gate, shift, scale, _ = _make_adaln_tables(
        num_params, hidden, contiguous=False
    )
    indices = torch.randint(0, num_params, (rows,), device="cuda", dtype=torch.int64)
    norm = _make_norm(hidden)

    expected_res, expected_y = _eager_gate_norm_scale_shift(
        residual.clone(), update, gate, norm, scale, shift, indices
    )
    compiled = torch.compile(indexed_gate_norm_scale_shift, fullgraph=True)
    with torch.no_grad():  # diffusion inference never records autograd
        actual_y = compiled(
            residual, update, gate, norm.weight, scale, shift, indices, _EPS
        )
    assert torch.equal(residual, expected_res)
    assert torch.equal(actual_y, expected_y)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
