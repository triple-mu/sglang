"""Bit-exactness tests for the CUDA indexed AdaLN modulation kernels.

MiniMax-H3 runs a ``quality: lossless`` path, so the CUDA kernels must match
the Triton reference bit for bit -- every assertion here is ``torch.equal``
against the Triton kernel, never against a re-derived formula.
"""

import sys

import pytest
import torch

from sglang.kernels.ops.diffusion.indexed_modulation import (
    _jit_indexed_modulation_module,
)
from sglang.kernels.ops.diffusion.triton.indexed_modulation import (
    indexed_gate_bf16_ as triton_indexed_gate_bf16_,
)
from sglang.kernels.ops.diffusion.triton.indexed_modulation import (
    indexed_scale_shift_bf16_ as triton_indexed_scale_shift_bf16_,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


# (rows, hidden): production shape first, then non-aligned rows, a hidden that
# is not a multiple of 256, and a hidden that is not a multiple of 8 (which
# takes the scalar fallback inside the CUDA kernel).
SHAPES = [
    (9456, 5376),
    (1, 5376),
    (7, 1024),
    (4095, 3072),
    (129, 4096),
    (333, 1023),
]


@pytest.fixture(autouse=True)
def cuda_setup():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(0)


def _make_params(num_params: int, hidden: int, layout: str, num_tensors: int):
    """Build the modulation tables; ``chunk`` mirrors the production AdaLN
    layout where each table is a ``chunk(6, dim=-1)`` view of one linear
    output, i.e. row stride 6*hidden and a non-zero storage offset."""
    if layout == "contig":
        return [
            torch.randn(num_params, hidden, device="cuda", dtype=torch.bfloat16)
            for _ in range(num_tensors)
        ]
    packed = torch.randn(num_params, 6 * hidden, device="cuda", dtype=torch.bfloat16)
    return list(packed.chunk(6, dim=-1))[:num_tensors]


def _make_indices(rows: int, num_params: int) -> torch.Tensor:
    index = torch.randint(0, num_params, (rows,), device="cuda", dtype=torch.int64)
    index[: min(rows, num_params)] = torch.arange(
        min(rows, num_params), device="cuda", dtype=torch.int64
    )
    return index[torch.randperm(rows, device="cuda")]


def _wide_range_like(x: torch.Tensor) -> torch.Tensor:
    """Random values spanning ~2^-20 .. 2^20 to exercise the rounding path."""
    exponent = torch.randint(-20, 21, x.shape, device=x.device, dtype=torch.float32)
    return (torch.randn_like(x, dtype=torch.float32) * exponent.exp2()).to(x.dtype)


@pytest.mark.parametrize("rows,hidden", SHAPES)
@pytest.mark.parametrize("modality_steps", [1, 2, 3])
@pytest.mark.parametrize("layout", ["contig", "chunk"])
def test_indexed_scale_shift_bit_exact(rows, hidden, modality_steps, layout):
    num_params = 3 * modality_steps
    x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    shift, scale = _make_params(num_params, hidden, layout, 2)
    indices = _make_indices(rows, num_params)

    reference = triton_indexed_scale_shift_bf16_(x.clone(), shift, scale, indices)
    out = x.clone()
    _jit_indexed_modulation_module().indexed_scale_shift(out, shift, scale, indices)

    assert torch.equal(out, reference)


@pytest.mark.parametrize("rows,hidden", SHAPES)
@pytest.mark.parametrize("modality_steps", [1, 2, 3])
@pytest.mark.parametrize("layout", ["contig", "chunk"])
def test_indexed_gate_bit_exact(rows, hidden, modality_steps, layout):
    num_params = 3 * modality_steps
    x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    other = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    (gate,) = _make_params(num_params, hidden, layout, 1)
    indices = _make_indices(rows, num_params)

    reference = triton_indexed_gate_bf16_(x.clone(), gate, other, indices)
    out = x.clone()
    _jit_indexed_modulation_module().indexed_gate(out, gate, other, indices)

    assert torch.equal(out, reference)


@pytest.mark.parametrize("rows,hidden", [(1024, 5376), (333, 1023)])
def test_wide_dynamic_range_bit_exact(rows, hidden):
    num_params = 9
    x = _wide_range_like(torch.empty(rows, hidden, device="cuda", dtype=torch.bfloat16))
    other = _wide_range_like(x)
    shift, scale, gate = (
        _wide_range_like(
            torch.empty(num_params, hidden, device="cuda", dtype=torch.bfloat16)
        )
        for _ in range(3)
    )
    indices = _make_indices(rows, num_params)
    module = _jit_indexed_modulation_module()

    reference = triton_indexed_scale_shift_bf16_(x.clone(), shift, scale, indices)
    out = x.clone()
    module.indexed_scale_shift(out, shift, scale, indices)
    assert torch.equal(out, reference)

    reference = triton_indexed_gate_bf16_(x.clone(), gate, other, indices)
    out = x.clone()
    module.indexed_gate(out, gate, other, indices)
    assert torch.equal(out, reference)


def _finite_bf16_patterns(count: int, stride: int) -> torch.Tensor:
    """`count` BF16 bit patterns spread over the whole 16-bit space, with the
    non-finite ones (exponent 0xFF) zeroed."""
    bits = (
        torch.arange(count, device="cuda", dtype=torch.int32) * stride % (1 << 16)
    ).to(torch.int16)
    bits = torch.where(bits & 0x7F80 == 0x7F80, torch.zeros_like(bits), bits)
    return bits.view(torch.bfloat16)


@pytest.mark.parametrize("op", ["scale_shift", "gate"])
def test_bit_pattern_sweep_bit_exact(op):
    """Every finite BF16 value of the table operand against 512 BF16 values of
    the swept operand -- the rounding boundary is where the CUDA kernel could
    diverge from Triton, so it gets a systematic sweep, not just randoms."""
    rows, hidden = 512, 1 << 16
    table = _finite_bf16_patterns(hidden, 1)[None, :]
    swept = _finite_bf16_patterns(rows, 127)[:, None].expand(rows, hidden).contiguous()
    zeros = torch.zeros(1, hidden, device="cuda", dtype=torch.bfloat16)
    indices = torch.zeros(rows, device="cuda", dtype=torch.int64)
    module = _jit_indexed_modulation_module()

    if op == "scale_shift":
        # sweep (x, scale) with shift == 0, then (x, shift) with scale == 0
        cases = [(zeros, table), (table, zeros)]
        for shift, scale in cases:
            reference = triton_indexed_scale_shift_bf16_(
                swept.clone(), shift, scale, indices
            )
            out = swept.clone()
            module.indexed_scale_shift(out, shift, scale, indices)
            assert torch.equal(out, reference)
    else:
        # sweep (other, gate) with x == 0, then (x, other) with gate == 1
        ones = torch.ones(1, hidden, device="cuda", dtype=torch.bfloat16)
        reference = triton_indexed_gate_bf16_(
            torch.zeros_like(swept), table, swept, indices
        )
        out = torch.zeros_like(swept)
        module.indexed_gate(out, table, swept, indices)
        assert torch.equal(out, reference)

        other = table.expand(rows, hidden).contiguous()
        reference = triton_indexed_gate_bf16_(swept.clone(), ones, other, indices)
        out = swept.clone()
        module.indexed_gate(out, ones, other, indices)
        assert torch.equal(out, reference)


def test_dispatch_defaults_to_cuda():
    from sglang.kernels.ops.diffusion.indexed_modulation import (
        indexed_gate_bf16_,
        indexed_scale_shift_bf16_,
    )

    rows, hidden, num_params = 300, 5376, 3
    x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    other = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    shift, scale, gate = _make_params(num_params, hidden, "chunk", 3)
    indices = _make_indices(rows, num_params)

    out = x.clone()
    assert indexed_scale_shift_bf16_(out, shift, scale, indices) is out
    assert torch.equal(
        out, triton_indexed_scale_shift_bf16_(x.clone(), shift, scale, indices)
    )

    out = x.clone()
    assert indexed_gate_bf16_(out, gate, other, indices) is out
    assert torch.equal(out, triton_indexed_gate_bf16_(x.clone(), gate, other, indices))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
