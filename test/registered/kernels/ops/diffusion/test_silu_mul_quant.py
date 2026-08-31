"""Fused packed SwiGLU + per-token FP8 quant vs the separated fp8 chain.

The kernel's contract is *lossless relative to the separated chain*: the fp8
payload and fp32 scale must match ``sgl_per_token_quant_fp8(F.silu(g) * u)``
on every byte, so ``torch.equal`` on a ``uint8`` view -- not ``assert_close``
-- is the assertion throughout.  The reference quant kernel's arithmetic is
dispatch-dependent (its warp-per-token variant guards a zero scale, its
CTA-per-token variant does not, and the SM90 build is fast-math), so the
cases sweep both dispatch regimes and the amax=0 / denormal corners where
those variants differ.
"""

import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.activation.activation import (
    silu_and_mul_with_activation_rounding,
)
from sglang.kernels.ops.diffusion import (
    can_use_fused_silu_mul_per_token_quant_fp8,
    fused_silu_mul_per_token_quant_fp8,
)

# Deep imports pin the C++ JIT backend against its Triton fallback directly
# (allowlisted in test_import_surface.py).
from sglang.kernels.ops.diffusion.activation import (
    silu_mul_quant_jit as _silu_mul_quant_jit,
)
from sglang.kernels.ops.diffusion.activation import (
    silu_mul_quant_triton as _silu_mul_quant_triton,
)
from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_quant_fp8
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

DEVICE = "cuda"
# MiniMax-H3 production fc1 output width: [gate | up] halves of 14336.
HIDDEN = 14336


def _warp_dispatch_num_tokens() -> int:
    # The reference launcher switches to its warp-per-token variant at
    # sm_count * 2 * 8 rows; below that the CTA variant (whose zero-scale
    # reciprocal is unguarded) runs.
    return torch.cuda.get_device_properties(0).multi_processor_count * 16


def _separated_chain(x: torch.Tensor):
    """The eager fp8 path: ``_silu_mul`` under quant + standalone quant."""
    gate, up = x.chunk(2, dim=-1)
    return sglang_per_token_quant_fp8(F.silu(gate) * up)


def _assert_bitwise_equal(actual, expected):
    assert torch.equal(actual[0].view(torch.uint8), expected[0].view(torch.uint8))
    assert torch.equal(actual[1].view(torch.uint8), expected[1].view(torch.uint8))


def _adversarial_fill(x: torch.Tensor) -> torch.Tensor:
    """Rows whose *activation* exercises every scale regime of the reference."""
    hidden = x.shape[-1] // 2
    x[::7] = 0  # act == 0 -> amax == 0
    # act ~ silu(1) * up: bf16-denormal activations (fast-math FTZ in amax).
    denormal = torch.rand(hidden, dtype=torch.float32, device=DEVICE) * 9.0e-41
    x[1::11, :hidden] = 1.0
    x[1::11, hidden:] = denormal.to(torch.bfloat16)
    # Normal act amax whose scale (amax / 448) underflows to a flushed zero.
    tiny = torch.rand(hidden, dtype=torch.float32, device=DEVICE) * 1e-38
    x[2::13, :hidden] = 1.0
    x[2::13, hidden:] = tiny.to(torch.bfloat16)
    huge = torch.randn(hidden, dtype=torch.float32, device=DEVICE) * 3e38
    x[3::17, hidden:] = huge.to(torch.bfloat16)
    return x


@pytest.mark.parametrize("rows", [17, 1797, 18944, 20992])
def test_matches_separated_chain_across_dispatch(rows):
    torch.manual_seed(0)
    x = torch.randn(rows, 2 * HIDDEN, dtype=torch.bfloat16, device=DEVICE)
    x = _adversarial_fill(x)
    assert can_use_fused_silu_mul_per_token_quant_fp8(x)
    _assert_bitwise_equal(fused_silu_mul_per_token_quant_fp8(x), _separated_chain(x))


@pytest.mark.parametrize("dispatch", ["cta", "warp"])
def test_all_zero_rows_match_dispatch_semantics(dispatch):
    # A zero row's payload depends on the reference dispatch variant (the CTA
    # variant multiplies by an unguarded 1/0); the fused kernel must mirror it.
    rows = 3 if dispatch == "cta" else _warp_dispatch_num_tokens()
    x = torch.zeros(rows, 2 * HIDDEN, dtype=torch.bfloat16, device=DEVICE)
    _assert_bitwise_equal(fused_silu_mul_per_token_quant_fp8(x), _separated_chain(x))


@pytest.mark.parametrize("dispatch", ["cta", "warp"])
def test_denormal_rows_match_dispatch_semantics(dispatch):
    # bf16-denormal activations flush out of the fast-math amax, degrading the
    # row to the zero-scale path even though its payload bits are nonzero.
    torch.manual_seed(1)
    rows = 5 if dispatch == "cta" else _warp_dispatch_num_tokens()
    x = torch.ones(rows, 2 * HIDDEN, dtype=torch.bfloat16, device=DEVICE)
    denormal = torch.rand(rows, HIDDEN, dtype=torch.float32, device=DEVICE)
    x[:, HIDDEN:] = (denormal * 9.0e-41).to(torch.bfloat16)  # act ~ silu(1) * up
    _assert_bitwise_equal(fused_silu_mul_per_token_quant_fp8(x), _separated_chain(x))


def test_matches_act_and_mul_chain():
    # Bitwise vs the fused bf16 activation + quant chain too: the kernel
    # restores the ``reuse_fc1_activation`` semantics fp8 serving disables.
    torch.manual_seed(2)
    x = torch.randn(1797, 2 * HIDDEN, dtype=torch.bfloat16, device=DEVICE)
    act = silu_and_mul_with_activation_rounding(x)
    expected = sglang_per_token_quant_fp8(act)
    _assert_bitwise_equal(fused_silu_mul_per_token_quant_fp8(x), expected)


def test_accepts_strided_rows():
    # A wider projection slice hands the kernel rows with stride > width.
    torch.manual_seed(3)
    wide = torch.randn(64, 3 * 1024, dtype=torch.bfloat16, device=DEVICE)
    x = wide[:, : 2 * 1024]
    assert can_use_fused_silu_mul_per_token_quant_fp8(x)
    _assert_bitwise_equal(
        fused_silu_mul_per_token_quant_fp8(x), _separated_chain(x.contiguous())
    )


def test_rejects_unsupported_inputs():
    good = torch.randn(4, 512, dtype=torch.bfloat16, device=DEVICE)
    assert can_use_fused_silu_mul_per_token_quant_fp8(good)
    assert not can_use_fused_silu_mul_per_token_quant_fp8(good.half())
    assert not can_use_fused_silu_mul_per_token_quant_fp8(good[:, :-1])  # odd width
    assert not can_use_fused_silu_mul_per_token_quant_fp8(good.t())  # column strides
    with pytest.raises(RuntimeError, match="unsupported"):
        fused_silu_mul_per_token_quant_fp8(good.half())


# ---------------------------------------------------------------------------
# C++ JIT backend vs the Triton fallback, and the dispatch-order gate.  The
# public symbol dispatches C++ -> Triton; these pin the backends directly.
# ---------------------------------------------------------------------------


def _skip_unless_cpp_backend(x: torch.Tensor) -> None:
    if not _silu_mul_quant_jit.can_use_silu_mul_quant_cuda(x):
        pytest.skip("JIT C++ backend unavailable (needs SM90 CUDA)")


@pytest.mark.parametrize("rows", [17, 20992])
def test_cpp_matches_triton_bitwise(rows):
    """The C++ port replicates the Triton replica's exact arithmetic bitwise
    across both reference dispatch regimes (rows above/below the warp-dispatch
    threshold flip the zero-scale guard) and the adversarial scale corners."""
    torch.manual_seed(4)
    x = _adversarial_fill(
        torch.randn(rows, 2 * HIDDEN, dtype=torch.bfloat16, device=DEVICE)
    )
    _skip_unless_cpp_backend(x)
    _assert_bitwise_equal(
        _silu_mul_quant_jit.silu_mul_quant_cuda(x),
        _silu_mul_quant_triton.fused_silu_mul_per_token_quant_fp8(x),
    )


def test_silu_mul_quant_dispatch_order(monkeypatch):
    """Supported input runs the C++ backend; unsupported falls back to Triton."""
    torch.manual_seed(5)
    x = torch.randn(64, 2 * HIDDEN, dtype=torch.bfloat16, device=DEVICE)
    _skip_unless_cpp_backend(x)

    fallback_calls: list[str] = []
    real_triton = _silu_mul_quant_jit.triton_fused_silu_mul_per_token_quant_fp8

    def fallback_stub(value):
        fallback_calls.append("triton")
        return real_triton(value)

    monkeypatch.setattr(
        _silu_mul_quant_jit,
        "triton_fused_silu_mul_per_token_quant_fp8",
        fallback_stub,
    )
    _assert_bitwise_equal(fused_silu_mul_per_token_quant_fp8(x), _separated_chain(x))
    assert not fallback_calls, "supported input must not reach the Triton fallback"

    # hidden % 8 != 0 is outside the C++ gate; the call must route to Triton.
    narrow = torch.randn(64, 2 * 12, dtype=torch.bfloat16, device=DEVICE)
    assert can_use_fused_silu_mul_per_token_quant_fp8(narrow)
    assert not _silu_mul_quant_jit.can_use_silu_mul_quant_cuda(narrow)
    _assert_bitwise_equal(
        fused_silu_mul_per_token_quant_fp8(narrow), _separated_chain(narrow)
    )
    assert fallback_calls == ["triton"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
