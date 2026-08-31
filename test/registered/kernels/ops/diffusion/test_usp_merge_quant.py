"""Fused USP output head merge + per-token FP8 quant vs the separated chain.

The merge is pure data movement, so the contract is *lossless relative to the
separated chain*: payload and scale must match
``sgl_per_token_quant_fp8(usp_merge_heads(x).reshape(tokens, -1))`` on every
byte -- ``torch.equal`` on a ``uint8`` view throughout.  The reference quant
arithmetic is dispatch-dependent (zero-scale handling differs between its
warp- and CTA-per-token variants), so the cases sweep both regimes and the
amax=0 / denormal corners.
"""

import sys

import pytest
import torch

from sglang.kernels.ops.diffusion import (
    can_use_merge_two_sources_per_token_quant_fp8,
    can_use_usp_merge_heads_per_token_quant_fp8,
    merge_two_sources_per_token_quant_fp8,
    usp_merge_heads,
    usp_merge_heads_per_token_quant_fp8,
)

# Deep imports on purpose (allowlisted in test_import_surface): the
# backend-equality and dispatch-order cases pin one backend each.
from sglang.kernels.ops.diffusion.layout import (
    usp_merge_quant_jit,
    usp_merge_quant_triton,
)
from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_quant_fp8
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

DEVICE = "cuda"
# MiniMax-H3 production output merge: ulysses=2, 28 local heads, head_dim 128.
WORLD, H_LOCAL, HEAD_DIM = 2, 28, 128


def _warp_dispatch_num_tokens() -> int:
    # The reference launcher switches to its warp-per-token variant at
    # sm_count * 2 * 8 rows; below that the CTA variant (whose zero-scale
    # reciprocal is unguarded) runs.
    return torch.cuda.get_device_properties(0).multi_processor_count * 16


def _separated_chain(x: torch.Tensor):
    """The merge copy + standalone quant it replaces."""
    tokens = x.shape[1] * x.shape[2]
    return sglang_per_token_quant_fp8(usp_merge_heads(x).reshape(tokens, -1))


def _assert_bitwise_equal(actual, expected):
    assert torch.equal(actual[0].view(torch.uint8), expected[0].view(torch.uint8))
    assert torch.equal(actual[1].view(torch.uint8), expected[1].view(torch.uint8))


def _adversarial_fill(x: torch.Tensor) -> torch.Tensor:
    """Token rows exercising every scale regime of the reference arithmetic.

    Fills along the seq dim so every ``w`` shard of a chosen token gets the
    pattern (a token row spans all shards).
    """
    _, _, _, h_local, head_dim = x.shape
    shard = (h_local, head_dim)
    x[:, ::7] = 0  # amax == 0
    denormal = torch.rand(shard, dtype=torch.float32, device=DEVICE) * 9.0e-41
    x[:, 1::11] = denormal.to(torch.bfloat16)  # bf16 denormals (fast-math FTZ)
    tiny = torch.rand(shard, dtype=torch.float32, device=DEVICE) * 1e-38
    x[:, 2::13] = tiny.to(torch.bfloat16)  # normal amax whose scale underflows
    huge = torch.randn(shard, dtype=torch.float32, device=DEVICE) * 3e38
    x[:, 3::17] = huge.to(torch.bfloat16)
    return x


@pytest.mark.parametrize("seq", [17, 1797, 18944, 20992])
def test_matches_separated_chain_across_dispatch(seq):
    torch.manual_seed(0)
    x = torch.randn(
        WORLD, seq, 1, H_LOCAL, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE
    )
    x = _adversarial_fill(x)
    assert can_use_usp_merge_heads_per_token_quant_fp8(x)
    _assert_bitwise_equal(usp_merge_heads_per_token_quant_fp8(x), _separated_chain(x))


def test_generic_world_and_batch():
    # The NCCL path is not 2-rank-specific; row order must stay b * S + s.
    torch.manual_seed(1)
    x = torch.randn(4, 33, 2, 7, 64, dtype=torch.bfloat16, device=DEVICE)
    _assert_bitwise_equal(usp_merge_heads_per_token_quant_fp8(x), _separated_chain(x))


@pytest.mark.parametrize("dispatch", ["cta", "warp"])
def test_all_zero_rows_match_dispatch_semantics(dispatch):
    # A zero row's payload depends on the reference dispatch variant (the CTA
    # variant multiplies by an unguarded 1/0); the fused kernel must mirror it.
    seq = 3 if dispatch == "cta" else _warp_dispatch_num_tokens()
    x = torch.zeros(
        WORLD, seq, 1, H_LOCAL, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE
    )
    _assert_bitwise_equal(usp_merge_heads_per_token_quant_fp8(x), _separated_chain(x))


@pytest.mark.parametrize("dispatch", ["cta", "warp"])
def test_denormal_rows_match_dispatch_semantics(dispatch):
    # bf16-denormal values flush out of the fast-math amax, degrading the row
    # to the zero-scale path even though its payload bits are nonzero.
    torch.manual_seed(2)
    seq = 5 if dispatch == "cta" else _warp_dispatch_num_tokens()
    shape = (WORLD, seq, 1, H_LOCAL, HEAD_DIM)
    denormal = torch.rand(shape, dtype=torch.float32, device=DEVICE)
    x = (denormal * 9.0e-41).to(torch.bfloat16)
    _assert_bitwise_equal(usp_merge_heads_per_token_quant_fp8(x), _separated_chain(x))


def test_two_source_form_matches_cat_then_quant():
    # The 2-rank IPC output merge in output-column order: first fills columns
    # [0, C), second [C, 2C); amax must span both halves.
    torch.manual_seed(3)
    tokens, inner = 1797, H_LOCAL * HEAD_DIM
    first = torch.randn(tokens, inner, dtype=torch.bfloat16, device=DEVICE)
    second = torch.randn(tokens, inner, dtype=torch.bfloat16, device=DEVICE)
    first[::7] = 0
    second[3::7] = 0  # zero half with a nonzero sibling: amax is row-global
    assert can_use_merge_two_sources_per_token_quant_fp8(first, second)
    expected = sglang_per_token_quant_fp8(torch.cat((first, second), dim=1))
    _assert_bitwise_equal(
        merge_two_sources_per_token_quant_fp8(first, second), expected
    )


def test_two_source_form_accepts_strided_rows():
    # The IPC local half is a row slice of the FA3 output, not a fresh tensor.
    torch.manual_seed(4)
    tokens, inner = 64, 512
    backing = torch.randn(2 * tokens, inner, dtype=torch.bfloat16, device=DEVICE)
    first = backing[tokens:]  # offset base pointer
    second = torch.randn(tokens, 2 * inner, dtype=torch.bfloat16, device=DEVICE)[
        :, :inner
    ]  # row stride > width
    assert can_use_merge_two_sources_per_token_quant_fp8(first, second)
    expected = sglang_per_token_quant_fp8(
        torch.cat((first, second), dim=1).contiguous()
    )
    _assert_bitwise_equal(
        merge_two_sources_per_token_quant_fp8(first, second), expected
    )


def _ipc_staging_pair(tokens: int, inner: int):
    """Production 2-rank IPC operands: differing row strides, offset base."""
    first = torch.randn(tokens, inner, dtype=torch.bfloat16, device=DEVICE)
    staging = torch.randn(tokens, 2 * inner, dtype=torch.bfloat16, device=DEVICE)
    return first, staging[:, inner:]


@pytest.mark.parametrize("dispatch", ["cta", "warp"])
def test_two_source_cpp_backend_matches_triton_bitwise(dispatch):
    """C++ backend vs Triton replica on the production operand layout.

    Regression guard: the two sources' row strides differ in production (the
    local FA3 rows vs the peer-written staging half); a C++ launcher that pins
    one stride symbol for both sources rejects exactly this input and the op
    silently degrades to Triton.  Rows mixing bf16 denormals with tiny normals
    pin the payload FTZ (a denormal times a huge 1/scale reaches fp8 range).
    """
    torch.manual_seed(5)
    tokens = 257 if dispatch == "cta" else _warp_dispatch_num_tokens()
    first, second = _ipc_staging_pair(tokens, H_LOCAL * HEAD_DIM)
    first[::7] = 0
    second[3::7] = 0
    denormal = torch.rand_like(first[1::9], dtype=torch.float32) * 9.0e-41
    first[1::9] = denormal.to(torch.bfloat16)
    tiny = torch.rand_like(second[1::9], dtype=torch.float32) * 1.1e-38
    second[1::9] = tiny.to(torch.bfloat16)
    cpp = usp_merge_quant_jit._cpp_merge_two_sources_quant_fp8(first, second)
    if cpp is None:
        pytest.skip("C++ two-source backend unavailable on this device")
    _assert_bitwise_equal(
        cpp,
        usp_merge_quant_triton.merge_two_sources_per_token_quant_fp8(first, second),
    )


def test_two_source_dispatch_order_cpp_then_triton(monkeypatch):
    """cpp-jit is the default; only inputs it cannot vectorize reach Triton."""
    calls = []
    real = usp_merge_quant_triton.merge_two_sources_per_token_quant_fp8

    def spy(first, second):
        calls.append(first.shape[-1])
        return real(first, second)

    monkeypatch.setattr(usp_merge_quant_jit, "_triton_merge_two_sources", spy)
    cpp_ready = usp_merge_quant_jit._merge_two_module_or_none(1) is not None
    good = torch.randn(64, 512, dtype=torch.bfloat16, device=DEVICE)
    usp_merge_quant_jit.merge_two_sources_per_token_quant_fp8(good, good)
    assert calls == ([] if cpp_ready else [512])
    odd = torch.randn(64, 44, dtype=torch.bfloat16, device=DEVICE)  # 44 % 8 != 0
    usp_merge_quant_jit.merge_two_sources_per_token_quant_fp8(odd, odd)
    assert calls == ([44] if cpp_ready else [512, 44])


def test_facade_routes_two_source_through_jit_dispatcher():
    # The registry must hand callers the dispatching wrapper, not a backend.
    assert (
        merge_two_sources_per_token_quant_fp8
        is usp_merge_quant_jit.merge_two_sources_per_token_quant_fp8
    )


@pytest.mark.parametrize("dispatch", ["cta", "warp"])
def test_merge_heads_cpp_backend_matches_triton_bitwise(dispatch):
    """C++ backend vs Triton replica on adversarial head-merge rows.

    The dispatcher exercises only the backend it selects, so the shadowed one
    could drift from the separated-chain contract unnoticed; pin the two to
    each other across both reference dispatch regimes and the amax=0 /
    denormal / overflow corners.
    """
    torch.manual_seed(6)
    seq = 257 if dispatch == "cta" else _warp_dispatch_num_tokens()
    x = torch.randn(
        WORLD, seq, 1, H_LOCAL, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE
    )
    x = _adversarial_fill(x)
    cpp = usp_merge_quant_jit._cpp_merge_heads_quant_fp8(x)
    if cpp is None:
        pytest.skip("C++ head-merge backend unavailable on this device")
    _assert_bitwise_equal(
        cpp, usp_merge_quant_triton.usp_merge_heads_per_token_quant_fp8(x)
    )


def test_merge_heads_dispatch_order_cpp_then_triton(monkeypatch):
    """cpp-jit is the default; only inputs it cannot vectorize reach Triton."""
    calls = []
    real = usp_merge_quant_triton.usp_merge_heads_per_token_quant_fp8

    def spy(x):
        calls.append(x.shape[-1])
        return real(x)

    monkeypatch.setattr(usp_merge_quant_jit, "_triton_merge_heads", spy)
    cpp_ready = usp_merge_quant_jit._merge_heads_module_or_none(4) is not None
    good = torch.randn(
        WORLD, 64, 1, H_LOCAL, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE
    )
    usp_merge_heads_per_token_quant_fp8(good)
    assert calls == ([] if cpp_ready else [HEAD_DIM])
    narrow = torch.randn(2, 8, 1, 1, 12, dtype=torch.bfloat16, device=DEVICE)
    _assert_bitwise_equal(  # inner % 8 != 0: Triton serves it, contract holds
        usp_merge_heads_per_token_quant_fp8(narrow), _separated_chain(narrow)
    )
    assert calls[-1:] == [12]


def test_facade_routes_merge_heads_through_jit_dispatcher():
    # The registry must hand callers the dispatching wrapper, not a backend.
    assert (
        usp_merge_heads_per_token_quant_fp8
        is usp_merge_quant_jit.usp_merge_heads_per_token_quant_fp8
    )


def test_rejects_unsupported_inputs():
    good = torch.randn(2, 3, 1, 4, 8, dtype=torch.bfloat16, device=DEVICE)
    assert can_use_usp_merge_heads_per_token_quant_fp8(good)
    assert not can_use_usp_merge_heads_per_token_quant_fp8(good.half())
    assert not can_use_usp_merge_heads_per_token_quant_fp8(good[..., ::2])
    with pytest.raises(RuntimeError, match="unsupported"):
        usp_merge_heads_per_token_quant_fp8(good.half())
    a = torch.randn(4, 8, dtype=torch.bfloat16, device=DEVICE)
    assert not can_use_merge_two_sources_per_token_quant_fp8(a, a[:, :-1])
    assert not can_use_merge_two_sources_per_token_quant_fp8(a, a.half())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
