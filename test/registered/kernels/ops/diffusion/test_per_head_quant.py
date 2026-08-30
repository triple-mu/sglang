"""Per-head dynamic FP8 e4m3 quant for FA3 fp8 attention inputs.

The kernel's contract is bitwise against a three-op torch reference (exact
per-head amax, ``clamp(amax/448, min=1e-12)`` scale, ``div.rn`` payload) for
finite inputs -- FA3 consumes the payload with ``descale == scale``, so any
drift in the scale or the rounding of the payload silently changes attention
quality. ``torch.equal`` on a ``uint8`` view is the assertion throughout.
"""

import sys

import pytest
import torch

from sglang.kernels.ops.diffusion import (
    can_use_per_head_quant_fp8,
    per_head_quant_fp8,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

DEVICE = "cuda"


def _reference(x: torch.Tensor):
    amax = x.float().abs().amax(dim=(0, 2))
    scale = (amax / 448.0).clamp(min=1e-12)
    payload = (
        (x.float() / scale.view(1, -1, 1)).clamp(-448, 448).to(torch.float8_e4m3fn)
    )
    return payload, scale


def _assert_bitwise(actual, expected):
    assert torch.equal(actual[0].view(torch.uint8), expected[0].view(torch.uint8))
    assert torch.equal(actual[1], expected[1])


# Production head geometry (H3: 28 local heads x 128) plus odd shapes that
# exercise the sequence-tail and head-dim masks.
@pytest.mark.parametrize("shape", [(4096, 28, 128), (1797, 8, 128), (37, 3, 96)])
def test_matches_torch_reference(shape):
    torch.manual_seed(0)
    x = torch.randn(*shape, dtype=torch.bfloat16, device=DEVICE)
    assert can_use_per_head_quant_fp8(x)
    _assert_bitwise(per_head_quant_fp8(x), _reference(x))


def test_accepts_strided_views_of_packed_qkv():
    # The Ulysses exchange hands q/k/v as head-strided views of one packed
    # [S, H, 3*D] buffer; the kernel must read through both strides.
    torch.manual_seed(1)
    packed = torch.randn(2048, 28, 3 * 128, dtype=torch.bfloat16, device=DEVICE)
    for third in range(3):
        x = packed[..., third * 128 : (third + 1) * 128]
        assert can_use_per_head_quant_fp8(x)
        _assert_bitwise(per_head_quant_fp8(x), _reference(x.contiguous()))


def test_scale_corner_regimes():
    # Zero head (scale floor), bf16-max outlier (payload clamp boundary), and
    # bf16 denormals (must round like the fp32 reference, not flush).
    torch.manual_seed(2)
    x = torch.randn(512, 4, 128, dtype=torch.bfloat16, device=DEVICE)
    x[:, 2] = 0
    x[7, 3] = torch.finfo(torch.bfloat16).max
    x[11, 0, :64] = (torch.rand(64, dtype=torch.float32, device=DEVICE) * 9.0e-41).to(
        torch.bfloat16
    )
    _assert_bitwise(per_head_quant_fp8(x), _reference(x))


def test_rejects_unsupported_inputs():
    good = torch.randn(64, 4, 128, dtype=torch.bfloat16, device=DEVICE)
    assert can_use_per_head_quant_fp8(good)
    assert not can_use_per_head_quant_fp8(good.half())
    assert not can_use_per_head_quant_fp8(good[:, :, 0])  # 2D
    assert not can_use_per_head_quant_fp8(good.transpose(0, 2))  # column strides
    with pytest.raises(RuntimeError, match="unsupported"):
        per_head_quant_fp8(good.half())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
