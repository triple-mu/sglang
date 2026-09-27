# SPDX-License-Identifier: Apache-2.0
"""Fused SwiGLU + per-token FP8 quantisation against a torch reference."""

import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.diffusion import (
    can_use_silu_mul_per_token_quant_fp8,
    silu_mul_per_token_quant_fp8,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=40, stage="base-b-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA GPU"
)

HIDDEN = 14336  # MiniMax-H3 fc1 output is [gate | up] of this width each


def _reference(x: torch.Tensor):
    gate, up = x.chunk(2, dim=-1)
    act = (F.silu(gate.float()).to(torch.bfloat16).float() * up.float()).to(
        torch.bfloat16
    )
    amax = act.float().abs().amax(dim=-1, keepdim=True)
    scale = amax / 448.0
    return act, scale


@pytest.mark.parametrize("rows", [4736, 37, 1])
def test_scale_and_payload_follow_the_activation(rows):
    g = torch.Generator(device="cuda").manual_seed(rows)
    x = (torch.randn(rows, 2 * HIDDEN, generator=g, device="cuda") * 3.0).to(
        torch.bfloat16
    )
    x[0] = 0  # an all-zero row must quantise to zeros with a zero scale
    assert can_use_silu_mul_per_token_quant_fp8(x)
    q, s = silu_mul_per_token_quant_fp8(x)
    act, scale = _reference(x)
    assert q.dtype is torch.float8_e4m3fn and tuple(q.shape) == (rows, HIDDEN)
    assert tuple(s.shape) == (rows, 1)
    torch.testing.assert_close(s, scale, rtol=1e-6, atol=0)
    dequant = q.float() * s
    # fp8 e4m3 carries 3 mantissa bits: one ulp is at most 1/8 of the value.
    tolerance = act.float().abs() * 0.0625 + scale * 1e-3
    assert torch.all((dequant - act.float()).abs() <= tolerance + 1e-6)
    assert not q[0].float().any() and s[0].item() == 0.0


def test_strided_rows_and_wrong_widths():
    x = torch.randn(64, 2 * HIDDEN + 16, device="cuda", dtype=torch.bfloat16)
    view = x[:, : 2 * HIDDEN]  # row stride keeps 16-byte alignment
    assert can_use_silu_mul_per_token_quant_fp8(view)
    q, s = silu_mul_per_token_quant_fp8(view)
    act, scale = _reference(view)
    torch.testing.assert_close(s, scale, rtol=1e-6, atol=0)
    assert not can_use_silu_mul_per_token_quant_fp8(x[:, : 2 * HIDDEN - 8])
    assert not can_use_silu_mul_per_token_quant_fp8(x.float())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
