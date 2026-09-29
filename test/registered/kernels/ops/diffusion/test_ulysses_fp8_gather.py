# SPDX-License-Identifier: Apache-2.0
"""Per-token fp8 rows for the Ulysses attention-output gather against torch references."""

import pytest
import torch

from sglang.kernels.ops.diffusion import (
    can_use_ulysses_fp8_gather,
    ulysses_fp8_gather_quant,
    ulysses_fp8_gather_requant,
    ulysses_fp8_gather_row_bytes,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA GPU"
)

GROUP = 896  # 7 local heads x 128
ROW = 912


def _rows(n: int, seed: int, scale: float = 1.0) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = (torch.randn(n, GROUP, generator=g, device="cuda") * scale).to(torch.bfloat16)
    x[::7] *= 30.0
    return x


def _quant_ref(x: torch.Tensor):
    """scale = amax / 448 and q = sat(x / scale), both correctly rounded."""
    f = x.float()
    amax = f.abs().amax(dim=-1, keepdim=True)
    scale = amax / torch.full_like(amax, 448.0)
    inv = torch.where(scale == 0, torch.zeros_like(scale), 1.0 / scale)
    return (f * inv).clamp(-448.0, 448.0).to(torch.float8_e4m3fn), scale


def _split(payload: torch.Tensor):
    codes = payload[..., :GROUP].view(torch.float8_e4m3fn)
    scales = payload[..., GROUP : GROUP + 4].contiguous().view(torch.float32)
    return codes, scales


def test_row_bytes_pads_to_sixteen():
    assert ulysses_fp8_gather_row_bytes(GROUP) == ROW
    assert ulysses_fp8_gather_row_bytes(128) == 144


def test_quant_rows_carry_codes_scale_and_zero_pad():
    x = _rows(1000, 1)
    x[3] = 0
    assert can_use_ulysses_fp8_gather(x)
    out = torch.full((1000, ROW), 255, dtype=torch.uint8, device="cuda")
    ulysses_fp8_gather_quant(x, out=out)
    codes, scales = _split(out)
    want_q, want_s = _quant_ref(x)
    assert torch.equal(codes.view(torch.uint8), want_q.view(torch.uint8))
    assert torch.equal(scales, want_s)
    assert not out[:, GROUP + 4 :].any()
    assert scales[3].item() == 0 and not codes[3].view(torch.uint8).any()


def test_requant_rescales_every_source_to_the_token_maximum():
    world, s_local = 8, 512
    sources = [_rows(s_local, 10 + r, scale=0.3 + 0.4 * r) for r in range(world)]
    payload = torch.empty((s_local, world, ROW), dtype=torch.uint8, device="cuda")
    for r, x in enumerate(sources):
        rows = torch.empty((s_local, ROW), dtype=torch.uint8, device="cuda")
        payload[:, r] = ulysses_fp8_gather_quant(x, out=rows)
    q, s = ulysses_fp8_gather_requant(payload, group=GROUP)
    codes, scales = _split(payload)  # [S, W, G], [S, W, 1]
    want_s = scales.amax(dim=1)
    ratio = scales / want_s.unsqueeze(1)
    rescaled = (codes.float() * ratio).to(torch.float8_e4m3fn)
    want_q = torch.where(
        (ratio == 1).expand_as(codes),
        codes.view(torch.uint8),
        rescaled.view(torch.uint8),
    )
    assert torch.equal(s, want_s)
    assert torch.equal(q.view(torch.uint8), want_q.reshape(s_local, world * GROUP))
    # Two roundings at most: within one e4m3 step of the token scale grid.
    dequant = q.float() * s
    x_all = torch.cat(sources, dim=1).float()
    tolerance = x_all.abs() * 0.125 + s * 2**-8
    assert (dequant - x_all).abs().le(tolerance).all()
