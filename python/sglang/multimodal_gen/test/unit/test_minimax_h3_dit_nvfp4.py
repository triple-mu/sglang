"""The MiniMax-H3 DiT NVFP4 overlay: FP4 GEMM within tolerance, FP8 for early steps and fp8 rows."""

import types

import pytest
import torch

pytest.importorskip("flashinfer")
from sglang.multimodal_gen.runtime.models.dits import minimax_h3_nvfp4 as mod

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _fp8_layer(n: int, k: int):
    torch.manual_seed(0)
    w = (torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)
    scale = w.float().abs().amax(dim=1) / 448.0
    w8 = (w.float() / scale[:, None]).to(torch.float8_e4m3fn)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(w8.t(), requires_grad=False)  # [K, N] as Fp8LinearMethod stores it
    layer.weight_scale = torch.nn.Parameter(scale, requires_grad=False)
    dequant = (w8.float() * scale[:, None]).to(torch.bfloat16)
    return layer, dequant


def test_overlay_gemm_tracks_bf16_and_routes_early_steps_and_fp8_rows_to_fp8(monkeypatch):
    n, k, m = 256, 512, 300
    layer, w = _fp8_layer(n, k)
    calls = []
    fp8 = types.SimpleNamespace(
        apply=lambda layer, x, bias=None: calls.append("fp8")
        or torch.zeros(m, n, device="cuda", dtype=torch.bfloat16)
    )
    overlay = mod.MiniMaxH3NVFP4Overlay(fp8, backend="cutlass", from_step=2)
    overlay.prepare(layer)
    assert layer.nvfp4_weight.shape == (n, k // 2) and layer.nvfp4_weight.dtype == torch.uint8
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    ref = x.float() @ w.float().t()

    monkeypatch.setattr(mod, "_current_step", lambda: 3)
    out = overlay.apply(layer, x)
    rel = ((out.float() - ref).norm() / ref.norm()).item()
    assert out.shape == (m, n) and out.dtype == torch.bfloat16
    assert rel < 0.2, rel  # E2M1 codes with block-16 E4M3 scales on Gaussian data
    assert not calls

    monkeypatch.setattr(mod, "_current_step", lambda: 1)
    overlay.apply(layer, x)
    assert calls == ["fp8"]

    monkeypatch.setattr(mod, "_current_step", lambda: 5)
    overlay.apply(layer, (x, torch.ones(m, 1, device="cuda")))
    assert calls == ["fp8", "fp8"]

    # Outside a forward (no context) the overlay never falls back silently to FP8.
    monkeypatch.setattr(mod, "_current_step", lambda: None)
    overlay.apply(layer, x)
    assert calls == ["fp8", "fp8"]


def test_install_is_a_no_op_when_unset(monkeypatch):
    monkeypatch.setenv("SGLANG_DIFFUSION_MINIMAX_H3_DIT_NVFP4_LAYERS", "")
    assert mod.install_minimax_h3_dit_nvfp4(model=None) == 0
