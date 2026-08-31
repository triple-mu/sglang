# SPDX-License-Identifier: Apache-2.0
"""Numerical contract of the merged [video|audio] final-layer output GEMM."""

import pytest
import torch
from torch import nn

from sglang.multimodal_gen.configs.models.dits.minimax_h3 import (
    MiniMaxH3DiTArchConfig,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
    MiniMaxH3FinalLayer,
    _modulate_scale_shift,
)
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    ensure_distributed_env_defaults,
)

_ARCH = MiniMaxH3DiTArchConfig(
    num_layers=2,
    hidden_size=64,
    latents_dim=6,
    audio_latents_dim=8,
    time_embed_dim=8,
)
_GROUPS = 3
_TOKENS = 33

_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _ensure_single_process_parallel_runtime() -> None:
    if model_parallel_is_initialized():
        return
    ensure_distributed_env_defaults()
    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)


def _build_layer(device: torch.device) -> MiniMaxH3FinalLayer:
    _ensure_single_process_parallel_runtime()
    layer = MiniMaxH3FinalLayer(
        _ARCH,
        None,
        prefix="final_layer",
        use_adaln_cache=True,
    ).to(device)
    generator = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for param in layer.parameters():
            param.copy_(
                torch.randn(param.shape, generator=generator, dtype=torch.float32).to(
                    param.dtype
                )
            )
    return layer


def _build_inputs(
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(5)
    x = torch.randn(
        _TOKENS, _ARCH.hidden_size, generator=generator, dtype=torch.float32
    ).to(device=device, dtype=torch.bfloat16)
    shift = torch.randn(
        _GROUPS, _ARCH.hidden_size, generator=generator, dtype=torch.float32
    ).to(device=device, dtype=torch.bfloat16)
    scale = torch.randn(
        _GROUPS, _ARCH.hidden_size, generator=generator, dtype=torch.float32
    ).to(device=device, dtype=torch.bfloat16)
    indices = torch.randint(0, _GROUPS, (_TOKENS,), generator=generator).to(device)
    return x, shift, scale, indices


def _reference_two_gemm(
    layer: MiniMaxH3FinalLayer,
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-merge forward: norm -> modulate -> fp32 -> one GEMM per head."""
    h = layer.norm(x)
    h = _modulate_scale_shift(h, shift, scale, indices, dtype=torch.bfloat16)
    h = h.to(torch.float32)
    video, _ = layer.video_out(h)
    audio, _ = layer.audio_out(h)
    return video, audio


class _WrappedHead(nn.Module):
    """Stand-in for any head wrapper (e.g. LoRA) that owns the projection."""

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.base = base

    def forward(self, x: torch.Tensor):
        return self.base(x)


@pytest.mark.parametrize("device_type", _DEVICES)
def test_merged_final_heads_match_two_gemm_cat_bitwise(device_type):
    device = torch.device(device_type)
    layer = _build_layer(device)
    x, shift, scale, indices = _build_inputs(device)
    with torch.inference_mode():
        video_ref, audio_ref = _reference_two_gemm(layer, x, shift, scale, indices)
        video, audio, merged = layer(
            x,
            adaln_input=None,
            inverse_indices=indices,
            adaln_params=(shift, scale),
        )
    assert merged is not None
    assert torch.equal(merged, torch.cat((video_ref, audio_ref), dim=-1))
    assert torch.equal(video, video_ref)
    assert torch.equal(audio, audio_ref)
    # video/audio must alias merged so downstream cat consumers reuse it
    assert video.data_ptr() == merged.data_ptr()
    assert (
        audio.data_ptr() == merged.data_ptr() + video.shape[-1] * merged.element_size()
    )


@pytest.mark.parametrize("device_type", _DEVICES)
def test_merged_head_cache_reuses_and_tracks_in_place_updates(device_type):
    device = torch.device(device_type)
    layer = _build_layer(device)
    x, shift, scale, indices = _build_inputs(device)
    with torch.inference_mode():
        layer(x, adaln_input=None, inverse_indices=indices, adaln_params=(shift, scale))
    first_weight = layer._merged_out_weight
    with torch.inference_mode():
        layer(x, adaln_input=None, inverse_indices=indices, adaln_params=(shift, scale))
    assert layer._merged_out_weight is first_weight

    # In-place merges (LoRA add_, weight reloads) keep the storage pointer but
    # bump the version counter; the merged concat must rebuild, not go stale.
    with torch.no_grad():
        layer.audio_out.weight.mul_(1.5)
        layer.video_out.bias.add_(0.25)
    with torch.inference_mode():
        video_ref, audio_ref = _reference_two_gemm(layer, x, shift, scale, indices)
        video, audio, merged = layer(
            x,
            adaln_input=None,
            inverse_indices=indices,
            adaln_params=(shift, scale),
        )
    assert layer._merged_out_weight is not first_weight
    assert torch.equal(merged, torch.cat((video_ref, audio_ref), dim=-1))
    assert torch.equal(video, video_ref)
    assert torch.equal(audio, audio_ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="production-shape GEMM")
def test_merged_final_heads_bitwise_at_production_shape():
    """Serving-shape parity: at the per-rank token counts this model runs
    (fl2va/t2va ~19-21k rows, hidden 5376), cuBLAS keeps the same K-reduction
    for N=128 as for N=96/N=32, so the merged GEMM is bitwise. Mid-range row
    counts (~1k-4k) can pick a split-K kernel and drift by fp32 reassociation;
    that regime is outside this model's serving shapes."""
    device = torch.device("cuda")
    arch = MiniMaxH3DiTArchConfig(num_layers=2)
    layer = MiniMaxH3FinalLayer(
        arch,
        None,
        prefix="final_layer",
        use_adaln_cache=True,
    ).to(device)
    with torch.no_grad():
        for param in layer.parameters():
            param.normal_(generator=torch.Generator(device=device).manual_seed(7))
    tokens = 18944
    generator = torch.Generator(device=device).manual_seed(9)
    x = torch.randn(tokens, arch.hidden_size, generator=generator, device=device).to(
        torch.bfloat16
    )
    shift = torch.randn(2, arch.hidden_size, generator=generator, device=device).to(
        torch.bfloat16
    )
    scale = torch.randn(2, arch.hidden_size, generator=generator, device=device).to(
        torch.bfloat16
    )
    indices = torch.randint(0, 2, (tokens,), generator=generator, device=device)
    with torch.inference_mode():
        video_ref, audio_ref = _reference_two_gemm(layer, x, shift, scale, indices)
        video, audio, merged = layer(
            x,
            adaln_input=None,
            inverse_indices=indices,
            adaln_params=(shift, scale),
        )
    assert merged is not None
    assert torch.equal(merged, torch.cat((video_ref, audio_ref), dim=-1))
    assert torch.equal(video, video_ref)
    assert torch.equal(audio, audio_ref)


@pytest.mark.parametrize("device_type", _DEVICES)
def test_merged_head_falls_back_under_head_wrappers(device_type):
    device = torch.device(device_type)
    layer = _build_layer(device)
    x, shift, scale, indices = _build_inputs(device)
    with torch.inference_mode():
        video_ref, audio_ref = _reference_two_gemm(layer, x, shift, scale, indices)
    layer.video_out = _WrappedHead(layer.video_out)
    with torch.inference_mode():
        video, audio, merged = layer(
            x,
            adaln_input=None,
            inverse_indices=indices,
            adaln_params=(shift, scale),
        )
    assert merged is None
    assert torch.equal(video, video_ref)
    assert torch.equal(audio, audio_ref)
