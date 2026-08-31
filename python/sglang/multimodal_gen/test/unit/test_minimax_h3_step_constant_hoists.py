# SPDX-License-Identifier: Apache-2.0
"""Step-constant hoists in the H3 embed path: skipped time embedding under a
resident AdaLN cache, and the persistent decoder-input buffer that rewrites
only target rows after the priming forward."""

import pytest
import torch
from torch import nn

from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import MiniMaxH3DiTModel
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.denoise_loop import (
    _build_local_embedding_layout,
)

_HIDDEN = 32
_IMG_WIDTH = 12
_AUDIO_WIDTH = 6
_TEXT_LEN = 3
_SEQ_LEN = 16
# packed row map: text [0..2], img [3..10] (first 3 condition), audio
# [11..14] (first 1 reference), padding [15]
_IMG_POS = torch.arange(3, 11)
_AUDIO_POS = torch.arange(11, 15)
_TEXT_POS = torch.arange(0, 3)
_IMG_UPDATE_MASK = torch.tensor([False, False, False, True, True, True, True, True])
_AUDIO_UPDATE_MASK = torch.tensor([False, True, True, True])

_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


class _TupleLinear(nn.Linear):
    """Patch-proj stand-in returning the (output, bias) linear-layer tuple."""

    def forward(self, x: torch.Tensor):
        return nn.functional.linear(x, self.weight, self.bias), None


class _EmbedStub:
    _embed = MiniMaxH3DiTModel._embed
    _write_projected_latent_rows = MiniMaxH3DiTModel._write_projected_latent_rows

    def __init__(self, device: torch.device) -> None:
        generator = torch.Generator().manual_seed(13)
        self.hidden_size = _HIDDEN
        self.video_patch_proj = _TupleLinear(
            _IMG_WIDTH, _HIDDEN, dtype=torch.float32
        ).to(device)
        self.audio_patch_proj = _TupleLinear(
            _AUDIO_WIDTH, _HIDDEN, dtype=torch.float32
        ).to(device)
        with torch.no_grad():
            for module in (self.video_patch_proj, self.audio_patch_proj):
                for param in module.parameters():
                    param.copy_(
                        torch.randn(
                            param.shape, generator=generator, dtype=torch.float32
                        )
                    )
        self.time_embedding_calls = 0

    def _time_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        self.time_embedding_calls += 1
        return timesteps.to(torch.float32)[:, None].repeat(1, 4)


def _layout(device: torch.device) -> dict:
    return _build_local_embedding_layout(
        seq_len=_SEQ_LEN,
        text_pos=_TEXT_POS,
        img_pos=_IMG_POS,
        audio_pos=_AUDIO_POS,
        world_size=1,
        rank=0,
        device=device,
        img_update_mask=_IMG_UPDATE_MASK,
        audio_update_mask=_AUDIO_UPDATE_MASK,
    )


def _embed_kwargs(device: torch.device, layout: dict) -> dict:
    generator = torch.Generator().manual_seed(17)
    return {
        "x": torch.randn(1, _SEQ_LEN, _IMG_WIDTH, generator=generator).to(device),
        "audio_x": torch.randn(1, _SEQ_LEN, _AUDIO_WIDTH, generator=generator).to(
            device
        ),
        "text_embeddings_selected": torch.randn(
            _TEXT_LEN, _HIDDEN, generator=generator
        ).to(device),
        "unique_timesteps": torch.tensor([0.25, 0.999], device=device),
        "img_pos": _IMG_POS.to(device),
        "audio_pos": _AUDIO_POS.to(device),
        "text_pos": _TEXT_POS.to(device),
        "refiner_cu_seqlens": torch.tensor(
            [0, _TEXT_LEN, _TEXT_LEN], dtype=torch.int32, device=device
        ),
        "refiner_max_seqlen": _TEXT_LEN,
        "row_start": 0,
        "row_stop": _SEQ_LEN,
        "device": device,
        "refined_prompt_embeds_length": _TEXT_LEN,
        "local_embedding_layout": layout,
    }


@pytest.mark.parametrize("device_type", _DEVICES)
def test_time_embedding_skipped_when_adaln_cache_covers_consumers(device_type):
    device = torch.device(device_type)
    stub = _EmbedStub(device)
    kwargs = _embed_kwargs(device, _layout(device))

    _, t_emb = stub._embed(**kwargs, skip_time_embedding=True)
    assert t_emb is None
    assert stub.time_embedding_calls == 0

    _, t_emb = stub._embed(**kwargs, skip_time_embedding=False)
    assert stub.time_embedding_calls == 1
    torch.testing.assert_close(
        t_emb,
        stub._time_embedding(kwargs["unique_timesteps"]),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("device_type", _DEVICES)
def test_primed_buffer_matches_full_recompute(device_type):
    """The persistent buffer must agree with a from-scratch _embed on the same
    inputs; only the target-row GEMM batch differs between the two paths."""
    device = torch.device(device_type)
    stub = _EmbedStub(device)
    layout = _layout(device)
    kwargs = _embed_kwargs(device, layout)

    stub._embed(**kwargs, skip_time_embedding=True)
    img_target_pos = _IMG_POS[_IMG_UPDATE_MASK].to(device)
    kwargs["x"][0, img_target_pos] *= 1.5
    second, _ = stub._embed(**kwargs, skip_time_embedding=True)

    fresh_layout = _layout(device)
    fresh_kwargs = dict(kwargs, local_embedding_layout=fresh_layout)
    fresh, _ = stub._embed(**fresh_kwargs, skip_time_embedding=True)

    step_constant = torch.ones(_SEQ_LEN, dtype=torch.bool, device=device)
    step_constant[layout["img_target_row_ids"]] = False
    step_constant[layout["audio_target_row_ids"]] = False
    assert torch.equal(second[step_constant], fresh[step_constant])
    # Target rows go through a GEMM whose batch is the target count instead of
    # the full row count; kernel selection may round differently, so compare
    # with a one-bf16-ulp budget instead of bitwise.
    torch.testing.assert_close(second.float(), fresh.float(), rtol=8e-3, atol=1e-6)
