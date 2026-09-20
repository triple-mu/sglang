# SPDX-License-Identifier: Apache-2.0
"""H3 batching order, explicit deployment options, and request isolation."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch import nn

from sglang.multimodal_gen.configs.models.vaes.minimax_h3_video import (
    MiniMaxH3VideoVAEConfig,
)
from sglang.multimodal_gen.runtime.models.vaes.fast_path_gate import use_vae_fast_path
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.batching import (
    decode_windows,
    flat_decode_tiles,
    forward_many,
    partitions,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.fp8 import (
    LINEAR_SHAPES,
    inspect_fp8_scope,
    target_paths,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.optimizations import (
    install_optimization_state,
    optimization_active,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.vit_utils import (
    prepare_rotary_pos_emb,
)


def test_options_are_opt_in_and_fp8_is_independent():
    config = MiniMaxH3VideoVAEConfig()
    config.validate_optimization_options()
    assert not config.enable_optimizations
    assert config.decoder_quantization is None
    assert config.decoder_output_projection_precision == "fp32"
    assert (
        config.encoder_tile_batch_size,
        config.decoder_tile_batch_size,
        config.decoder_window_batch_size,
    ) == (8, 64, 0)
    config.decoder_quantization = "fp8"
    config.validate_optimization_options()
    assert not config.enable_optimizations


@pytest.mark.parametrize(
    "field,value",
    [
        ("enable_optimizations", 1),
        ("encoder_tile_batch_size", 0),
        ("encoder_tile_batch_size", True),
        ("decoder_tile_batch_size", 65),
        ("decoder_window_batch_size", -1),
        ("decoder_window_batch_size", 1.5),
        ("decoder_quantization", "int8"),
        ("decoder_output_projection_precision", "bf16"),
    ],
)
def test_invalid_options(field, value):
    config = MiniMaxH3VideoVAEConfig()
    setattr(config, field, value)
    with pytest.raises(ValueError):
        config.validate_optimization_options()


def test_adjacent_batches_preserve_complete_sample_order_and_capacity():
    values = [torch.full((2, 3, 4), float(i)) for i in range(5)]
    calls = []
    collector = mock.Mock()

    def forward(x):
        calls.append(x.clone())
        return x + 7

    outputs = forward_many(values, forward, capacity=5, collector=collector)
    assert [x.shape[0] for x in calls] == [4, 4, 2]
    for original, actual in zip(values, outputs):
        torch.testing.assert_close(actual, original + 7)
    assert collector.collect_stacked.call_args_list == [
        mock.call(2, 2),
        mock.call(2, 2),
    ]
    collector.collect.assert_called_once()
    values.insert(2, torch.zeros(2, 3, 5))
    assert list(partitions(values, 8)) == [(0, 2), (2, 3), (3, 6)]


@pytest.mark.parametrize("capacity", [1, 3, 8, 64])
def test_decoder_flatten_restores_rank_tile_and_window_order(capacity):
    tiles = [torch.arange(5 * 2 * 3).reshape(5, 2, 3) + 100 * i for i in range(7)]
    indices = [1, 4, 6]
    sizes = []

    def forward(x):
        sizes.append(x.shape[0])
        return x * 2

    result = flat_decode_tiles(tiles, indices, forward, capacity)
    assert max(sizes) <= capacity
    for index, actual in zip(indices, result):
        torch.testing.assert_close(actual, tiles[index] * 2)


@pytest.mark.parametrize("limit", [0, 1, 2, 3, 64])
@pytest.mark.parametrize("use_head_tail", [False, True])
def test_windows_keep_time_axis_and_boundary_shapes(limit, use_head_tail):
    z = torch.arange(2 * 3 * 13).reshape(2, 3, 13, 1, 1)
    head = torch.full((2, 3, 1, 1, 1), -1) if use_head_tail else None
    tail = torch.full((2, 3, 1, 1, 1), -2) if use_head_tail else None
    calls = []

    def decode(x):
        calls.append(x.clone())
        return x.repeat_interleave(2, dim=2)

    vae = SimpleNamespace(tokens_chunk_size=3, token_overlap=1, _adaptive_decode=decode)
    outputs = dict(decode_windows(vae, z, head, tail, 4, limit))
    for index in range(4):
        expected = z[:, :, index * 3 : index * 3 + 4]
        if index == 0 and head is not None:
            expected = torch.cat([head, expected], dim=2)
        if index == 3 and tail is not None:
            expected = torch.cat([expected, tail], dim=2)
        torch.testing.assert_close(outputs[index], expected.repeat_interleave(2, dim=2))
    assert all(x.shape[0] % 2 == 0 for x in calls)
    if limit:
        assert max(x.shape[0] for x in calls) <= 2 * limit
    if use_head_tail:
        assert calls[0].shape[2] == 5 and calls[-1].shape[2] == 5


def test_request_gate_resets_on_error_and_does_not_enable_cpu():
    vae = nn.Module()
    vae.encoder = nn.Sequential(nn.Identity())
    vae.decoder = nn.Sequential(nn.Identity())
    vae.processor = SimpleNamespace()
    config = MiniMaxH3VideoVAEConfig(
        enable_optimizations=True, decoder_quantization="fp8"
    )
    install_optimization_state(vae, config)
    state = vae._sgl_h3_vae_optimization
    assert vae.encoder[0]._sgl_h3_vae_optimization is state
    assert vae.processor._sgl_h3_vae_optimization is state
    with pytest.raises(RuntimeError, match="request failed"):
        with use_vae_fast_path(vae, True):
            assert state.gate.enabled
            assert not optimization_active(vae, torch.ones(2))
            with use_vae_fast_path(vae, False):
                assert not state.gate.enabled
            raise RuntimeError("request failed")
    assert not state.gate.enabled
    assert config.decoder_quantization == "fp8"


def test_batched_rope_has_reference_fallback_on_cpu():
    cos = torch.randn(3, 7, 1, 48)
    sin = torch.randn_like(cos)
    for enabled in (False, True):
        result = prepare_rotary_pos_emb(
            (cos, sin), dtype=torch.float16, allow_batched_native=enabled
        )
        assert len(result) == 2
        torch.testing.assert_close(result[0], cos)
        torch.testing.assert_close(result[1], sin)


def test_fp8_scope_is_exactly_the_released_144_linears():
    # Meta parameters validate the real topology without allocating 10 GB.
    decoder = nn.Module()
    decoder.transformer_blocks = nn.ModuleList()
    with torch.device("meta"):
        for _ in range(36):
            block = nn.Module()
            block.attn = nn.Module()
            block.ff = nn.Module()
            for path, (out_features, in_features) in LINEAR_SHAPES.items():
                owner, name = path.split(".")
                setattr(
                    getattr(block, owner), name, nn.Linear(in_features, out_features)
                )
            decoder.transformer_blocks.append(block)
        decoder.proj_out = nn.Linear(2048, 3072)
    paths = inspect_fp8_scope(decoder)
    assert len(paths) == len(set(paths)) == 144
    assert paths == target_paths()
    assert "proj_out" not in paths
    decoder.transformer_blocks[5].ff.w2 = nn.Linear(1, 1, device="meta")
    with pytest.raises(ValueError, match="transformer_blocks.5.ff.w2"):
        inspect_fp8_scope(decoder)


def test_documented_cli_options_are_registered():
    import argparse

    from sglang.multimodal_gen.configs.models.vaes.base import VAEConfig

    parser = argparse.ArgumentParser()
    VAEConfig.add_cli_args(parser)
    values = vars(
        parser.parse_args(
            [
                "--vae-config.enable-optimizations",
                "true",
                "--vae-config.encoder-tile-batch-size",
                "8",
                "--vae-config.decoder-tile-batch-size",
                "64",
                "--vae-config.decoder-window-batch-size",
                "0",
                "--vae-config.decoder-quantization",
                "fp8",
            ]
        )
    )
    assert values["vae_config.enable_optimizations"] is True
    assert values["vae_config.decoder_quantization"] == "fp8"
    assert values["vae_config.decoder_tile_batch_size"] == 64


def test_precision_choice_is_independent_of_request_fusion_gate():
    from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.optimizations import (
        MiniMaxH3VAEOptimizationState,
    )

    config = MiniMaxH3VideoVAEConfig(enable_optimizations=True)
    state = MiniMaxH3VAEOptimizationState(config)
    with mock.patch.object(state, "supported", return_value=True):
        state.gate.enabled = True
        assert state.enabled(None)
        assert not state.output_projection_enabled(None)
        config.decoder_output_projection_precision = "fp16"
        assert state.output_projection_enabled(None)
        state.gate.enabled = False
        assert not state.enabled(None)
        assert state.output_projection_enabled(None)
        assert config.decoder_quantization is None


def test_output_projection_precision_cli_is_explicit():
    import argparse

    from sglang.multimodal_gen.configs.models.vaes.base import VAEConfig

    parser = argparse.ArgumentParser()
    VAEConfig.add_cli_args(parser)
    values = vars(
        parser.parse_args(["--vae-config.decoder-output-projection-precision", "fp16"])
    )
    assert values["vae_config.decoder_output_projection_precision"] == "fp16"
    assert values["vae_config.decoder_quantization"] is None
    assert values["vae_config.enable_optimizations"] is None
