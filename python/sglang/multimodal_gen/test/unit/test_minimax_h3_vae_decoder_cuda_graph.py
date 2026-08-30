# SPDX-License-Identifier: Apache-2.0
"""Replay bit-exactness, static-buffer routing, fallback, and allocation
stability for the MiniMax-H3 VAE decoder per-tile CUDA graph (V2 fusion)."""

import gc
import os
import socket

import pytest
import torch
import torch.nn as nn

from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.decoder_cuda_graph import (
    _CUDA_GRAPH_ENV,
    DecoderTileCudaGraphRunner,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.vae_vit import (
    ViT3DDecoder,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)

# Production ViT decoder config with reduced depth/width; dim_head, rope ratio,
# norm types, and suffix tokens match MiniMaxH3VideoVAE so the fused
# qknorm+rope and FA paths inside the tile forward are exercised for real.
DECODER_KWARGS = dict(
    patch_size=16,
    patch_size_t=4,
    t_causal=False,
    in_channels=24,
    out_channels=3,
    num_layers=2,
    heads=2,
    dim_head=64,
    norm_type="rms_norm",
    norm_affine=True,
    qk_norm_type="rms_norm",
    qk_norm_affine=False,
    ffn_activation_fn="silu",
    ffn_use_gated=True,
    rope_theta=100.0,
    rope_dim_ratio=0.75,
    bias=True,
    eps=1e-5,
    num_register_tokens=4,
)
TILE_SHAPE = (1, 24, 3, 8, 8)
OTHER_TILE_SHAPE = (1, 24, 3, 8, 6)


def _ensure_test_runtime(monkeypatch):
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        maybe_init_distributed_environment_and_model_parallel,
        model_parallel_is_initialized,
    )
    from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
        ensure_distributed_env_defaults,
    )

    if not model_parallel_is_initialized():
        if "MASTER_PORT" not in os.environ:
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                monkeypatch.setenv("MASTER_PORT", str(sock.getsockname()[1]))
        ensure_distributed_env_defaults()
        maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)


def _make_decoder() -> ViT3DDecoder:
    torch.manual_seed(0)
    decoder = ViT3DDecoder(**DECODER_KWARGS).to("cuda")
    decoder.eval()
    decoder.prepare_autocast_linear_weights(torch.float16)
    return decoder


def _forward_context():
    from sglang.multimodal_gen.runtime.managers.forward_context import (
        set_forward_context,
    )

    return set_forward_context(current_timestep=0, attn_metadata=None)


def _autocast():
    return torch.autocast("cuda", dtype=torch.float16)


def _tile(shape=TILE_SHAPE, seed=0) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(shape, device="cuda", generator=generator)


def _graph_entries(runner):
    return list(runner._entries.values())


@requires_cuda
@torch.no_grad()
def test_replay_is_bit_exact_and_routes_through_static_buffers(monkeypatch):
    """A graphed tile must be torch.equal with eager for the captured input and
    for fresh input values fed through the static input buffer; anything less
    breaks the decoder's lossless-first contract."""
    _ensure_test_runtime(monkeypatch)
    decoder = _make_decoder()
    runner = DecoderTileCudaGraphRunner(decoder=decoder, offload_owner=None)

    x1, x2, x3 = _tile(seed=1), _tile(seed=2), _tile(seed=3)
    with _forward_context(), _autocast():
        ref1 = decoder(x1)
        ref2 = decoder(x2)
        ref3 = decoder(x3)

        out1 = runner.run(x1)  # first sight: eager
        assert torch.equal(out1, ref1)
        (entry,) = _graph_entries(runner)
        assert entry.graph is None

        out2 = runner.run(x2)  # capture + first-replay self-check + replay
        assert runner._disabled_reason is None
        assert entry.graph is not None
        assert torch.equal(out2, ref2)
        # The retained first-tile clones are released once verified.
        assert entry.saved_input is None and entry.saved_output is None

        out3 = runner.run(x3)  # steady state: copy-in + replay + copy-out
        assert torch.equal(out3, ref3)
        # Returned tensors are fresh copies, never the static output buffer.
        assert out3.data_ptr() != entry.static_output.data_ptr()

        out1_replay = runner.run(x1)
        assert torch.equal(out1_replay, ref1)


@requires_cuda
@torch.no_grad()
def test_strided_tile_views_replay_bit_exact_with_own_signature(monkeypatch):
    """Tiled decode feeds strided canvas views, and the decoder's patchify
    copies for those but aliases for contiguous inputs, changing GEMM layouts;
    the runner must key on strides and replay each layout bit-exact."""
    _ensure_test_runtime(monkeypatch)
    decoder = _make_decoder()
    runner = DecoderTileCudaGraphRunner(decoder=decoder, offload_owner=None)

    canvas = _tile((1, 24, 3, 16, 24), seed=1)
    views = [canvas[..., 0:8, j : j + 8] for j in (0, 8, 16)]
    with _forward_context(), _autocast():
        refs = [decoder(view) for view in views]

        assert torch.equal(runner.run(views[0]), refs[0])  # eager first sight
        assert torch.equal(runner.run(views[1]), refs[1])  # capture + verify
        assert runner._disabled_reason is None
        (entry,) = _graph_entries(runner)
        assert entry.graph is not None
        assert torch.equal(runner.run(views[2]), refs[2])  # replay

        # A contiguous tensor of the same shape is a different signature.
        contiguous = views[0].clone()
        ref_contiguous = decoder(contiguous)
        assert torch.equal(runner.run(contiguous), ref_contiguous)
        assert len(runner._entries) == 2


@requires_cuda
@torch.no_grad()
def test_new_tile_shape_falls_back_then_captures_its_own_graph(monkeypatch):
    _ensure_test_runtime(monkeypatch)
    decoder = _make_decoder()
    runner = DecoderTileCudaGraphRunner(decoder=decoder, offload_owner=None)

    a1, a2 = _tile(seed=1), _tile(seed=2)
    b1, b2 = _tile(OTHER_TILE_SHAPE, seed=3), _tile(OTHER_TILE_SHAPE, seed=4)
    with _forward_context(), _autocast():
        runner.run(a1)
        runner.run(a2)
        assert len(runner._entries) == 1

        ref_b1 = decoder(b1)
        out_b1 = runner.run(b1)  # new signature: eager first sight
        assert torch.equal(out_b1, ref_b1)
        assert len(runner._entries) == 2
        entry_a, entry_b = _graph_entries(runner)
        assert entry_a.graph is not None and entry_b.graph is None

        ref_b2 = decoder(b2)
        out_b2 = runner.run(b2)  # second signature captures independently
        assert torch.equal(out_b2, ref_b2)
        assert entry_b.graph is not None
        assert runner._disabled_reason is None

        # Interleaved replays across both graphs stay exact.
        assert torch.equal(runner.run(a1), decoder(a1))
        assert torch.equal(runner.run(b1), ref_b1)


@requires_cuda
@torch.no_grad()
def test_self_check_mismatch_disables_permanently_and_stays_correct(monkeypatch):
    _ensure_test_runtime(monkeypatch)
    decoder = _make_decoder()
    runner = DecoderTileCudaGraphRunner(decoder=decoder, offload_owner=None)

    x1, x2, x3 = _tile(seed=1), _tile(seed=2), _tile(seed=3)
    with _forward_context(), _autocast():
        runner.run(x1)
        (entry,) = _graph_entries(runner)
        # Poison the retained eager output so the first-replay check must fail.
        entry.saved_output.add_(1.0)

        ref2 = decoder(x2)
        out2 = runner.run(x2)
        assert torch.equal(out2, ref2)
        assert runner._disabled_reason is not None
        assert not runner._entries

        ref3 = decoder(x3)
        assert torch.equal(runner.run(x3), ref3)
        assert not runner._entries  # disabled: no new capture attempts


@requires_cuda
@torch.no_grad()
def test_env_gate_and_layerwise_offload_force_eager(monkeypatch):
    _ensure_test_runtime(monkeypatch)
    decoder = _make_decoder()

    monkeypatch.setenv(_CUDA_GRAPH_ENV, "0")
    runner = DecoderTileCudaGraphRunner(decoder=decoder, offload_owner=None)
    with _forward_context(), _autocast():
        out = runner.run(_tile(seed=1))
        runner.run(_tile(seed=2))
    assert out is not None
    assert not runner._entries
    monkeypatch.delenv(_CUDA_GRAPH_ENV)

    from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
        LayerwiseOffloadableModuleMixin,
    )

    class _OffloadedHost(nn.Module, LayerwiseOffloadableModuleMixin):
        pass

    host = _OffloadedHost()
    host.layerwise_offload_managers = [object()]
    runner = DecoderTileCudaGraphRunner(decoder=decoder, offload_owner=host)
    with _forward_context(), _autocast():
        runner.run(_tile(seed=1))
        runner.run(_tile(seed=2))
    assert not runner._entries


@requires_cuda
@torch.no_grad()
def test_weight_rematerialization_invalidates_captures(monkeypatch):
    """`.to()` and autocast weight prep replace parameter storages; a stale
    graph would replay against freed pointers, so captures must be dropped."""
    _ensure_test_runtime(monkeypatch)
    decoder = _make_decoder()
    runner = DecoderTileCudaGraphRunner(decoder=decoder, offload_owner=None)

    x1, x2 = _tile(seed=1), _tile(seed=2)
    with _forward_context(), _autocast():
        runner.run(x1)
        runner.run(x2)
        (entry,) = _graph_entries(runner)
        assert entry.graph is not None

        epoch = decoder._graph_epoch
        decoder.to("cuda")  # same-device move still re-runs _apply
        assert decoder._graph_epoch == epoch + 1

        ref = decoder(x1)
        out = runner.run(x1)  # epoch change: back to eager first sight
        assert torch.equal(out, ref)
        (entry,) = _graph_entries(runner)
        assert entry.graph is None


@requires_cuda
@torch.no_grad()
def test_wired_tiled_decode_captures_and_repeats_byte_identical(monkeypatch):
    """Acceptance for the decode() wiring: repeated decodes of the same latent
    must stay byte-identical across the eager -> captured -> replayed
    transitions, and the runner must actually capture through the tile loop."""
    from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.klvae import (
        AutoencoderKLLegacy,
    )

    _ensure_test_runtime(monkeypatch)
    torch.manual_seed(0)
    vae = AutoencoderKLLegacy(
        in_channels=3,
        out_ch=3,
        ch=32,
        embed_dim=8,
        z_channels=8,
        use_3d_conv=True,
        num_res_blocks=1,
        ch_mult=[1, 1],
        space_down=[2, 2],
        time_down=[1, 1],
        causal_encoder=True,
        causal_decoder=False,
        use_vit_decoder=True,
        vit_decoder_kwargs=dict(
            heads=2,
            dim_head=64,
            num_layers=2,
            norm_type="rms_norm",
            qk_norm_type="rms_norm",
            qk_norm_affine=False,
            norm_affine=True,
            ffn_activation_fn="silu",
            ffn_use_gated=True,
            rope_dim_ratio=0.75,
            rope_theta=100.0,
        ),
        decoder_tiling=True,
        tile_size=32,
        tile_overlap_min=8,
    ).to("cuda")
    vae.eval()
    # Random weights stand in for a folded checkpoint; mark the folds done so
    # _require_folded_weights admits this test-only instance.
    vae.conv_in_pixel_norm_folded.fill_(True)
    vae.proj_out_pixel_denorm_folded.fill_(True)
    vae.decoder.prepare_autocast_linear_weights(torch.float16)

    z = _tile((1, 8, 2, 16, 24), seed=1)
    runner = vae._decoder_graph_runner
    with _forward_context(), _autocast():
        out1 = vae._adaptive_decode(z)
        out2 = vae._adaptive_decode(z)
        assert runner._disabled_reason is None
        assert runner._entries
        assert any(entry.graph is not None for entry in runner._entries.values())
        out3 = vae._adaptive_decode(z)
    assert torch.equal(out1, out2)
    assert torch.equal(out2, out3)


@requires_cuda
@torch.no_grad()
def test_no_allocation_growth_across_replays(monkeypatch):
    _ensure_test_runtime(monkeypatch)
    decoder = _make_decoder()
    runner = DecoderTileCudaGraphRunner(decoder=decoder, offload_owner=None)

    x = _tile(seed=1)
    with _forward_context(), _autocast():
        runner.run(x)
        x.normal_()
        runner.run(x)  # capture
        x.normal_()
        out = runner.run(x)  # prime steady-state allocations (output clone)
        del out

        # Collect first: in a shared-process suite, unrelated garbage freed
        # mid-loop would make an equality baseline flaky. Growth is the bug.
        gc.collect()
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        for _ in range(10):
            x.normal_()
            out = runner.run(x)
            del out
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() <= allocated
        assert torch.cuda.memory_reserved() <= reserved
