# SPDX-License-Identifier: Apache-2.0
"""The fused MiniMax-H3 block loop equals the eager block loop bit for bit."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from sglang.multimodal_gen.runtime.models.dits import minimax_h3 as m

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA GPU"
)

HIDDEN = 5376
EPS = 1e-5


class _Mixer(nn.Module):
    """Deterministic bf16 stand-in for attention and the MLP."""

    def __init__(self, seed: int):
        super().__init__()
        # The blocks probe these projections for pre-quantised input support.
        self.qkv_proj = self.fc1 = SimpleNamespace(quant_method=None)
        self.needs_bf16_rows = False
        g = torch.Generator(device="cuda").manual_seed(seed)
        self.weight = nn.Parameter(
            (torch.randn(HIDDEN, HIDDEN, generator=g, device="cuda") * HIDDEN**-0.5).to(
                torch.bfloat16
            ),
            requires_grad=False,
        )

    def forward(self, x, **_kwargs):
        return x @ self.weight


def _block(seed: int) -> m.MiniMaxH3DiTBlock:
    block = m.MiniMaxH3DiTBlock.__new__(m.MiniMaxH3DiTBlock)
    nn.Module.__init__(block)
    g = torch.Generator(device="cuda").manual_seed(seed)
    block.norm1 = nn.RMSNorm(HIDDEN, eps=EPS, dtype=torch.bfloat16, device="cuda")
    block.norm2 = nn.RMSNorm(HIDDEN, eps=EPS, dtype=torch.bfloat16, device="cuda")
    with torch.no_grad():
        for norm in (block.norm1, block.norm2):
            norm.weight.copy_(
                1.0 + 0.2 * torch.randn(HIDDEN, generator=g, device="cuda")
            )
    block.attn = _Mixer(seed + 100)
    block.mlp = _Mixer(seed + 200)
    block.adaln_proj = None
    block.preserve_input_for_cache_dit = False
    block.adaln_layer_index = -1
    block._adaln_cache_ref = ()
    return block


def _params(num_blocks: int, groups: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    slabs = [
        (torch.randn(groups, 6 * HIDDEN, generator=g, device="cuda") * 0.3).to(
            torch.bfloat16
        )
        for _ in range(num_blocks)
    ]
    return tuple(tuple(slab.chunk(6, dim=-1)) for slab in slabs)


def _model(blocks) -> m.MiniMaxH3DiTModel:
    model = m.MiniMaxH3DiTModel.__new__(m.MiniMaxH3DiTModel)
    nn.Module.__init__(model)
    model.blocks = nn.ModuleList(blocks)
    model.adaln_cache = None
    return model


def _kwargs(rows: int):
    return dict(
        rope_cache=None,
        cu_seqlens=torch.tensor([0, rows], device="cuda", dtype=torch.int32),
        max_seqlen=rows,
    )


@requires_cuda
def test_fused_block_loop_is_bitwise_the_eager_loop(monkeypatch):
    monkeypatch.setenv("SGLANG_DIFFUSION_MINIMAX_H3_FUSED_ADALN", "true")
    monkeypatch.setenv("SGLANG_CACHE_DIT_ENABLED", "false")
    monkeypatch.setattr(m, "_FUSED_ADALN_GATE", m.BitExactFusionGate("test"))
    rows, groups = 2048, 4
    blocks = [_block(seed) for seed in range(3)]
    params = _params(len(blocks), groups, 7)
    model = _model(blocks)
    g = torch.Generator(device="cuda").manual_seed(11)
    x = torch.randn(rows, HIDDEN, generator=g, device="cuda").to(torch.bfloat16)
    indices = torch.randint(0, groups, (rows,), generator=g, device="cuda")

    eager = x.clone()
    for block, block_params in zip(blocks, params):
        eager = block(
            eager,
            adaln_input=None,
            combined_indices=indices,
            adaln_params=block_params,
            **_kwargs(rows),
        )

    assert model._fused_adaln_ready(x, None, params, indices)
    assert m._FUSED_ADALN_GATE.verified
    fused = x.clone()
    pending = None
    for index, block in enumerate(blocks):
        fused, pending = block.forward_fused(
            fused,
            adaln_params=model._block_adaln_params(block, index, None, params),
            pending_gate=pending,
            combined_indices=indices,
            **_kwargs(rows),
        )
    fused = m._modulate_gate(fused, *pending, indices, dtype=torch.bfloat16)
    assert torch.equal(fused, eager)


@requires_cuda
def test_fused_loop_stays_off_under_cache_dit_or_the_kill_switch(monkeypatch):
    blocks = [_block(0)]
    params = _params(1, 2, 1)
    model = _model(blocks)
    x = torch.randn(256, HIDDEN, device="cuda").to(torch.bfloat16)
    indices = torch.zeros(256, dtype=torch.int64, device="cuda")
    monkeypatch.setattr(m, "_FUSED_ADALN_GATE", m.BitExactFusionGate("test"))
    monkeypatch.setenv("SGLANG_DIFFUSION_MINIMAX_H3_FUSED_ADALN", "false")
    assert not model._fused_adaln_ready(x, None, params, indices)
    monkeypatch.setenv("SGLANG_DIFFUSION_MINIMAX_H3_FUSED_ADALN", "true")
    monkeypatch.setenv("SGLANG_CACHE_DIT_ENABLED", "true")
    assert not model._fused_adaln_ready(x, None, params, indices)
    monkeypatch.setenv("SGLANG_CACHE_DIT_ENABLED", "false")
    blocks[0].preserve_input_for_cache_dit = True
    assert not model._fused_adaln_ready(x, None, params, indices)
    blocks[0].preserve_input_for_cache_dit = False
    assert model._fused_adaln_ready(x, None, params, indices)
    assert not model._fused_adaln_ready(x.float(), None, params, indices)


class _AdalnCache:
    """Stand-in for the online AdaLN cache: one parameter set per layer."""

    def __init__(self, params):
        self.params = params

    def block_for_current_step(self, layer_index):
        return self.params[layer_index]


def _cache_dit_blocks(num_blocks: int, groups: int):
    """Blocks as Cache-DiT sees them: inputs preserved, AdaLN resolved per layer."""
    blocks = [_block(seed) for seed in range(num_blocks)]
    params = _params(num_blocks, groups, 7)
    cache = _AdalnCache(params)
    for index, block in enumerate(blocks):
        block.preserve_input_for_cache_dit = True
        block._adaln_cache_ref = (cache,)
        block.adaln_layer_index = index
    return blocks, params


def _eager_middle(blocks, params, x, indices, rows):
    out = x
    for index, block in enumerate(blocks, start=1):
        out = block(
            out,
            adaln_input=None,
            combined_indices=indices,
            adaln_params=params[index],
            **_kwargs(rows),
        )
    return out


@requires_cuda
def test_cache_dit_middle_blocks_run_the_fused_chain_bitwise(monkeypatch):
    monkeypatch.setenv("SGLANG_DIFFUSION_MINIMAX_H3_FUSED_ADALN", "true")
    monkeypatch.setenv("SGLANG_CACHE_DIT_ENABLED", "true")
    monkeypatch.setattr(m, "_FUSED_ADALN_GATE", m.BitExactFusionGate("test"))
    rows, groups = 2048, 4
    blocks, params = _cache_dit_blocks(4, groups)
    model = _model(blocks)
    g = torch.Generator(device="cuda").manual_seed(11)
    x = torch.randn(rows, HIDDEN, generator=g, device="cuda").to(torch.bfloat16)
    indices = torch.randint(0, groups, (rows,), generator=g, device="cuda")
    middle = blocks[1:]
    want = _eager_middle(middle, params, x, indices, rows)

    kept = x.clone()
    got = model.run_cache_dit_middle_blocks(
        middle, x, adaln_input=None, combined_indices=indices, **_kwargs(rows)
    )
    assert got is not None
    assert torch.equal(got, want)
    assert torch.equal(x, kept)  # Cache-DiT still holds the range input
    # the model-level fused loop stays off while Cache-DiT owns the loop
    assert not model._fused_adaln_ready(x, None, params, indices)


@requires_cuda
def test_cache_dit_middle_blocks_decline_when_the_chain_cannot_run(monkeypatch):
    monkeypatch.setenv("SGLANG_DIFFUSION_MINIMAX_H3_FUSED_ADALN", "false")
    monkeypatch.setattr(m, "_FUSED_ADALN_GATE", m.BitExactFusionGate("test"))
    blocks, params = _cache_dit_blocks(2, 2)
    model = _model(blocks)
    x = torch.randn(256, HIDDEN, device="cuda").to(torch.bfloat16)
    indices = torch.zeros(256, dtype=torch.int64, device="cuda")
    assert (
        model.run_cache_dit_middle_blocks(
            blocks[1:], x, adaln_input=None, combined_indices=indices, **_kwargs(256)
        )
        is None
    )
    monkeypatch.setenv("SGLANG_DIFFUSION_MINIMAX_H3_FUSED_ADALN", "true")
    # per-block parameters handed in by position belong to the eager loop
    assert (
        model.run_cache_dit_middle_blocks(
            blocks[1:],
            x,
            adaln_input=None,
            adaln_params=params[0],
            combined_indices=indices,
            **_kwargs(256),
        )
        is None
    )


@requires_cuda
def test_patched_cache_dit_middle_range_returns_hidden_and_residual(monkeypatch):
    from cache_dit.caching.cache_blocks.pattern_3_4_5 import (
        CachedBlocks_Pattern_3_4_5,
    )

    from sglang.multimodal_gen.runtime.cache import cache_dit_integration

    monkeypatch.setenv("SGLANG_DIFFUSION_MINIMAX_H3_FUSED_ADALN", "true")
    monkeypatch.setenv("SGLANG_CACHE_DIT_ENABLED", "true")
    monkeypatch.setattr(m, "_FUSED_ADALN_GATE", m.BitExactFusionGate("test"))
    cache_dit_integration._patch_cache_dit_middle_blocks()
    rows, groups = 1024, 3
    blocks, params = _cache_dit_blocks(3, groups)
    model = _model(blocks)
    g = torch.Generator(device="cuda").manual_seed(5)
    x = torch.randn(rows, HIDDEN, generator=g, device="cuda").to(torch.bfloat16)
    indices = torch.randint(0, groups, (rows,), generator=g, device="cuda")
    middle = blocks[1:]
    want = _eager_middle(middle, params, x, indices, rows)

    wrapper = SimpleNamespace(transformer=model, _Mn_blocks=lambda: middle)
    hidden, encoder, residual = CachedBlocks_Pattern_3_4_5.call_Mn_blocks(
        wrapper, x, adaln_input=None, combined_indices=indices, **_kwargs(rows)
    )
    assert encoder is None
    assert torch.equal(hidden, want)
    assert torch.equal(residual, want - x)


def test_static_activation_scale_blocks_the_per_token_fp8_hand_over():
    from sglang.multimodal_gen.runtime.layers.quantization.fp8 import (
        Fp8Config,
        Fp8LinearMethod,
    )

    method = Fp8LinearMethod(
        Fp8Config(is_checkpoint_fp8_serialized=True, activation_scheme="static")
    )
    linear = SimpleNamespace(
        quant_method=method,
        input_scale=torch.ones((), device="cuda"),
        weight_scale=torch.ones(8, device="cuda"),
        weight=torch.empty(4, 8, device="cuda"),
    )
    assert "static activation scale" in m._per_token_fp8_blockers(linear)
    linear.input_scale = None
    assert "static activation scale" not in m._per_token_fp8_blockers(linear)


def test_lora_wrapped_linear_keeps_the_bf16_hand_over():
    """A LoRA wrapper exposes only weight/bias (#41272); it must be declined, not crash."""

    class _Wrapped(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(4, 8), requires_grad=False)

    wrapped = _Wrapped()
    assert m._per_token_fp8_blockers(wrapped) == ["quant method NoneType"]
    assert not m._fused_norm_feeds_fp8(wrapped)
