# SPDX-License-Identifier: Apache-2.0
"""The fused MiniMax-H3 block loop equals the eager block loop bit for bit."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from sglang.multimodal_gen import envs
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
        # The eager block probes these projections for MXFP8 input support.
        self.qkv_proj = self.fc1 = SimpleNamespace(quant_method=None)
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
    monkeypatch.setattr(envs, "SGLANG_DIFFUSION_MINIMAX_H3_FUSED_ADALN", True)
    monkeypatch.setattr(envs, "SGLANG_CACHE_DIT_ENABLED", False)
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
    monkeypatch.setattr(envs, "SGLANG_DIFFUSION_MINIMAX_H3_FUSED_ADALN", False)
    assert not model._fused_adaln_ready(x, None, params, indices)
    monkeypatch.setattr(envs, "SGLANG_DIFFUSION_MINIMAX_H3_FUSED_ADALN", True)
    monkeypatch.setattr(envs, "SGLANG_CACHE_DIT_ENABLED", True)
    assert not model._fused_adaln_ready(x, None, params, indices)
    monkeypatch.setattr(envs, "SGLANG_CACHE_DIT_ENABLED", False)
    blocks[0].preserve_input_for_cache_dit = True
    assert not model._fused_adaln_ready(x, None, params, indices)
    blocks[0].preserve_input_for_cache_dit = False
    assert model._fused_adaln_ready(x, None, params, indices)
    assert not model._fused_adaln_ready(x.float(), None, params, indices)
