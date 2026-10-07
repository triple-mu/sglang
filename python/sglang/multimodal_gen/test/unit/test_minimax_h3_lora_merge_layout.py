"""Diffusers-layout LoRA deltas land on the native fused layout: fc1 swaps [value, gate] -> [gate, value]."""

import torch

from sglang.multimodal_gen.tools.build_minimax_h3_pdd_weights import (
    _interleave_qkv,
    _native_delta,
    _target_of,
)


def test_fc1_delta_swaps_its_output_halves_and_nothing_else_moves():
    value = torch.full((4, 3), 1.0)
    gate = torch.full((4, 3), 2.0)
    swapped = _native_delta("blocks.0.mlp.fc1", torch.cat((value, gate)))
    assert torch.equal(swapped, torch.cat((gate, value)))
    same = torch.arange(12.0).reshape(4, 3)
    assert torch.equal(_native_delta("blocks.0.mlp.fc2", same), same)
    assert torch.equal(_native_delta("token_refiner.blocks.1.attn.out_proj", same), same)


def test_target_mapping_covers_the_dit_and_refiner_modules():
    assert _target_of("transformer_blocks.3.ff.net.0.proj") == "blocks.3.mlp.fc1"
    assert _target_of("transformer_blocks.3.ff.net.2") == "blocks.3.mlp.fc2"
    assert _target_of("transformer_blocks.3.attn.to_out.0") == "blocks.3.attn.out_proj"
    assert _target_of("token_refiner.refiner_blocks.1.attn.to_q") == "token_refiner.blocks.1.attn.to_q"
    assert _target_of("time_embedder.linear_1") is None


def test_qkv_interleave_is_per_head():
    q = torch.zeros((256, 8)); k = torch.ones((256, 8)); v = torch.full((256, 8), 2.0)
    out = _interleave_qkv(q, k, v)
    assert out.shape == (768, 8)
    assert torch.equal(out[0:128], q[0:128]) and torch.equal(out[128:256], k[0:128]) and torch.equal(out[256:384], v[0:128])
