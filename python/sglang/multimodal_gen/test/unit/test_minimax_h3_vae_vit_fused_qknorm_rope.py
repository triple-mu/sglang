# SPDX-License-Identifier: Apache-2.0
"""Bit-exactness and fallback tests for the MiniMax-H3 VAE decoder fused
qknorm+rope chain (V1 fusion)."""

import os
import socket

import pytest
import torch
import torch.nn as nn

from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.attention import (
    Attention,
    _apply_qk_norm,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.base_module import (
    RotaryEmbeddingND,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.vit_utils import (
    _FUSED_QKNORM_ROPE_ENV,
    apply_rotary_pos_emb_qk,
    create_token_ids,
    fused_qknorm_rope_qk,
    prepare_rotary_pos_emb,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)

HEADS = 32
HEAD_DIM = 64
ROPE_DIM = 48
# 7*16*16 latent patches + 4 register tokens + 1 cls token = 1797, the
# production decode tile shape.
LATENT_DIMS = [7, 16, 16]
NUM_SUFFIX_TOKENS = 5


def _make_rotary_pos_emb(dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    pos_embed = RotaryEmbeddingND(ROPE_DIM, 100.0, n_dim=3, use_angle=True).to("cuda")
    token_ids = create_token_ids(LATENT_DIMS, "cuda", torch.float32)
    token_ids = torch.cat(
        [token_ids, torch.zeros(1, NUM_SUFFIX_TOKENS, 3, device="cuda")], dim=1
    )
    cos, sin = pos_embed(token_ids)
    rotary = prepare_rotary_pos_emb((cos, sin), dtype=dtype)
    assert len(rotary) == 4, "native rotary cache path did not trigger"
    return rotary


def _make_qkv_views(
    seq_len: int, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    qkv = torch.randn(1, seq_len, HEADS, 3 * HEAD_DIM, device="cuda", dtype=dtype)
    query, key, value = torch.chunk(qkv, 3, dim=-1)
    return qkv, query, key, value


def _weightless_rms_norm() -> nn.RMSNorm:
    return nn.RMSNorm(HEAD_DIM, eps=1e-5, elementwise_affine=False).to("cuda")


@requires_cuda
@torch.no_grad()
def test_fused_qknorm_rope_matches_eager_chain_bitwise():
    """The fused kernel replicates aten's fp16 RMSNorm reduction order and the
    sgl_kernel NeoX rope rounding, so the decoder chain must stay torch.equal;
    a tolerance regression here breaks the decoder's lossless-first contract.
    """
    torch.manual_seed(0)
    rotary = _make_rotary_pos_emb(torch.float16)
    seq_len = rotary[2].shape[0]
    norm = _weightless_rms_norm()

    qkv, query, key, value = _make_qkv_views(seq_len, torch.float16)
    value_before = value.clone()
    q_ref = _apply_qk_norm(norm, query)
    k_ref = _apply_qk_norm(norm, key)
    q_ref, k_ref = apply_rotary_pos_emb_qk(q_ref, k_ref, rotary)

    applied = fused_qknorm_rope_qk(query, key, rotary, norm_q=norm, norm_k=norm)

    assert applied
    assert torch.equal(query, q_ref)
    assert torch.equal(key, k_ref)
    assert torch.equal(value, value_before)


@requires_cuda
@torch.no_grad()
def test_fused_qknorm_rope_gate_falls_back_without_mutating(monkeypatch):
    """Rejected inputs must fall through untouched: the fused kernel computes
    an RMS norm, so LayerNorm or affine RMSNorm inputs would be silently
    mis-normalized if the gate ever let them in."""
    torch.manual_seed(0)
    rotary = _make_rotary_pos_emb(torch.float16)
    seq_len = rotary[2].shape[0]

    def _assert_rejected(norm_q, norm_k, rotary_pos_emb=rotary):
        _, query, key, _ = _make_qkv_views(seq_len, torch.float16)
        q_before, k_before = query.clone(), key.clone()
        assert not fused_qknorm_rope_qk(
            query, key, rotary_pos_emb, norm_q=norm_q, norm_k=norm_k
        )
        assert torch.equal(query, q_before)
        assert torch.equal(key, k_before)

    weightless = _weightless_rms_norm()
    layer_norm = nn.LayerNorm(HEAD_DIM, eps=1e-5, elementwise_affine=False).to("cuda")
    affine = nn.RMSNorm(HEAD_DIM, eps=1e-5, elementwise_affine=True).to(
        device="cuda", dtype=torch.float16
    )

    _assert_rejected(layer_norm, layer_norm)
    _assert_rejected(affine, affine)
    # legacy (cos, sin) rotary tuple without the native cache
    _assert_rejected(weightless, weightless, rotary_pos_emb=rotary[:2])

    monkeypatch.setenv(_FUSED_QKNORM_ROPE_ENV, "0")
    _assert_rejected(weightless, weightless)


@requires_cuda
@torch.no_grad()
def test_attention_forward_fused_matches_eager(monkeypatch):
    """End-to-end wiring guard: Attention.forward with the fused chain must
    reproduce the eager chain bit-for-bit, including the in-place qkv views
    handed to the attention backend."""
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        maybe_init_distributed_environment_and_model_parallel,
        model_parallel_is_initialized,
    )
    from sglang.multimodal_gen.runtime.managers.forward_context import (
        set_forward_context,
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

    torch.manual_seed(0)
    module = Attention(
        heads=HEADS,
        dim_head=HEAD_DIM,
        qk_norm_type="rms_norm",
        qk_norm_affine=False,
        eps=1e-5,
    ).to(device="cuda", dtype=torch.float16)
    rotary = _make_rotary_pos_emb(torch.float16)
    seq_len = rotary[2].shape[0]
    hidden = torch.randn(
        1, seq_len, HEADS * HEAD_DIM, device="cuda", dtype=torch.float16
    )

    with set_forward_context(current_timestep=0, attn_metadata=None):
        monkeypatch.setenv(_FUSED_QKNORM_ROPE_ENV, "1")
        out_fused = module(hidden, rotary)
        monkeypatch.setenv(_FUSED_QKNORM_ROPE_ENV, "0")
        out_eager = module(hidden, rotary)

    assert torch.equal(out_fused, out_eager)
