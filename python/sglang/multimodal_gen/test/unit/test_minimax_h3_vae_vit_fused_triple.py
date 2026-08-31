# SPDX-License-Identifier: Apache-2.0
"""Bit-exactness and gating tests for the MiniMax-H3 VAE decoder fused
residual triple (residual add + RMSNorm + autocast cast, R3 fusion)."""

import os
import socket

import pytest
import torch
import torch.nn as nn

from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.vit_utils import (
    _FUSED_TRIPLE_ENV,
    fused_add_rmsnorm_cast_site,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)

# Production ViT decoder config at reduced depth; dim (heads * dim_head),
# norm types and eps match MiniMaxH3VideoVAE so the fused sites see the
# production dtype flow (fp32 trunk, fp16 autocast branches).
DECODER_KWARGS = dict(
    patch_size=16,
    patch_size_t=4,
    in_channels=24,
    out_channels=3,
    num_layers=3,
    heads=32,
    dim_head=64,
    norm_type="rms_norm",
    norm_affine=True,
    qk_norm_type="rms_norm",
    qk_norm_affine=False,
    ffn_activation_fn="silu",
    ffn_use_gated=True,
    rope_dim_ratio=0.75,
    rope_theta=100.0,
    bias=True,
    eps=1e-5,
    num_register_tokens=4,
)
TILE_SHAPE = (1, 24, 3, 8, 8)


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


def _forward_context():
    from sglang.multimodal_gen.runtime.managers.forward_context import (
        set_forward_context,
    )

    return set_forward_context(current_timestep=0, attn_metadata=None)


def _make_decoder():
    from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.vae_vit import (
        ViT3DDecoder,
    )

    torch.manual_seed(0)
    decoder = ViT3DDecoder(**DECODER_KWARGS).to("cuda")
    decoder.eval()
    decoder.prepare_autocast_linear_weights(torch.float16)
    return decoder


@requires_cuda
@torch.no_grad()
def test_fused_triple_decoder_bitwise_vs_eager_loop(monkeypatch):
    """The deferred-residual loop plus the fused triple kernel must reproduce
    the plain eager block loop bit-for-bit over a full tile forward: the
    kernel replicates aten's add + fp32 rms reduction + autocast cast, and
    deferring an add across a block boundary must not change its operands."""
    _ensure_test_runtime(monkeypatch)
    decoder = _make_decoder()
    x = torch.randn(
        TILE_SHAPE, device="cuda", generator=torch.Generator("cuda").manual_seed(3)
    )

    def run():
        with _forward_context(), torch.autocast("cuda", dtype=torch.float16):
            return decoder(x).clone()

    monkeypatch.setenv(_FUSED_TRIPLE_ENV, "0")
    reference = run()
    monkeypatch.setenv(_FUSED_TRIPLE_ENV, "1")
    fused = run()

    assert torch.equal(reference, fused)


@requires_cuda
def test_fused_triple_site_engages_and_matches_eager_chain():
    """Vacuity guard for the loop test above: at the production site shapes
    the fused site must actually apply (return non-None), update the residual
    in place, and match the eager triple exactly."""
    torch.manual_seed(1)
    dim = 2048
    residual = torch.randn(1, 1797, dim, device="cuda", dtype=torch.float32)
    branch = torch.randn(1, 1797, dim, device="cuda", dtype=torch.float16)
    norm = nn.RMSNorm(dim, eps=1e-5, elementwise_affine=True).to("cuda")
    with torch.no_grad():
        norm.weight.normal_()

    y_ref = residual + branch
    out_ref = norm(y_ref).to(torch.float16)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        out = fused_add_rmsnorm_cast_site(residual, branch, norm)

    assert out is not None, "fused triple did not engage at the production shape"
    assert out.dtype is torch.float16
    assert torch.equal(residual, y_ref)
    assert torch.equal(out, out_ref)


@requires_cuda
def test_fused_triple_site_rejects_non_autocast():
    """Outside autocast there is no downstream cast to absorb; the site must
    fall back to the eager chain (and leave the residual untouched)."""
    dim = 2048
    residual = torch.randn(1, 8, dim, device="cuda", dtype=torch.float32)
    reference = residual.clone()
    branch = torch.randn(1, 8, dim, device="cuda", dtype=torch.float16)
    norm = nn.RMSNorm(dim, eps=1e-5, elementwise_affine=True).to("cuda")

    with torch.inference_mode():
        out = fused_add_rmsnorm_cast_site(residual, branch, norm)

    assert out is None
    assert torch.equal(residual, reference)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
