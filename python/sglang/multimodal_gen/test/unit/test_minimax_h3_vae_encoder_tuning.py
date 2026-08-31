# SPDX-License-Identifier: Apache-2.0
"""Tests for the MiniMax-H3 VAE encoder tuning gates (E2/E3/E4).

- MINIMAX_H3_VAE_ENCODER_BF16: quality-gated bf16 encode weights (fp32 accum)
- MINIMAX_H3_VAE_ENCODER_TILE_SIZE: encoder-only tile size knob
- MINIMAX_H3_VAE_ENCODER_STACK_TILING: batched per-tile encode forwards

All three default OFF/unchanged; the flag-off paths must be bit-identical to
the pre-gate behavior.
"""

import contextlib
import copy
import importlib.util
import os
import unittest
from unittest import mock

import torch

_HAS_CUDA = torch.cuda.is_available()
_HAS_CUDNN_CONV = importlib.util.find_spec("cudnn_conv") is not None

BF16_ENV = "MINIMAX_H3_VAE_ENCODER_BF16"
TILE_ENV = "MINIMAX_H3_VAE_ENCODER_TILE_SIZE"
STACK_ENV = "MINIMAX_H3_VAE_ENCODER_STACK_TILING"
_ALL_ENVS = (BF16_ENV, TILE_ENV, STACK_ENV, "MINIMAX_H3_VAE_ENCODER_CUDNN_CONV")


def _tiny_legacy_vae(**overrides) -> torch.nn.Module:
    """Small AutoencoderKLLegacy with the H3 encoder structure invariants."""
    from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae import (
        AutoencoderKLLegacy,
    )

    torch.manual_seed(0)
    kwargs = dict(
        in_channels=3,
        out_ch=3,
        ch=32,
        embed_dim=8,
        z_channels=8,
        use_3d_conv=True,
        zq_ch_encoder=None,
        zq_ch_decoder=None,
        num_res_blocks=1,
        ch_mult=[1, 2],
        # Real-like: downsampling levels pair space and time strides, so a
        # pixel tile maps to exactly tile/vae_ratio latents (the H3 config
        # never has a space_stride=1 Downsample3D, which would shrink
        # spatially without padding and make canvas shape depend on tiling).
        space_down=[2, 1],
        space_up=[1, 2],
        time_down=[2, 1],
        padding_mode="reflect",
        padding_mode_t=None,
        use_t_isolated_gn=True,
        causal_encoder=True,
        causal_decoder=False,
        use_vit_decoder=True,
        vit_decoder_kwargs={
            "dim_head": 16,
            "heads": 2,
            "num_layers": 1,
            "norm_type": "rms_norm",
            "norm_affine": True,
            "qk_norm_type": "rms_norm",
            "qk_norm_affine": False,
            "ffn_activation_fn": "silu",
            "ffn_use_gated": True,
            "rope_dim_ratio": 0.75,
            "rope_theta": 100.0,
        },
        clip_length=5,
        tile_size=32,
        # The config default (64) exceeds the test tile size; split_tiles
        # requires overlap < tile.
        tile_overlap_min=16,
    )
    kwargs.update(overrides)
    vae = AutoencoderKLLegacy(**kwargs)
    # encode requires weights loaded through load_state_dict so the weight
    # folds run; round-trip the random init as a synthetic checkpoint.
    vae.load_state_dict({key: value.clone() for key, value in vae.state_dict().items()})
    return vae


@contextlib.contextmanager
def _clean_env(**values):
    with mock.patch.dict("os.environ"):
        for name in _ALL_ENVS:
            os.environ.pop(name, None)
        os.environ.update(values)
        yield


def _max_rel_vs_peak(test: torch.Tensor, ref: torch.Tensor) -> float:
    diff = (test.float() - ref.float()).abs().max()
    peak = ref.float().abs().max().clamp_min(1e-12)
    return (diff / peak).item()


@contextlib.contextmanager
def _tf32(enabled: bool):
    prev_conv = torch.backends.cudnn.allow_tf32
    prev_mm = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = enabled
    torch.backends.cuda.matmul.allow_tf32 = enabled
    try:
        yield
    finally:
        torch.backends.cudnn.allow_tf32 = prev_conv
        torch.backends.cuda.matmul.allow_tf32 = prev_mm


class TestEncoderTileSizeKnob(unittest.TestCase):
    """E3: encoder-only tile size, decoupled from the decoder tile size."""

    def test_default_unchanged(self):
        with _clean_env():
            vae = _tiny_legacy_vae()
        self.assertEqual(vae.tile_size, 32)
        self.assertEqual(vae.decoder_tile_size, 32)
        self.assertFalse(vae.encoder_stack_tiling)

    def test_env_overrides_encoder_only(self):
        # Tiny VAE vae_ratio is 2 (space_down [2, 1]).
        with _clean_env(**{TILE_ENV: "48"}):
            vae = _tiny_legacy_vae()
        self.assertEqual(vae.tile_size, 48)
        self.assertEqual(vae.decoder_tile_size, 32)
        # Encoder split uses the override, decoder split keeps the config.
        _, enc_len, _ = vae.split_tiles(100, is_decoder=False)
        _, dec_len, _ = vae.split_tiles(100, is_decoder=True)
        self.assertEqual(enc_len[0], 48)
        self.assertEqual(dec_len[0], 32)

    def test_env_rejects_bad_values(self):
        for raw in ("abc", "-16", "47"):
            with self.subTest(raw=raw), _clean_env(**{TILE_ENV: raw}):
                with self.assertRaises(ValueError):
                    _tiny_legacy_vae()

    def test_env_zero_means_default(self):
        with _clean_env(**{TILE_ENV: "0"}):
            vae = _tiny_legacy_vae()
        self.assertEqual(vae.tile_size, 32)

    @unittest.skipUnless(_HAS_CUDA, "requires CUDA")
    def test_larger_tile_encodes_same_canvas(self):
        """Tile 64 vs 32 must produce the same latent canvas geometry; the
        values legitimately move everywhere because GroupNorm statistics span
        the whole tile (encode is tile-global, not receptive-field-local), so
        the divergence is reported, not asserted -- that is exactly why the
        knob is quality-gated with the default left at the config value."""
        with _clean_env():
            vae = _tiny_legacy_vae().cuda().eval()
        x = torch.rand(1, 3, 5, 64, 96, device="cuda")

        with _tf32(False), torch.inference_mode():
            z_small = vae.tiled_encode(x).float()
            with _clean_env(**{TILE_ENV: "64"}):
                vae.setup_forward(clip_length=5, tile_size=32, tile_overlap_min=16)
            self.assertEqual(vae.tile_size, 64)
            self.assertEqual(vae.decoder_tile_size, 32)
            z_large = vae.tiled_encode(x).float()

        self.assertEqual(z_small.shape, z_large.shape)
        self.assertTrue(torch.isfinite(z_large).all())
        global_rel = _max_rel_vs_peak(z_large, z_small)
        cos = torch.nn.functional.cosine_similarity(
            z_large.flatten(), z_small.flatten(), dim=0
        ).item()
        print(
            f"[E3] tile 64 vs 32: global_rel={global_rel:.3e} cosine={cos:.6f} "
            f"(GroupNorm tile statistics change, quality-gated)"
        )


class TestEncoderStackTiling(unittest.TestCase):
    """E4: batched per-tile encode forwards behind an env flag."""

    def test_env_flag_sets_encoder_only(self):
        with _clean_env(**{STACK_ENV: "1"}):
            vae = _tiny_legacy_vae()
        self.assertTrue(vae.encoder_stack_tiling)
        # decode tiling keeps reading the config value.
        self.assertFalse(vae.stack_tiling)

    def test_config_stack_tiling_still_covers_encode(self):
        with _clean_env():
            vae = _tiny_legacy_vae(stack_tiling=True)
        self.assertTrue(vae.stack_tiling)
        self.assertTrue(vae.encoder_stack_tiling)

    @unittest.skipUnless(_HAS_CUDA, "requires CUDA")
    def test_stacked_matches_sequential(self):
        with _clean_env():
            vae = _tiny_legacy_vae().cuda().eval()
        x = torch.rand(1, 3, 5, 64, 96, device="cuda")

        with torch.inference_mode():
            with _tf32(False):
                vae.encoder_stack_tiling = False
                z_seq = vae.tiled_encode(x).float()
                vae.encoder_stack_tiling = True
                z_stack = vae.tiled_encode(x).float()
            with _tf32(True):
                vae.encoder_stack_tiling = False
                z_seq_tf32 = vae.tiled_encode(x).float()
                vae.encoder_stack_tiling = True
                z_stack_tf32 = vae.tiled_encode(x).float()
        vae.encoder_stack_tiling = False

        self.assertEqual(z_stack.shape, z_seq.shape)
        rel = _max_rel_vs_peak(z_stack, z_seq)
        rel_tf32 = _max_rel_vs_peak(z_stack_tf32, z_seq_tf32)
        # Batch-dim change only alters conv/GEMM kernel selection; with TF32
        # off this is fp32 accumulation-order noise (~ulp scale).
        self.assertLess(rel, 1e-5, f"stacked vs sequential max rel {rel}")
        self.assertLess(rel_tf32, 5e-3, f"stacked vs sequential (TF32) {rel_tf32}")
        print(f"[E4] stacked vs sequential: fp32_rel={rel:.3e} tf32_rel={rel_tf32:.3e}")


class TestEncoderBf16(unittest.TestCase):
    """E2: quality-gated bf16 encode weights with fp32 statistics."""

    def test_flag_off_is_a_no_op(self):
        from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.encoder_precision import (
            maybe_cast_minimax_h3_vae_encoder_bf16,
        )

        with _clean_env():
            vae = _tiny_legacy_vae()
            x = torch.rand(1, 3, 5, 32, 32)
            with torch.inference_mode():
                before = vae.encode(x).clone()
                self.assertFalse(maybe_cast_minimax_h3_vae_encoder_bf16(vae))
                after = vae.encode(x)
        self.assertEqual(next(vae.encoder.parameters()).dtype, torch.float32)
        self.assertTrue(torch.equal(before, after))

    def test_cast_is_encoder_only_and_idempotent(self):
        from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.encoder_precision import (
            maybe_cast_minimax_h3_vae_encoder_bf16,
        )

        with _clean_env(**{BF16_ENV: "1"}):
            vae = _tiny_legacy_vae()
            self.assertTrue(maybe_cast_minimax_h3_vae_encoder_bf16(vae))
            self.assertFalse(maybe_cast_minimax_h3_vae_encoder_bf16(vae))
        self.assertEqual(next(vae.encoder.parameters()).dtype, torch.bfloat16)
        self.assertEqual(next(vae.decoder.parameters()).dtype, torch.float32)

    def test_scoped_encode_dtype_respects_bf16_gate(self):
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.keyframe_encoding import (
            minimax_h3_scoped_encode_dtype,
        )

        with _clean_env(**{BF16_ENV: "1"}):
            vae = _tiny_legacy_vae()
            with minimax_h3_scoped_encode_dtype(vae):
                self.assertEqual(next(vae.encoder.parameters()).dtype, torch.bfloat16)
                self.assertEqual(next(vae.decoder.parameters()).dtype, torch.float32)
            # One-time cast: no restore on exit.
            self.assertEqual(next(vae.encoder.parameters()).dtype, torch.bfloat16)

    def test_scoped_encode_dtype_default_pins_fp32(self):
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.keyframe_encoding import (
            minimax_h3_scoped_encode_dtype,
        )

        with _clean_env():
            vae = _tiny_legacy_vae().to(torch.float16)
            with minimax_h3_scoped_encode_dtype(vae):
                self.assertEqual(next(vae.encoder.parameters()).dtype, torch.float32)
            self.assertEqual(next(vae.encoder.parameters()).dtype, torch.float16)

    @unittest.skipUnless(_HAS_CUDA, "requires CUDA")
    def test_bf16_encode_close_to_fp32(self):
        from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.encoder_precision import (
            maybe_cast_minimax_h3_vae_encoder_bf16,
        )

        with _clean_env():
            vae = _tiny_legacy_vae().cuda().eval()
        vae_bf16 = copy.deepcopy(vae)
        with _clean_env(**{BF16_ENV: "1"}):
            self.assertTrue(maybe_cast_minimax_h3_vae_encoder_bf16(vae_bf16))

        x = torch.rand(1, 3, 5, 64, 96, device="cuda")
        with torch.inference_mode(), _tf32(True):
            z_fp32 = vae.tiled_encode(x).float()
            z_bf16 = vae_bf16.tiled_encode(x).float()

        self.assertEqual(z_bf16.shape, z_fp32.shape)
        self.assertTrue(torch.isfinite(z_bf16).all())
        rel = _max_rel_vs_peak(z_bf16, z_fp32)
        # Quality-gated: no exact contract. Random weights measure ~1.1e-2 vs
        # peak on this tiny net (bf16 rounding compounding through GroupNorm
        # tile statistics); the bound only catches order-of-magnitude bugs.
        self.assertLess(rel, 3e-2, f"bf16 vs fp32 max rel vs peak {rel}")
        print(f"[E2] bf16 vs fp32 tiled_encode: max_rel_vs_peak={rel:.3e}")

    @unittest.skipUnless(_HAS_CUDA and _HAS_CUDNN_CONV, "requires cudnn_conv")
    def test_bf16_composes_with_cudnn_fast_path(self):
        from sglang.multimodal_gen.runtime.models.vaes import (
            minimax_h3_vae_cuda_opt as opt,
        )

        with _clean_env():
            vae = _tiny_legacy_vae().cuda().eval()
        eager_bf16 = copy.deepcopy(vae)
        eager_bf16.encoder.to(torch.bfloat16)

        with _clean_env(**{BF16_ENV: "1"}):
            out = opt.maybe_optimize_minimax_h3_vae_encoder(vae)
        self.assertIs(out, vae)
        self.assertEqual(next(vae.encoder.parameters()).dtype, torch.bfloat16)
        self.assertIs(vae.encoder.forward.__func__, opt._fused_encoder_forward)

        x = torch.rand(1, 3, 5, 32, 32, device="cuda")
        with torch.inference_mode():
            fused = vae.encode(x).float()
            ref = eager_bf16.encode(x).float()
        self.assertTrue(fused.is_contiguous())
        rel = _max_rel_vs_peak(fused, ref)
        # Both sides run bf16 io with fp32 accumulation; engine differences
        # land at bf16 output-rounding scale.
        self.assertLess(rel, 2e-2, f"cudnn bf16 vs eager bf16 max rel {rel}")
        print(f"[E2] cudnn bf16 vs eager bf16: max_rel_vs_peak={rel:.3e}")


if __name__ == "__main__":
    unittest.main()
