# SPDX-License-Identifier: Apache-2.0
"""Parity and fail-loud tests for the MiniMax-H3 VAE encoder cudnn_conv path."""

import copy
import importlib.util
import os
import unittest
from unittest import mock

import torch
import torch.nn as nn

from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.conv import (
    BaseConv3d,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.norm import (
    TemporalIsolatedGroupNorm,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.vae_cnn import (
    Downsample3D,
    EncoderFCN3D,
    ResnetBlock3D,
)

_HAS_CUDA = torch.cuda.is_available()
_HAS_CUDNN_CONV = importlib.util.find_spec("cudnn_conv") is not None
_GPU_REASON = "requires CUDA and the cudnn_conv package"

# TF32 runs on both sides (eager F.conv3d and cudnn_conv), so parity is
# accumulation-order noise at TF32 scale, normalized by the reference peak.
_CONV_REL_TOL = 5e-3

_H3_CONV_KWARGS = dict(padding_mode="reflect", padding_mode_t=None, causal=True)


def _wrap_encoder(encoder: nn.Module) -> nn.Module:
    wrapper = nn.Module()
    wrapper.encoder = encoder
    return wrapper


def _install(module: nn.Module, *, channels_last: bool = True) -> nn.Module:
    from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_vae_cuda_opt import (
        install_minimax_h3_vae_encoder_cudnn_conv,
    )

    wrapper = _wrap_encoder(module)
    install_minimax_h3_vae_encoder_cudnn_conv(wrapper, channels_last=channels_last)
    # Keep the wrapper alive so the installed WeakSet entry does not free the
    # patched module while the test still uses it.
    module._sgl_test_wrapper = wrapper
    return module


def _max_rel_vs_peak(test: torch.Tensor, ref: torch.Tensor) -> float:
    diff = (test.float() - ref.float()).abs().max()
    peak = ref.float().abs().max().clamp_min(1e-12)
    return (diff / peak).item()


@unittest.skipUnless(_HAS_CUDA and _HAS_CUDNN_CONV, _GPU_REASON)
class TestNormSiluChannelsLast(unittest.TestCase):
    def test_matches_eager_norm_silu_and_keeps_layout(self):
        import torch.nn.functional as F

        from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_vae_cuda_opt import (
            _norm_silu_channels_last,
        )

        # The Triton kernel import happens inside install; trigger it once.
        _install(BaseConv3d(32, 32, kernel_size=3, padding=1, **_H3_CONV_KWARGS).cuda())
        torch.manual_seed(0)
        norm = TemporalIsolatedGroupNorm(32, 128, eps=1e-6, affine=True).cuda()
        x = torch.randn(2, 128, 5, 32, 48, device="cuda")
        x_cl = x.contiguous(memory_format=torch.channels_last_3d)
        with torch.inference_mode():
            out = _norm_silu_channels_last(norm, x_cl)
            ref = F.silu(norm(x), inplace=False)
        self.assertTrue(out.is_contiguous(memory_format=torch.channels_last_3d))
        torch.testing.assert_close(out.contiguous(), ref, atol=1e-5, rtol=1e-5)


@unittest.skipUnless(_HAS_CUDA and _HAS_CUDNN_CONV, _GPU_REASON)
class TestConvParity(unittest.TestCase):
    def _parity(self, eager_module, fused_module, x, *, channels_last=True):
        with torch.inference_mode():
            ref = eager_module(x.clone())
            out = fused_module(x.clone())
        self.assertEqual(out.shape, ref.shape)
        if channels_last:
            self.assertTrue(out.is_contiguous(memory_format=torch.channels_last_3d))
        rel = _max_rel_vs_peak(out, ref)
        self.assertLess(rel, _CONV_REL_TOL, f"max rel vs peak {rel}")

    def test_base_conv3d(self):
        for depth in (1, 5):
            for channels_last in (True, False):
                with self.subTest(depth=depth, channels_last=channels_last):
                    torch.manual_seed(0)
                    conv = BaseConv3d(
                        16, 32, kernel_size=3, padding=1, **_H3_CONV_KWARGS
                    ).cuda()
                    eager = copy.deepcopy(conv)
                    _install(conv, channels_last=channels_last)
                    x = torch.randn(1, 16, depth, 32, 48, device="cuda")
                    self._parity(eager, conv, x, channels_last=channels_last)

    def test_base_conv3d_temporal_pad_values(self):
        # conv_in carries per-channel constant temporal pad values once the
        # pixel norm is folded into its weights (weight_folds); the fused
        # path must inject them instead of the descriptor's zero pad.
        for depth in (1, 5):
            with self.subTest(depth=depth):
                torch.manual_seed(0)
                conv = BaseConv3d(3, 32, kernel_size=3, padding=1, **_H3_CONV_KWARGS)
                conv.temporal_pad_values = torch.tensor([0.485, 0.456, 0.406])
                conv = conv.cuda()
                eager = copy.deepcopy(conv)
                _install(conv)
                x = torch.rand(1, 3, depth, 32, 48, device="cuda")
                self._parity(eager, conv, x)

    def test_downsample3d_merged_asymmetric_pad(self):
        for time_stride in (1, 2):
            with self.subTest(time_stride=time_stride):
                torch.manual_seed(0)
                down = Downsample3D(
                    16, 16, time_stride=time_stride, space_stride=2, **_H3_CONV_KWARGS
                ).cuda()
                eager = copy.deepcopy(down)
                _install(down)
                x = torch.randn(1, 16, 5, 33, 47, device="cuda")
                self._parity(eager, down, x)

    def test_resnet_block(self):
        for in_ch, out_ch in ((32, 32), (32, 64)):
            with self.subTest(in_ch=in_ch, out_ch=out_ch):
                torch.manual_seed(0)
                block = ResnetBlock3D(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    zq_ch=None,
                    use_t_isolated_gn=True,
                    **_H3_CONV_KWARGS,
                ).cuda()
                eager = copy.deepcopy(block)
                _install(block)
                x = torch.randn(1, in_ch, 5, 32, 32, device="cuda")
                self._parity(eager, block, x)

    def test_small_encoder_end_to_end(self):
        torch.manual_seed(0)
        kwargs = dict(
            ch=32,
            ch_mult=[1, 2],
            space_down=[2, 1],
            time_down=[1, 2],
            num_res_blocks=2,
            in_channels=3,
            z_channels=8,
            double_z=True,
            zq_ch=None,
            padding_mode="reflect",
            padding_mode_t=None,
            causal=True,
            use_t_isolated_gn=True,
        )
        encoder = EncoderFCN3D(**kwargs).cuda()
        eager = copy.deepcopy(encoder)
        _install(encoder)
        for depth in (1, 5):
            with self.subTest(depth=depth):
                x = torch.randn(1, 3, depth, 64, 96, device="cuda")
                with torch.inference_mode():
                    ref = eager(x)
                    out = encoder(x)
                # Boundary contract: the tile-parallel all-gather requires
                # standard-contiguous moments, exactly what eager returns.
                self.assertTrue(out.is_contiguous())
                rel = _max_rel_vs_peak(out, ref)
                self.assertLess(rel, _CONV_REL_TOL, f"max rel vs peak {rel}")


@unittest.skipUnless(_HAS_CUDA and _HAS_CUDNN_CONV, _GPU_REASON)
class TestFailLoud(unittest.TestCase):
    def test_unsupported_padding_mode_raises_before_patching(self):
        conv = BaseConv3d(
            8, 8, kernel_size=3, padding=1, padding_mode="zeros", causal=True
        ).cuda()
        with self.assertRaisesRegex(ValueError, "pad_mode"):
            _install(conv)
        self.assertNotIn("forward", conv.__dict__)

    def test_fused_norm_env_rejected(self):
        conv = BaseConv3d(8, 8, kernel_size=3, padding=1, **_H3_CONV_KWARGS).cuda()
        with mock.patch.dict("os.environ", {"MINIMAX_H3_USE_FUSED_NORM": "true"}):
            with self.assertRaisesRegex(ValueError, "MINIMAX_H3_USE_FUSED_NORM"):
                _install(conv)

    def test_conditioned_forward_rejected(self):
        torch.manual_seed(0)
        block = ResnetBlock3D(
            in_channels=32, zq_ch=None, use_t_isolated_gn=True, **_H3_CONV_KWARGS
        ).cuda()
        _install(block)
        x = torch.randn(1, 32, 3, 16, 16, device="cuda")
        with self.assertRaisesRegex(ValueError, "zq"):
            block(x, zq=x)


class TestFlagGate(unittest.TestCase):
    def test_flag_off_returns_module_unchanged(self):
        from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_vae_cuda_opt import (
            ENV_FLAG,
            maybe_optimize_minimax_h3_vae_encoder,
        )

        module = nn.Module()
        with mock.patch.dict("os.environ", {ENV_FLAG: "0"}):
            self.assertIs(maybe_optimize_minimax_h3_vae_encoder(module), module)


def _tiny_legacy_vae() -> nn.Module:
    """A small AutoencoderKLLegacy with the H3 encoder structure invariants
    (reflect spatial pad, causal temporal pad, TemporalIsolatedGroupNorm)."""
    from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae import (
        AutoencoderKLLegacy,
    )

    torch.manual_seed(0)
    vae = AutoencoderKLLegacy(
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
        space_down=[2, 1],
        space_up=[1, 2],
        time_down=[1, 2],
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
    )
    # encode/decode require weights loaded through load_state_dict so the
    # weight folds run; round-trip the random init as a synthetic checkpoint.
    vae.load_state_dict(
        {key: value.clone() for key, value in vae.state_dict().items()}
    )
    return vae


class TestOptimizeVaeWiring(unittest.TestCase):
    """The platform optimize_vae hook must install the fast path by default
    (env unset), honor the 0/1 overrides, and never break load on failure."""

    def _opt(self):
        from sglang.multimodal_gen.runtime.models.vaes import (
            minimax_h3_vae_cuda_opt as opt,
        )

        return opt

    def _assert_installed(self, vae: nn.Module, installed: bool):
        opt = self._opt()
        if installed:
            self.assertIs(vae.encoder.forward.__func__, opt._fused_encoder_forward)
            self.assertIs(vae.encoder.conv_in.forward.__func__, opt._fused_conv_forward)
        else:
            self.assertNotIn("forward", vae.encoder.__dict__)
            self.assertNotIn("forward", vae.encoder.conv_in.__dict__)

    @unittest.skipUnless(_HAS_CUDA and _HAS_CUDNN_CONV, _GPU_REASON)
    def test_env_unset_installs_through_platform_hook(self):
        from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_vae_cuda_opt import (
            ENV_FLAG,
        )
        from sglang.multimodal_gen.runtime.platforms.cuda import CudaPlatformBase

        vae = _tiny_legacy_vae().cuda()
        with mock.patch.dict("os.environ"):
            os.environ.pop(ENV_FLAG, None)
            out = CudaPlatformBase.optimize_vae(vae)
        self.assertIs(out, vae)
        self._assert_installed(vae, True)
        with torch.inference_mode():
            moments = vae.encode(torch.randn(1, 3, 5, 32, 32, device="cuda"))
        self.assertTrue(moments.is_contiguous())

    def test_env_zero_keeps_eager(self):
        opt = self._opt()
        vae = _tiny_legacy_vae()
        with mock.patch.dict("os.environ", {opt.ENV_FLAG: "0"}):
            out = opt.maybe_optimize_minimax_h3_vae_encoder(vae)
        self.assertIs(out, vae)
        self._assert_installed(vae, False)

    def test_env_unset_falls_back_when_kernels_unavailable(self):
        opt = self._opt()
        vae = _tiny_legacy_vae()
        with (
            mock.patch.dict("os.environ"),
            mock.patch.object(opt, "_import_kernels", side_effect=RuntimeError("boom")),
        ):
            os.environ.pop(opt.ENV_FLAG, None)
            out = opt.maybe_optimize_minimax_h3_vae_encoder(vae)
        self.assertIs(out, vae)
        self._assert_installed(vae, False)

    def test_env_one_fails_loud_when_kernels_unavailable(self):
        opt = self._opt()
        vae = _tiny_legacy_vae()
        with (
            mock.patch.dict("os.environ", {opt.ENV_FLAG: "1"}),
            mock.patch.object(opt, "_import_kernels", side_effect=RuntimeError("boom")),
        ):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                opt.maybe_optimize_minimax_h3_vae_encoder(vae)

    @unittest.skipUnless(_HAS_CUDA and _HAS_CUDNN_CONV, _GPU_REASON)
    def test_env_unset_rolls_back_on_warmup_failure(self):
        opt = self._opt()
        vae = _tiny_legacy_vae().cuda()
        with (
            mock.patch.dict("os.environ"),
            mock.patch.object(
                opt, "_warmup_plan_cache", side_effect=RuntimeError("boom")
            ),
        ):
            os.environ.pop(opt.ENV_FLAG, None)
            out = opt.maybe_optimize_minimax_h3_vae_encoder(vae)
        self.assertIs(out, vae)
        self._assert_installed(vae, False)
        # The rolled-back encoder must still run the eager path.
        with torch.inference_mode():
            moments = vae.encode(torch.randn(1, 3, 5, 32, 32, device="cuda"))
        self.assertTrue(moments.is_contiguous())


if __name__ == "__main__":
    unittest.main()
