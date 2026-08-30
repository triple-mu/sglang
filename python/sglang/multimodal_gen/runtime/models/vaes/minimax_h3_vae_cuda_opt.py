# SPDX-License-Identifier: Apache-2.0
"""cudnn_conv fast path for the MiniMax H3 video VAE encoder.

Replaces each ``BaseConv3d``'s ``F.pad(reflect) + F.pad(causal zeros) +
F.conv3d`` sequence with a fused cudnn-frontend prologue + convolution: the
reflect spatial pad becomes one fused pad kernel, the causal temporal zero pad
folds into the convolution's pre-padding, and bias plus the ``ResnetBlock3D``
residual add run in the conv epilogue. Encoder conv weights are converted to
channels_last_3d, and every GroupNorm+SiLU site runs the in-tree Triton
``group_norm_silu_4d`` kernel on a zero-copy 4D channels_last view. This
removes both of ``TemporalIsolatedGroupNorm``'s permute+contiguous copies per
call; aten ``F.group_norm`` cannot be used here because it returns a
standard-contiguous tensor for channels_last input, which would force a
layout round trip at every norm.

The CUDA platform's ``optimize_vae`` hook calls
``maybe_optimize_minimax_h3_vae_encoder`` at load time.
``MINIMAX_H3_VAE_ENCODER_CUDNN_CONV`` selects the gate mode:

- unset / "auto" (default): install when cudnn_conv is available and the
  encoder validates; on any failure roll back and run the eager encoder.
- "1": fail-loud -- a missing cudnn_conv package, a cuDNN version mismatch,
  or an encoder whose structure deviates from the released H3 checkpoint
  raises instead of silently running eager.
- "0": keep the eager encoder.

Install is all-or-nothing; validation walks the full module tree before the
first patch. Validation is structural only, so load-time weight-value folds
(e.g. quant_conv folded into conv_out, pixel normalize folded into conv_in)
stay compatible -- but they must run BEFORE install so the channels_last_3d
weight conversion applies to the folded weights.

NOTE: cudnn-frontend execution plans ignore ``torch.backends.cudnn.enabled``,
``.deterministic``, and ``.benchmark``. No determinism context currently wraps
the video encode (the audio-scoped ones do not reach it); if one is ever added
around visual encoding, this fast path would keep running cuDNN.
"""

import weakref
from types import MethodType

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.utils import logging

from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.conv import (
    BaseConv3d,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.norm import (
    FusedGroupNorm3D,
    SpatialNorm3D,
    TemporalIsolatedGroupNorm,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.vae_cnn import (
    Downsample3D,
    EncoderFCN3D,
    ResnetBlock3D,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.vit_utils import (
    _env_flag,
    _env_optional_bool,
)

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

ENV_FLAG = "MINIMAX_H3_VAE_ENCODER_CUDNN_CONV"

_cudnn_conv = None
_gn_silu_4d = None
_installed: "weakref.WeakSet[nn.Module]" = weakref.WeakSet()

_SUPPORTED_PADDINGS = {(0, 0, 0), (1, 1, 1), (1, 0, 0)}
# group_norm_silu_4d needs a power-of-two channel count of at most 2048.
_GN_MAX_CHANNELS = 2048


def _import_kernels():
    try:
        import cudnn_conv
    except ImportError as e:
        raise RuntimeError(
            f"{ENV_FLAG} fast path requires the cudnn_conv package. "
            "Install it with: pip install nvidia-cudnn-frontend && "
            "pip install -e <cudnn_conv checkout> --no-build-isolation"
        ) from e
    build = cudnn_conv.cudnn_build_version()
    runtime = cudnn_conv.cudnn_runtime_version()
    torch_version = torch.backends.cudnn.version()
    if not build == runtime == torch_version:
        raise RuntimeError(
            f"cuDNN version mismatch: cudnn_conv build={build}, "
            f"runtime={runtime}, torch={torch_version}; all three must be equal"
        )
    from sglang.kernels.ops.diffusion import group_norm_silu_4d

    return cudnn_conv, group_norm_silu_4d


def _validate_encoder(encoder: nn.Module) -> None:
    if _env_flag("MINIMAX_H3_USE_FUSED_NORM"):
        raise ValueError(
            f"{ENV_FLAG} does not support MINIMAX_H3_USE_FUSED_NORM=true; "
            "disable one of the two flags"
        )
    conv_count = 0
    for name, m in encoder.named_modules():
        if isinstance(m, (SpatialNorm3D, FusedGroupNorm3D)):
            raise ValueError(
                f"{ENV_FLAG} expects unconditioned TemporalIsolatedGroupNorm "
                f"norms, got {type(m).__name__} at encoder.{name}"
            )
        if isinstance(m, nn.GroupNorm):
            if type(m) is not TemporalIsolatedGroupNorm:
                raise ValueError(
                    f"{ENV_FLAG} expects TemporalIsolatedGroupNorm, got "
                    f"{type(m).__name__} at encoder.{name}"
                )
            if m.weight is None or m.bias is None:
                raise ValueError(
                    f"{ENV_FLAG} expects affine norms, got affine=False at "
                    f"encoder.{name}"
                )
            c = int(m.num_channels)
            if c % int(m.num_groups) or (c & (c - 1)) or c > _GN_MAX_CHANNELS:
                raise ValueError(
                    f"{ENV_FLAG} norm at encoder.{name} outside "
                    f"group_norm_silu_4d support: expected power-of-two "
                    f"channels <= {_GN_MAX_CHANNELS} divisible by "
                    f"{m.num_groups}, got {c}"
                )
        if isinstance(m, ResnetBlock3D) and m.use_fused_norm:
            raise ValueError(
                f"{ENV_FLAG} does not support use_fused_norm at encoder.{name}"
            )
        if not isinstance(m, BaseConv3d):
            continue
        conv_count += 1
        padding = tuple(int(p) for p in m.padding)
        checks = [
            ("causal", m.causal, True),
            ("pad_mode_t", m.pad_mode_t, "constant"),
            ("groups", m.groups, 1),
            ("dilation", tuple(m.dilation), (1, 1, 1)),
        ]
        for field, got, expected in checks:
            if got != expected:
                raise ValueError(
                    f"{ENV_FLAG} unsupported conv at encoder.{name}: "
                    f"{field} expected {expected!r}, got {got!r}"
                )
        if padding not in _SUPPORTED_PADDINGS:
            raise ValueError(
                f"{ENV_FLAG} unsupported conv at encoder.{name}: padding "
                f"expected one of {sorted(_SUPPORTED_PADDINGS)}, got {padding}"
            )
        if (padding[1] or padding[2]) and m.pad_mode != "reflect":
            raise ValueError(
                f"{ENV_FLAG} unsupported conv at encoder.{name}: spatial "
                f"pad_mode expected 'reflect', got {m.pad_mode!r}"
            )
        # The fused path folds the causal pad as (k-1, 0), which equals the
        # eager (2*p, 0) only when k == 2*p + 1.
        if padding[0] and int(m.kernel_size[0]) != 2 * padding[0] + 1:
            raise ValueError(
                f"{ENV_FLAG} unsupported conv at encoder.{name}: temporal "
                f"kernel expected {2 * padding[0] + 1}, got {m.kernel_size[0]}"
            )
    if conv_count == 0:
        raise ValueError(f"{ENV_FLAG} found no BaseConv3d modules in the encoder")


def _packed(x: torch.Tensor, channels_last: bool) -> torch.Tensor:
    fmt = torch.channels_last_3d if channels_last else torch.contiguous_format
    if x.is_contiguous(memory_format=fmt):
        return x
    return x.contiguous(memory_format=fmt)


def _run_fused_conv(
    conv: BaseConv3d,
    x: torch.Tensor,
    *,
    residual: torch.Tensor | None = None,
    extra_spatial: tuple[tuple[int, int], tuple[int, int]] | None = None,
) -> torch.Tensor:
    x = _packed(x, conv._sgl_channels_last)
    temporal_pre = conv._sgl_temporal_pre
    if temporal_pre and conv.temporal_pad_values is not None:
        # conv_in consumes raw pixels once the pixel norm is folded into its
        # weights (weight_folds); its constant temporal pad injects the
        # per-channel mean, which the conv descriptor's zero pad cannot
        # express. Materialize it here; conv_in is 3-channel, the cat is cheap.
        batch, channels, _, height, width = x.shape
        values = conv.temporal_pad_values.to(dtype=x.dtype).view(1, -1, 1, 1, 1)
        pad_block = values.expand(batch, channels, temporal_pre, height, width)
        x = _packed(torch.cat([pad_block, x], dim=2), conv._sgl_channels_last)
        temporal_pre = 0
    spatial = conv._sgl_spatial_pad
    if extra_spatial is not None:
        base = spatial if spatial is not None else ((0, 0), (0, 0))
        spatial = (
            (base[0][0] + extra_spatial[0][0], base[0][1] + extra_spatial[0][1]),
            (base[1][0] + extra_spatial[1][0], base[1][1] + extra_spatial[1][1]),
        )
    if residual is not None:
        residual = _packed(residual, conv._sgl_channels_last)
    if spatial is None:
        padding = ((temporal_pre, 0), (0, 0), (0, 0))
        pad_mode = "zeros"
    else:
        # Per-axis modes: the causal temporal zero padding folds into the conv
        # descriptor while the reflect spatial axes go through one prologue pass.
        padding = ((temporal_pre, 0), spatial[0], spatial[1])
        pad_mode = ("zeros", "reflect", "reflect")
    return _cudnn_conv.conv3d(
        x,
        conv.weight,
        conv.bias,
        stride=conv.stride,
        padding=padding,
        pad_mode=pad_mode,
        residual=residual,
    )


def _norm_silu_channels_last(norm: nn.GroupNorm, x: torch.Tensor) -> torch.Tensor:
    # For a channels_last_3d [B,C,D,H,W] tensor, folding D into the batch is a
    # pure view whose strides are exactly 4D channels_last, so the eager
    # permute+contiguous round trip (two full copies) disappears. The Triton
    # kernel preserves the channels_last layout; aten F.group_norm does not.
    x = _packed(x, True)
    b, c, d, h, w = x.shape
    x4 = x.transpose(1, 2).reshape(b * d, c, h, w)
    y4 = _gn_silu_4d(x4, norm.weight, norm.bias, norm.num_groups, norm.eps, True)
    return y4.reshape(b, d, c, h, w).transpose(1, 2)


def _norm_silu(norm: nn.GroupNorm, x: torch.Tensor, channels_last: bool):
    if channels_last:
        return _norm_silu_channels_last(norm, x)
    return F.silu(norm(x), inplace=True)


def _fused_conv_forward(self, x):
    return _run_fused_conv(self, x)


def _fused_downsample_forward(self, x):
    extra = ((0, 1), (0, 1)) if self.space_stride == 2 else None
    return _run_fused_conv(self.conv, x, extra_spatial=extra)


def _fused_resnet_forward(self, x, zq=None):
    if zq is not None:
        raise ValueError(f"{ENV_FLAG} does not support conditioned norms (zq)")
    channels_last = self._sgl_channels_last
    if self.in_channels != self.out_channels:
        shortcut = _run_fused_conv(self.nin_shortcut, x)
    else:
        shortcut = x
    h = _norm_silu(self.norm1, x, channels_last)
    h = _run_fused_conv(self.conv1, h)
    h = _norm_silu(self.norm2, h, channels_last)
    return _run_fused_conv(self.conv2, h, residual=shortcut)


def _fused_encoder_forward(self, x, zq=None):
    if zq is not None:
        raise ValueError(f"{ENV_FLAG} does not support conditioned norms (zq)")
    h = self.conv_in(x)
    for i_level in range(self.num_levels):
        for i_block in range(self.num_res_blocks[i_level]):
            h = self.down[i_level].block[i_block](h)
        if hasattr(self.down[i_level], "downsample"):
            h = self.down[i_level].downsample(h)
    h = _norm_silu(self.norm_out, h, self._sgl_channels_last)
    # Restore the eager layout contract at the module boundary: callers (the
    # tile-parallel all-gather in particular) require standard-contiguous
    # moments. The tile is small, so the copy is negligible.
    return self.conv_out(h).contiguous()


def _warmup_plan_cache(vae: nn.Module) -> None:
    """Build and autotune the encoder's cudnn plans at install time.

    The H3 tiled encode sees exactly two temporal extents (1 for keyframe
    images, clip_length for video clips) at the fixed tile size, so two dummy
    forwards cover the whole plan cache. Without this, the one-time autotune
    cost (several seconds) lands inside the first request's encode stage.
    """
    encoder = vae.encoder
    parameter = next(encoder.parameters())
    if parameter.device.type != "cuda":
        logger.info(
            "MiniMax-H3 VAE encoder cudnn_conv warmup skipped (encoder on %s); "
            "plans build on first encode.",
            parameter.device,
        )
        return
    import time

    tile = int(vae.tile_size)
    start = time.perf_counter()
    with torch.inference_mode():
        for depth in (1, int(vae.clip_length)):
            dummy = torch.zeros(
                1,
                3,
                depth,
                tile,
                tile,
                device=parameter.device,
                dtype=parameter.dtype,
            )
            encoder(dummy)
    torch.cuda.synchronize(parameter.device)
    logger.info(
        "MiniMax-H3 VAE encoder cudnn_conv warmup built %d plans in %.1f s.",
        _cudnn_conv.plan_cache_size(),
        time.perf_counter() - start,
    )


def install_minimax_h3_vae_encoder_cudnn_conv(
    vae: nn.Module, *, channels_last: bool = True
) -> None:
    """Patch the encoder of an AutoencoderKLLegacy-style VAE in place.

    Raises if cudnn_conv is unavailable, the cuDNN versions disagree, or the
    encoder structure differs from the released H3 checkpoint. Safe to call
    twice on the same VAE (second call is a no-op).
    """
    global _cudnn_conv, _gn_silu_4d
    if vae in _installed:
        return
    _cudnn_conv, _gn_silu_4d = _import_kernels()
    encoder = vae.encoder
    _validate_encoder(encoder)

    n_convs = n_blocks = n_downsamples = n_encoders = 0
    for m in encoder.modules():
        if isinstance(m, BaseConv3d):
            m._sgl_channels_last = channels_last
            m._sgl_temporal_pre = 2 if m.padding[0] else 0
            p1, p2 = int(m.padding[1]), int(m.padding[2])
            m._sgl_spatial_pad = ((p1, p1), (p2, p2)) if (p1 or p2) else None
            if channels_last:
                m.weight.data = m.weight.data.contiguous(
                    memory_format=torch.channels_last_3d
                )
            m.forward = MethodType(_fused_conv_forward, m)
            n_convs += 1
        elif isinstance(m, ResnetBlock3D):
            m._sgl_channels_last = channels_last
            m.forward = MethodType(_fused_resnet_forward, m)
            n_blocks += 1
        elif isinstance(m, Downsample3D):
            m.forward = MethodType(_fused_downsample_forward, m)
            n_downsamples += 1
        elif isinstance(m, EncoderFCN3D):
            m._sgl_channels_last = channels_last
            m.forward = MethodType(_fused_encoder_forward, m)
            n_encoders += 1

    _installed.add(vae)
    logger.info(
        "MiniMax-H3 VAE encoder cudnn_conv fast path installed "
        "(%d convs, %d resnet blocks, %d downsamples, %d encoder forward(s), "
        "channels_last=%s).",
        n_convs,
        n_blocks,
        n_downsamples,
        n_encoders,
        channels_last,
    )


def _uninstall_minimax_h3_vae_encoder_cudnn_conv(vae: nn.Module) -> None:
    """Roll back the instance-level patches; used by the auto gate on failure.

    Conv weights go back to standard-contiguous: aten conv3d propagates the
    weight's channels_last layout to its output, and encode callers (the
    tile-parallel all-gather) require standard-contiguous moments.
    """
    for m in vae.encoder.modules():
        if isinstance(m, (BaseConv3d, ResnetBlock3D, Downsample3D, EncoderFCN3D)):
            m.__dict__.pop("forward", None)
        if isinstance(m, BaseConv3d):
            m.weight.data = m.weight.data.contiguous()
    _installed.discard(vae)


def maybe_optimize_minimax_h3_vae_encoder(vae: nn.Module) -> nn.Module:
    """Entry point called last from the CUDA platform's optimize_vae hook.

    ``MINIMAX_H3_VAE_ENCODER_CUDNN_CONV`` selects the gate mode: unset/"auto"
    installs the fast path when available and falls back to the eager encoder
    on any failure; "1" makes any failure raise; "0" keeps the eager encoder.
    """
    from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae import (
        AutoencoderKLLegacy,
    )

    if not isinstance(vae, AutoencoderKLLegacy):
        return vae
    enabled = _env_optional_bool(ENV_FLAG)
    if enabled is False:
        logger.info(
            "MiniMax-H3 VAE encoder cudnn_conv fast path disabled (%s=0); "
            "running the eager encoder.",
            ENV_FLAG,
        )
        return vae
    if enabled:
        install_minimax_h3_vae_encoder_cudnn_conv(vae)
        _warmup_plan_cache(vae)
        return vae
    # Auto mode: warmup runs the patched encoder for real, so an
    # execution-time cudnn failure surfaces here instead of inside the first
    # request; roll the patches back and keep serving eager.
    try:
        install_minimax_h3_vae_encoder_cudnn_conv(vae)
        _warmup_plan_cache(vae)
    except Exception:
        _uninstall_minimax_h3_vae_encoder_cudnn_conv(vae)
        logger.warning(
            "MiniMax-H3 VAE encoder cudnn_conv fast path unavailable; running "
            "the eager encoder. Set %s=1 to fail loud or %s=0 to silence.",
            ENV_FLAG,
            ENV_FLAG,
            exc_info=True,
        )
    return vae


__all__ = [
    "ENV_FLAG",
    "install_minimax_h3_vae_encoder_cudnn_conv",
    "maybe_optimize_minimax_h3_vae_encoder",
]
