# SPDX-License-Identifier: Apache-2.0
"""Compute-dtype gate for the MiniMax H3 visual VAE encoder.

The encode recipes' default contract pins fp32 weights (TF32 conv engines).
``MINIMAX_H3_VAE_ENCODER_BF16=1`` switches the encoder to bf16 io with fp32
accumulation instead: conv and GroupNorm parameters are cast to bf16 once
(``AutoencoderKLLegacy.encode`` then casts each input tile to match), while
statistics stay fp32 inside both the aten GroupNorm CUDA kernel and the fused
``group_norm_silu_4d`` Triton kernel, mirroring how the decoder's fp16
autocast recipe keeps its norms in higher precision.

Quality-gated and default OFF: bf16 conditioning latents change and propagate
through the whole generated clip, so the flag only flips ON after the e2e
video+audio quality gate passes.

The cast must run BEFORE the cudnn_conv fast-path install: cudnn execution
plans and the channels_last_3d weight conversion are dtype-specific, and the
install-time plan warmup must build the bf16 plans. cudnn_conv supports bf16
natively (compute type stays FLOAT), so the two gates compose; in auto mode a
cudnn failure still rolls back to the eager encoder, which then runs bf16.
"""

import torch
import torch.nn as nn
from diffusers.utils import logging

from .vit_utils import _env_flag

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

ENCODER_BF16_ENV_FLAG = "MINIMAX_H3_VAE_ENCODER_BF16"


def minimax_h3_vae_encoder_bf16_enabled() -> bool:
    return _env_flag(ENCODER_BF16_ENV_FLAG, "0")


def maybe_cast_minimax_h3_vae_encoder_bf16(vae: nn.Module) -> bool:
    """Cast ``vae.encoder`` to bf16 once when the gate is on.

    Idempotent; the cast is permanent (the encoder is encode-only, so no
    caller needs the fp32 weights back). Returns True when a cast happened.
    """
    if not minimax_h3_vae_encoder_bf16_enabled():
        return False
    parameter = next(vae.encoder.parameters())
    if parameter.dtype == torch.bfloat16:
        return False
    vae.encoder.to(torch.bfloat16)
    logger.info(
        "MiniMax-H3 VAE encoder cast to bf16 (%s=1); conv io bf16 with fp32 "
        "accumulation, GroupNorm statistics fp32.",
        ENCODER_BF16_ENV_FLAG,
    )
    return True


__all__ = [
    "ENCODER_BF16_ENV_FLAG",
    "maybe_cast_minimax_h3_vae_encoder_bf16",
    "minimax_h3_vae_encoder_bf16_enabled",
]
