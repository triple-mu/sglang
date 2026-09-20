# SPDX-License-Identifier: Apache-2.0
"""CUDA fast path for the MiniMax-H3 video VAE.

Installed once at VAE load and dispatched on one decode-scoped
:class:`VaeFastPathGate` that ``quality="lossless"`` and ``"high"`` enable while
the ``"exact"`` tier runs the original module path bit-for-bit. Two layers share
the gate:

* every ViT decoder attention block fuses its per-head QK RMSNorm + NeoX RoPE
  into one in-place launch and runs its attention on cuDNN SDPA (any CUDA GPU);
* on SM100 or newer a :class:`MiniMaxH3VaeFastPath` additionally batches
  equal-shaped tiles and temporal windows and dispatches the JIT kernels
  (GroupNorm+SiLU, paired Q/K RMSNorm+RoPE for the FP8 decoder, tile assembly,
  temporal blend write, output denormalization).

Install is all-or-nothing per layer and fail-closed.
"""

from types import MethodType

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from sglang.kernels.ops.diffusion import (
    can_use_fused_inplace_qknorm_rope,
    fused_inplace_qknorm_rope,
)
from sglang.kernels.ops.diffusion.common.platform import is_cuda_sm_at_least
from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.models.vaes.fast_path_gate import (
    VaeFastPathGate,
    register_vae_fast_path_gate,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.attention import (
    Attention,
    _apply_qk_norm,
    _sdpa_attention,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.fast_path import (
    MINIMAX_H3_VAE_MIN_SM,
    MiniMaxH3VaeFastPath,
    resolve_minimax_h3_vae_batch_caps,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.vae_cnn import (
    EncoderFCN3D,
    ResnetBlock3D,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.vae_vit import (
    ViT3DDecoder,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.vit_utils import (
    _env_flag,
    apply_rotary_pos_emb_qk,
    native_rope_cache,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def _fused_qknorm_rope(self, query, key, rotary_pos_emb) -> bool:
    native = native_rope_cache(query, key, rotary_pos_emb)
    if native is None:
        return False
    cache, positions = native
    if not can_use_fused_inplace_qknorm_rope(
        head_dim=self.dim_head,
        rope_dim=cache.shape[1],
        is_neox=True,
        dtype=query.dtype,
        cache_dtype=cache.dtype,
        round_norm_before_rope=True,
    ):
        return False
    weight = self._sgl_unit_weight
    if weight is None or weight.dtype != query.dtype or weight.device != query.device:
        weight = torch.ones(self.dim_head, dtype=query.dtype, device=query.device)
        self._sgl_unit_weight = weight
    # Batched decoder tiles flatten into rows; a view, so the update is in place.
    fused_inplace_qknorm_rope(
        query.flatten(0, 1),
        key.flatten(0, 1),
        weight,
        weight,
        cache,
        positions,
        is_neox=True,
        eps=self.norm_q.eps,
        head_dim=self.dim_head,
        rope_dim=cache.shape[1],
        round_norm_before_rope=True,
    )
    return True


def _cudnn_attention(self, query, key, value):
    if not self._sgl_cudnn_failed:
        try:
            with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                return F.scaled_dot_product_attention(
                    query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
                ).transpose(1, 2)
        except RuntimeError as e:
            logger.warning(
                "MiniMax-H3 VAE: cuDNN SDPA failed (%s); using the layer's "
                "attention backend.",
                e,
            )
            self._sgl_cudnn_failed = True
    return self.attn(query, key, value)


def _attn_fast_forward(self, hidden_states, rotary_pos_emb=None):
    if (
        not self._sgl_gate.enabled
        or rotary_pos_emb is None
        or torch.is_grad_enabled()
        or torch.compiler.is_compiling()
    ):
        return type(self).forward(self, hidden_states, rotary_pos_emb)

    batch_size, seq_len, _ = hidden_states.shape
    qkv = self.to_qkv(hidden_states)
    qkv = qkv.view(batch_size, seq_len, -1, 3 * self.dim_head)
    query, key, value = torch.chunk(qkv, 3, dim=-1)

    if not _fused_qknorm_rope(self, query, key, rotary_pos_emb):
        query = _apply_qk_norm(self.norm_q, query)
        key = _apply_qk_norm(self.norm_k, key)
        query, key = apply_rotary_pos_emb_qk(query, key, rotary_pos_emb)

    if self.attn is not None and query.dtype in (torch.float16, torch.bfloat16):
        hidden_states = _cudnn_attention(self, query, key, value)
    else:
        hidden_states = _sdpa_attention(query, key, value)

    hidden_states = hidden_states.reshape(batch_size, seq_len, -1)
    return self.to_out(hidden_states)


def _plain_rms_norm(module) -> bool:
    return isinstance(module, nn.RMSNorm) and module.weight is None


def _attn_fast_compatible(attn: nn.Module) -> bool:
    return (
        type(attn) is Attention
        and _plain_rms_norm(attn.norm_q)
        and _plain_rms_norm(attn.norm_k)
        and attn.norm_q.eps == attn.norm_k.eps
    )


def _install_fast_attention(decoder: nn.Module, gate: VaeFastPathGate) -> int:
    """Bind the fused QK norm+RoPE / cuDNN forward on every decoder attention block.

    Returns the number of blocks rebound; 0 when any block is non-standard or the
    fp32 QK norm is disabled, because a partial install would change numerics
    between blocks.
    """
    if type(decoder) is not ViT3DDecoder:
        return 0
    if not _env_flag("MINIMAX_H3_VAE_DECODER_VIT_FP32_NORM", "1"):
        logger.info("MiniMax-H3 VAE: fp32 QK norm disabled; skipping attention fast path.")
        return 0
    attn_modules = [block.attn for block in decoder.transformer_blocks]
    eligible = [attn for attn in attn_modules if _attn_fast_compatible(attn)]
    if len(eligible) != len(attn_modules):
        logger.warning(
            "MiniMax-H3 VAE: %d/%d decoder attention blocks non-standard; "
            "skipping attention fast path.",
            len(attn_modules) - len(eligible),
            len(attn_modules),
        )
        return 0
    for attn in eligible:
        attn._sgl_gate = gate
        attn._sgl_unit_weight = None
        attn._sgl_cudnn_failed = False
        attn.forward = MethodType(_attn_fast_forward, attn)
    return len(eligible)


def _install_fast_path_slots(
    vae: nn.Module, state: MiniMaxH3VaeFastPath
) -> tuple[int, int]:
    """Fill every explicit ``fast_path`` field; returns (norm, attention) site counts."""
    vae.fast_path = state
    vae.processor.fast_path = state
    norm_sites = 0
    attention_sites = 0
    for module in vae.modules():
        if isinstance(module, (ResnetBlock3D, EncoderFCN3D, ViT3DDecoder)):
            module.fast_path = state
            norm_sites += 2 if isinstance(module, ResnetBlock3D) else 0
            norm_sites += 1 if isinstance(module, EncoderFCN3D) else 0
        elif isinstance(module, Attention):
            module.fast_path = state
            attention_sites += 1
    return norm_sites, attention_sites


def _install_jit_fast_path(vae: nn.Module, gate: VaeFastPathGate) -> bool:
    label = "MiniMax-H3 video VAE"
    if envs.SGLANG_DIFFUSION_DISABLE_MINIMAX_H3_VAE_FAST_PATH:
        logger.info(
            "%s: JIT fast path disabled by "
            "SGLANG_DIFFUSION_DISABLE_MINIMAX_H3_VAE_FAST_PATH.",
            label,
        )
        return False
    if not is_cuda_sm_at_least(MINIMAX_H3_VAE_MIN_SM):
        logger.info("%s: JIT fast path needs CUDA SM100 or newer; skipping.", label)
        return False
    encoder_tile_batch, decoder_tile_batch, window_batch = (
        resolve_minimax_h3_vae_batch_caps()
    )
    state = MiniMaxH3VaeFastPath(
        gate=gate,
        encoder_tile_batch=encoder_tile_batch,
        decoder_tile_batch=decoder_tile_batch,
        window_batch=window_batch,
    )
    norm_sites, attention_sites = _install_fast_path_slots(vae, state)
    logger.info(
        "%s: installed quality-gated JIT fast path (%d GroupNorm+SiLU sites, "
        "%d paired Q/K norm+RoPE sites, batch caps enc%d/dec%d/win%d).",
        label,
        norm_sites,
        attention_sites,
        encoder_tile_batch,
        decoder_tile_batch,
        window_batch,
    )
    return True


def maybe_optimize_minimax_h3_vae(vae: nn.Module) -> nn.Module:
    """Install the quality-gated CUDA MiniMax-H3 video VAE fast paths."""
    from sglang.multimodal_gen.runtime.models.vaes.minimax_h3 import MiniMaxH3VideoVAE

    if not isinstance(vae, MiniMaxH3VideoVAE):
        return vae
    gate = VaeFastPathGate()
    attention_blocks = _install_fast_attention(vae.decoder, gate)
    jit_installed = _install_jit_fast_path(vae, gate)
    if attention_blocks == 0 and not jit_installed:
        return vae
    register_vae_fast_path_gate(vae, gate)
    if attention_blocks:
        logger.info(
            "MiniMax-H3 VAE: installed quality-gated attention fast path (%d QK "
            "RMSNorm+RoPE fusions, cuDNN SDPA).",
            attention_blocks,
        )
    return vae
