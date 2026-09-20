# SPDX-License-Identifier: Apache-2.0
"""Request-gated H3 decoder dataflow; all-or-nothing fused block selection."""

from typing import NamedTuple

import torch
from torch import nn

from sglang.kernels.ops.diffusion import (
    can_use_minimax_h3_vae_qk_rope,
    minimax_h3_vae_norm_quant,
    minimax_h3_vae_qk_rope,
    minimax_h3_vae_silu_quant,
)

from .fp8 import MiniMaxH3FP8Linear
from .optimizations import optimization_active


class Quantized(NamedTuple):
    values: torch.Tensor
    scales: torch.Tensor
    shape: tuple


class ResidualPending(NamedTuple):
    residual: torch.Tensor
    projected: torch.Tensor
    scale: torch.Tensor


def linear(module, value):
    if isinstance(value, Quantized):
        if not isinstance(module, MiniMaxH3FP8Linear):
            raise TypeError("A prequantized activation must feed an H3 FP8 linear")
        return module.forward_quantized(value.values, value.scales, value.shape)
    return module(value)


def paired_qk(module, qkv, rope):
    if (
        optimization_active(module, qkv)
        and len(rope) >= 3
        and isinstance(module.norm_q, nn.RMSNorm)
        and isinstance(module.norm_k, nn.RMSNorm)
        and module.norm_q.weight is None
        and module.norm_k.weight is None
        and can_use_minimax_h3_vae_qk_rope(qkv, rope[2])
    ):
        q, k = minimax_h3_vae_qk_rope(
            qkv, rope[2], module.norm_q.eps, module.norm_k.eps
        )
        return q, k, qkv[..., 128:]
    return None


def _vector(value, size, device):
    return (
        value is not None
        and value.shape == (size,)
        and value.dtype == torch.float32
        and value.device == device
        and value.is_contiguous()
    )


def can_use_fused_decoder(decoder, value, rope):
    if not (
        optimization_active(decoder, value)
        and getattr(decoder, "_sgl_fp8_installed", False)
        and value.dtype == torch.float32
        and value.is_contiguous()
        and value.shape[-1] == 2048
        and not decoder.mask_enabled
        and len(rope) == 4
        and rope[2].shape == (value.shape[0] * value.shape[1], 48)
        and rope[2].dtype == torch.float16
        and rope[2].device == value.device
        and rope[2].is_contiguous()
    ):
        return False
    if (
        not isinstance(decoder.norm_out, nn.LayerNorm)
        or not _vector(decoder.norm_out.weight, 2048, value.device)
        or not _vector(decoder.norm_out.bias, 2048, value.device)
    ):
        return False
    for block in decoder.transformer_blocks:
        if (
            not block.use_scale
            or not _vector(block.scale1, 2048, value.device)
            or not _vector(block.scale2, 2048, value.device)
        ):
            return False
        for norm in (block.norm1, block.norm2):
            if (
                not isinstance(norm, nn.RMSNorm)
                or not _vector(norm.weight, 2048, value.device)
                or norm.eps <= 0
            ):
                return False
        attn = block.attn
        if (
            attn.dim_head != 64
            or not isinstance(attn.norm_q, nn.RMSNorm)
            or not isinstance(attn.norm_k, nn.RMSNorm)
        ):
            return False
        if (
            attn.norm_q.weight is not None
            or attn.norm_k.weight is not None
            or attn.attn is None
        ):
            return False
        if not block.ff.use_gated or not isinstance(block.ff.act_fn, nn.SiLU):
            return False
        if not all(
            isinstance(m, MiniMaxH3FP8Linear)
            for m in (attn.to_qkv, attn.to_out, block.ff.w1, block.ff.w2)
        ):
            return False
    return True


def normalize(value, norm):
    if isinstance(value, ResidualPending):
        residual, q, scale = minimax_h3_vae_norm_quant(
            value.residual, norm.weight, norm.eps, value.projected, value.scale
        )
    else:
        residual, q, scale = minimax_h3_vae_norm_quant(value, norm.weight, norm.eps)
    return residual, Quantized(q, scale, tuple(residual.shape))


def block_forward(module, value, rope):
    residual, normed = normalize(value, module.norm1)
    shape = normed.shape
    packed = linear(module.attn.to_qkv, normed).view(
        shape[0], shape[1], module.attn.heads, 192
    )
    q, k = minimax_h3_vae_qk_rope(
        packed, rope[2], module.attn.norm_q.eps, module.attn.norm_k.eps
    )
    attended = module.attn.attn(q, k, packed[..., 128:]).reshape(shape[0], shape[1], -1)
    projected = linear(module.attn.to_out, attended)
    residual, normed = normalize(
        ResidualPending(residual, projected, module.scale1), module.norm2
    )
    expanded = linear(module.ff.w1, normed)
    q, scale = minimax_h3_vae_silu_quant(expanded)
    activation = Quantized(q, scale, (*expanded.shape[:-1], expanded.shape[-1] // 2))
    projected = linear(module.ff.w2, activation)
    # Keep the last add pending: the next block consumes it together with its
    # norm/quant, and the final block instead feeds residual+LayerNorm.
    return ResidualPending(residual, projected, module.scale2)


def finish(value, norm):
    return minimax_h3_vae_norm_quant(
        value.residual,
        norm.weight,
        norm.eps,
        value.projected,
        value.scale,
        norm.bias,
        final=True,
    )


def output_projection(module, value):
    """Use half-input/FP32-output GEMM without mutating global math flags."""
    if torch.backends.cuda.matmul.allow_fp16_accumulation:
        return None
    # Inference tensors have no version counter. Rebuild their rounded view
    # so an in-place checkpoint reload cannot reuse stale projection weights.
    version = None if module.weight.is_inference() else module.weight._version
    key = (module.weight.device, module.weight.dtype, module.weight.data_ptr(), version)
    cache = getattr(module, "_sgl_half_projection", None)
    if version is None or cache is None or cache[0] != key:
        cache = (key, module.weight.detach().to(torch.float16).t().contiguous())
        module._sgl_half_projection = cache
    with torch.autocast("cuda", enabled=False):
        result = torch.mm(
            value.reshape(-1, value.shape[-1]).to(torch.float16),
            cache[1],
            out_dtype=torch.float32,
        )
        if module.bias is not None:
            result = result + module.bias.float()
    return result.reshape(*value.shape[:-1], module.out_features)
