# SPDX-License-Identifier: Apache-2.0
"""Close-contract paired RMSNorm/RoPE for H3's packed FP16/BF16 QKV.

Validated prototype shapes: [B,S,32,192], head width 64 and rotary width 48
on SM100. Both Q and K share one launch. Callers must use the quality gate.
"""

import torch

from ..common.minimax_h3_vae_jit import load_vae_kernel, supported_cuda_tensor


def can_use_minimax_h3_vae_qk_rope(qkv, cache) -> bool:
    return (
        supported_cuda_tensor(qkv)
        and qkv.ndim == 4
        and qkv.shape[-1] == 192
        and qkv.is_contiguous()
        and qkv.dtype in (torch.float16, torch.bfloat16)
        and cache.device == qkv.device
        and cache.dtype == qkv.dtype
        and cache.is_contiguous()
        and cache.shape == (qkv.shape[0] * qkv.shape[1], 48)
    )


def minimax_h3_vae_qk_rope(qkv, cache, eps_q: float, eps_k: float):
    if not can_use_minimax_h3_vae_qk_rope(qkv, cache):
        raise ValueError("Unsupported packed QKV or rotary cache for MiniMax-H3 VAE")
    if eps_q <= 0 or eps_k <= 0:
        raise ValueError("RMSNorm epsilon must be positive")
    q = torch.empty((*qkv.shape[:-1], 64), device=qkv.device, dtype=qkv.dtype)
    k = torch.empty_like(q)
    load_vae_kernel("qk", qkv.dtype).run(
        qkv, cache, q, k, q, float(eps_q), float(eps_k)
    )
    return q, k
