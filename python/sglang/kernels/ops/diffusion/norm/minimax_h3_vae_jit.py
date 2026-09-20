# SPDX-License-Identifier: Apache-2.0
"""Close-contract H3 residual/norm and optional FP8 producers, CUDA/C++ JIT.

Rows have width 2048, with FP32 affine parameters. RMSNorm emits per-token
E4M3 data/scales; final LayerNorm emits FP32. Reduction/FMA order differs
from eager Torch, so all model call sites require the request quality gate.
"""

import torch

from ..common.minimax_h3_vae_jit import load_vae_kernel, supported_cuda_tensor


def can_use_minimax_h3_vae_norm_quant(
    x, weight, projected=None, layer_scale=None, bias=None, *, final=False
):
    if not (
        supported_cuda_tensor(x)
        and x.ndim >= 2
        and x.shape[-1] == 2048
        and x.is_contiguous()
        and x.dtype in (torch.float32, torch.float16)
        and weight.shape == (2048,)
        and weight.dtype == torch.float32
        and weight.device == x.device
        and weight.is_contiguous()
    ):
        return False
    if projected is not None:
        if not (
            projected.shape == x.shape
            and projected.is_contiguous()
            and projected.device == x.device
            and projected.dtype in (torch.float16, torch.float32)
            and layer_scale is not None
            and layer_scale.shape == (2048,)
            and layer_scale.device == x.device
            and layer_scale.dtype == torch.float32
            and layer_scale.is_contiguous()
        ):
            return False
    elif layer_scale is not None or final:
        return False
    return not final or (
        bias is not None
        and bias.shape == (2048,)
        and bias.dtype == torch.float32
        and bias.device == x.device
        and bias.is_contiguous()
    )


def minimax_h3_vae_norm_quant(
    x, weight, eps, projected=None, layer_scale=None, bias=None, *, final=False
):
    if not can_use_minimax_h3_vae_norm_quant(
        x, weight, projected, layer_scale, bias, final=final
    ):
        raise ValueError("Unsupported MiniMax-H3 residual/norm fusion inputs")
    if eps <= 0:
        raise ValueError("Normalization epsilon must be positive")
    flat = x.view(-1, 2048)
    mode = 2 if final else 1 if projected is not None else 0
    q = torch.empty_like(flat, dtype=torch.float8_e4m3fn) if not final else flat
    scales = torch.empty((flat.shape[0], 1), device=x.device, dtype=torch.float32)
    residual = torch.empty_like(flat, dtype=torch.float32) if mode == 1 else flat
    normalized = torch.empty_like(flat, dtype=torch.float32) if final else residual
    if final:
        # Mode 2 does not store a residual, but its host contract still expects
        # FP32 storage even when the incoming state is FP16.
        residual = normalized
    proj = projected.view(-1, 2048) if projected is not None else flat
    gamma_scale = layer_scale if layer_scale is not None else weight
    beta = bias if final else weight
    load_vae_kernel("norm", x.dtype, mode=mode, projected_dtype=proj.dtype).run(
        flat,
        proj,
        gamma_scale,
        weight,
        beta,
        residual,
        q,
        scales,
        normalized,
        float(eps),
    )
    if final:
        return normalized.reshape(x.shape)
    return (residual.reshape(x.shape) if mode else x), q, scales


def can_use_minimax_h3_vae_group_norm_silu(
    x, groups, weight, bias, eps, *, time_isolated=True
):
    if not (
        supported_cuda_tensor(x)
        and x.ndim in (4, 5)
        and x.dtype == torch.float32
        and isinstance(groups, int)
        and groups > 0
        and x.shape[1] % groups == 0
        and eps > 0
        and all(s >= 0 for s in x.stride())
    ):
        return False
    for value in (weight, bias):
        if (
            value is None
            or value.device != x.device
            or value.dtype != torch.float32
            or value.shape != (x.shape[1],)
            or not value.is_contiguous()
        ):
            return False
    return x.numel() < 2**31


def minimax_h3_vae_group_norm_silu(x, groups, weight, bias, eps, *, time_isolated=True):
    """FP32 GroupNorm+SiLU; causal groups never reduce across time.

    Contiguous groups with power-of-two channel spans use vector loads and
    one launch up to 8192 elements. General layouts use one launch up to 4096.
    Larger groups merge chunked central moments before fused affine/SiLU.
    This close-contract path is selected only by the request quality gate.
    """
    from ..common.minimax_h3_vae_jit import load_group_norm_kernel

    if not can_use_minimax_h3_vae_group_norm_silu(
        x, groups, weight, bias, eps, time_isolated=time_isolated
    ):
        raise ValueError("Unsupported FP32 MiniMax-H3 GroupNorm/SiLU inputs")
    value = x.unsqueeze(2) if x.ndim == 4 else x
    b, c, t, h, w = value.shape
    count = (c // groups) * h * w * (1 if time_isolated else t)
    rows = b * groups * (t if time_isolated else 1)
    out = torch.empty(value.shape, device=x.device, dtype=torch.float32)
    partial = (
        torch.empty(
            (rows, (count + 4095) // 4096, 2), device=x.device, dtype=torch.float32
        )
        if count > 4096
        else out
    )
    stats = (
        torch.empty((rows, 2), device=x.device, dtype=torch.float32)
        if count > 4096
        else out
    )
    load_group_norm_kernel().run(
        value,
        weight,
        bias,
        partial,
        stats,
        out,
        int(groups),
        bool(time_isolated),
        float(eps),
    )
    return out.squeeze(2) if x.ndim == 4 else out
