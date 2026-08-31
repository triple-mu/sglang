# SPDX-License-Identifier: Apache-2.0
"""Fused packed SwiGLU + per-token FP8 quantization for W8A8 MLPs.

Under ``--quantization fp8``, MiniMax-H3's fc2 input today is produced by the
eager pair ``F.silu(gate) * up`` (the ``reuse_fc1_activation`` fast path is
disabled by quantization) followed by a standalone
``sgl_per_token_quant_fp8`` pass.  This kernel computes, in one pass over the
``[M, 2*D]`` fc1 output laid out as ``[gate | up]``:

    act   = bf16(bf16(silu(gate)) * up)      (both aten rounding boundaries)
    scale = amax(|act|) / 448                 per row, fp32
    q     = fp8_e4m3(act / scale)             clamped, satfinite

Numerical contract: **bitwise equal to the separated chain** -- payload and
scale match ``sgl_per_token_quant_fp8(F.silu(gate) * up)`` on every byte
(``fp8_quant_replica`` documents the replicated reference arithmetic,
including its dispatch-dependent zero-scale semantics).  The bf16 activation
itself reproduces ``activation/silu_mul_bitexact.py``'s verified boundaries,
so the kernel is also bitwise vs ``silu_and_mul_with_activation_rounding``
followed by the quant -- restoring the fused-activation semantics fp8 serving
lost.  Verified on the production shape ``[20992, 28672] -> [20992, 14336]``.
"""

from __future__ import annotations

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore

from sglang.kernels.ops.diffusion.common.fp8_quant_replica import (
    amax_abs_f32,
    quantize_payload_f32,
    reference_fast_math,
    reference_zero_guard,
    scale_from_amax,
    scale_inv_from_scale,
)
from sglang.kernels.ops.diffusion.common.numerics import round_bf16_to_fp32


@triton.jit
def _round_bf16_to_fp32_keep_nan(value):
    """``round_bf16_to_fp32`` with hardware cvt semantics for NaN.

    The RNE bit trick adds a rounding bias, which carries a fresh NaN's
    near-all-ones mantissa (e.g. 0x7FFFFFFF from ``0 * inf``) into the sign
    bit; ``cvt.rn.bf16.f32`` -- what the eager chain's store executes --
    truncates a NaN mantissa instead.
    """
    rounded = round_bf16_to_fp32(value)
    bits = value.to(tl.uint32, bitcast=True)
    truncated = (bits & 0xFFFF0000).to(tl.float32, bitcast=True)
    return tl.where(value != value, truncated, rounded)


@triton.jit
def _silu_mul_act(gate, up):
    """Both bf16-rounded steps of the eager ``F.silu(gate) * up`` chain."""
    silu = _round_bf16_to_fp32_keep_nan(gate * tl.sigmoid(gate))
    return _round_bf16_to_fp32_keep_nan(silu * up)


@triton.jit
def _silu_mul_per_token_quant_fp8_kernel(
    x_ptr,
    q_ptr,
    s_ptr,
    D,
    stride_x_row,
    FAST_MATH: tl.constexpr,
    ZERO_GUARD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    base = x_ptr + row * stride_x_row
    # Pass 1: row amax over the recomputable bf16 activation.
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for off in range(0, D, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < D
        gate = tl.load(base + cols, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(base + D + cols, mask=mask, other=0.0).to(tl.float32)
        acc = tl.maximum(acc, amax_abs_f32(_silu_mul_act(gate, up), FAST_MATH))
    amax = tl.max(acc, axis=0, keep_dims=True)
    scale = scale_from_amax(amax, FAST_MATH)
    tl.store(s_ptr + row + tl.arange(0, 1), scale)
    scale_inv = tl.sum(scale_inv_from_scale(scale, FAST_MATH, ZERO_GUARD), axis=0)
    # Pass 2: recompute the activation (deterministic) and store fp8.
    for off in range(0, D, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < D
        gate = tl.load(base + cols, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(base + D + cols, mask=mask, other=0.0).to(tl.float32)
        val = quantize_payload_f32(_silu_mul_act(gate, up), scale_inv, FAST_MATH)
        tl.store(q_ptr + row * D + cols, val.to(tl.float8e4nv), mask=mask)


def can_use_fused_silu_mul_per_token_quant_fp8(x: torch.Tensor) -> bool:
    return (
        x.is_cuda
        and x.dtype is torch.bfloat16
        and x.dim() == 2
        and x.shape[-1] % 2 == 0
        and x.numel() > 0
        and x.stride(-1) == 1
        and x.stride(0) >= x.shape[-1]
    )


def fused_silu_mul_per_token_quant_fp8(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``silu(gate) * up`` + per-token FP8 quant over ``x = [gate | up]``.

    Returns ``(q, s)`` with ``q`` fp8_e4m3 of shape ``[M, D]`` and ``s`` fp32
    of shape ``[M, 1]``, bitwise equal to feeding the eager activation into
    ``sgl_per_token_quant_fp8``.
    """
    if not can_use_fused_silu_mul_per_token_quant_fp8(x):
        raise RuntimeError("unsupported input for fused SiLU-mul + fp8 quant")
    rows, packed = x.shape
    hidden = packed // 2
    q = torch.empty((rows, hidden), dtype=torch.float8_e4m3fn, device=x.device)
    s = torch.empty((rows, 1), dtype=torch.float32, device=x.device)
    with torch.cuda.device(x.device):
        _silu_mul_per_token_quant_fp8_kernel[(rows,)](
            x,
            q,
            s,
            hidden,
            x.stride(0),
            FAST_MATH=reference_fast_math(),
            ZERO_GUARD=reference_zero_guard(rows, x.device),
            # Fastest of the {512..4096} x {4,8,16} sweep on H200 at the
            # production shape [20992, 28672]; re-tune when the shape changes.
            BLOCK=1024,
            num_warps=4,
        )
    return q, s
