# SPDX-License-Identifier: Apache-2.0
"""Per-head dynamic FP8 e4m3 quantization for FA3 fp8 attention inputs.

FA3's fp8 path takes e4m3 q/k/v plus one fp32 descale per (segment, head);
per-token scales cannot be used there because a per-row q scale does not fold
into softmax and a per-row v scale does not fold out of the PV sum. This
kernel quantizes one ``[S, H, D]`` tensor with one dynamic scale per head:

    amax_h = max(|x[:, h, :]|)                    (exact, order-independent)
    scale  = clamp(amax / 448, min=1e-12)         fp32, [H]
    q      = fp8_e4m3(clamp(x / scale, +-448))    div.rn.f32, RN cast

Payload contract: bitwise equal to the torch reference
``(x.float() / scale.view(1, H, 1)).clamp(-448, 448).to(float8_e4m3fn)``
for finite inputs (tl.minimum/maximum drop a NaN where torch.clamp keeps it).
Unlike the per-token producer kernels this is not a lossless move of an
existing quant point -- feeding FA3 fp8 is quality-gated by construction.

Scale scheme note: dynamic per-head amax costs one extra read pass over x
(measured 595 us vs 363 us for delayed scaling per 3 production tensors of
[41984, 28, 128] on H200); dynamic was chosen because the ~230 us/block delta
buys exact scales with no cross-step amax state, drift, or bootstrap.
"""

from __future__ import annotations

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore

from sglang.kernels.ops.diffusion.common.numerics import div_rn_f32

_FP8_E4M3_MAX = 448.0
# Keeps an all-zero head finite: payload underflows to zero instead of 0/0.
_SCALE_FLOOR = 1e-12


@triton.jit
def _per_head_amax_kernel(
    x_ptr,
    amax_ptr,
    S,
    D,
    stride_s,
    stride_h,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    head = tl.program_id(0)
    chunk = tl.program_id(1).to(tl.int64)
    rows = chunk * BLOCK_S + tl.arange(0, BLOCK_S)
    cols = tl.arange(0, BLOCK_D)
    mask = (rows[:, None] < S) & (cols[None, :] < D)
    ptrs = x_ptr + rows[:, None] * stride_s + head * stride_h + cols[None, :]
    x = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
    tl.atomic_max(amax_ptr + head, tl.max(tl.abs(x)))


@triton.jit
def _per_head_quant_kernel(
    x_ptr,
    q_ptr,
    scale_ptr,
    S,
    D,
    stride_s,
    stride_h,
    heads,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    head = tl.program_id(0)
    chunk = tl.program_id(1).to(tl.int64)
    rows = chunk * BLOCK_S + tl.arange(0, BLOCK_S)
    cols = tl.arange(0, BLOCK_D)
    mask = (rows[:, None] < S) & (cols[None, :] < D)
    ptrs = x_ptr + rows[:, None] * stride_s + head * stride_h + cols[None, :]
    x = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr + head)
    # div_rn_f32 is elementwise inline asm: broadcast the scalar to the tile.
    val = div_rn_f32(x, tl.zeros_like(x) + scale)
    val = tl.minimum(tl.maximum(val, -448.0), 448.0)
    out_ptrs = q_ptr + rows[:, None] * (heads * D) + head * D + cols[None, :]
    tl.store(out_ptrs, val.to(tl.float8e4nv), mask=mask)


def can_use_per_head_quant_fp8(x: torch.Tensor) -> bool:
    return (
        x.is_cuda
        and x.dtype is torch.bfloat16
        and x.dim() == 3
        and x.numel() > 0
        and x.stride(-1) == 1
    )


def per_head_quant_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``[S, H, D]`` to fp8 e4m3 with one dynamic scale per head.

    Returns ``(payload, scale)``: ``payload`` fp8 ``[S, H, D]`` contiguous and
    ``scale`` fp32 ``[H]``; FA3 consumes them as (q/k/v, descale). Accepts
    row/head-strided inputs (e.g. views of the packed qkv exchange buffer).
    """
    if not can_use_per_head_quant_fp8(x):
        raise RuntimeError("unsupported input for per-head fp8 quant")
    seq, heads, head_dim = x.shape
    amax = torch.zeros(heads, dtype=torch.float32, device=x.device)
    payload = torch.empty(
        (seq, heads, head_dim), dtype=torch.float8_e4m3fn, device=x.device
    )
    # Fastest of the {64,128,256} sweep on H200 at [41984, 28, 128].
    block_s = 128
    grid = (heads, triton.cdiv(seq, block_s))
    with torch.cuda.device(x.device):
        _per_head_amax_kernel[grid](
            x,
            amax,
            seq,
            head_dim,
            x.stride(0),
            x.stride(1),
            BLOCK_S=block_s,
            BLOCK_D=triton.next_power_of_2(head_dim),
        )
        scale = amax.div_(_FP8_E4M3_MAX).clamp_(min=_SCALE_FLOOR)
        _per_head_quant_kernel[grid](
            x,
            payload,
            scale,
            seq,
            head_dim,
            x.stride(0),
            x.stride(1),
            heads,
            BLOCK_S=block_s,
            BLOCK_D=triton.next_power_of_2(head_dim),
        )
    return payload, scale
