# SPDX-License-Identifier: Apache-2.0
"""Ulysses output head-merge fused with per-token FP8 quantization.

Under ``--quantization fp8``, the Ulysses output-side merge writes a bf16
``[tokens, h_global * head_dim]`` tensor that ``out_proj`` immediately
re-reads and quantizes with a standalone ``sgl_per_token_quant_fp8`` pass.
These kernels gather each token's head shards and emit the fp8 payload and
per-token fp32 scale directly, skipping the bf16 round trip.

Two entry points, one per merge form:

- :func:`usp_merge_heads_per_token_quant_fp8` -- the NCCL all-to-all form.
  Input is the ``[W, S, B, h_local, D]`` tensor ``usp_merge_heads`` consumes
  (``usp.py::_usp_output_all_to_all``, ``head_dim=2``); output row ``b*S + s``
  is the concatenation over ``w`` of ``x[w, s, b]``, quantized.
- :func:`merge_two_sources_per_token_quant_fp8` -- the 2-rank IPC form for the
  wiring wave.  The IPC fast path (``usp.py::_ipc_varlen_fast``, output
  direction) merges a local FA3-output slice with a peer-written staging block
  into one ``[tokens, 2 * h_local * D]`` row; after ``IPC_A2A.wait()`` both
  halves are plain CUDA tensors, so the fused quant takes them as two
  row-strided sources in output-column order.  (Today the peer writes strided
  into the interleaved staging layout; wiring can either keep that layout and
  pass the two staging column-halves here, or switch staging to
  half-contiguous blocks and pass them directly.)

Numerical contract: **bitwise equal to the separated chain** -- payload and
scale match ``sgl_per_token_quant_fp8`` applied to the merged bf16 rows
(``fp8_quant_replica`` documents the replicated reference arithmetic,
including its dispatch-dependent zero-scale semantics).  The merge itself is
pure data movement.  Verified on the production shape
``[2, 20992, 1, 28, 128] -> [20992, 7168]``.
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


@triton.jit
def _merge_heads_per_token_quant_fp8_kernel(
    x_ptr,
    q_ptr,
    s_ptr,
    S,
    C,
    W,
    stride_w,
    stride_s,
    stride_b,
    FAST_MATH: tl.constexpr,
    ZERO_GUARD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    base = x_ptr + (token % S) * stride_s + (token // S) * stride_b
    row_out = q_ptr + token * (W * C)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for w in range(W):
        shard = base + w * stride_w
        for off in range(0, C, BLOCK):
            cols = off + tl.arange(0, BLOCK)
            mask = cols < C
            v = tl.load(shard + cols, mask=mask, other=0.0).to(tl.float32)
            acc = tl.maximum(acc, amax_abs_f32(v, FAST_MATH))
    amax = tl.max(acc, axis=0, keep_dims=True)
    scale = scale_from_amax(amax, FAST_MATH)
    tl.store(s_ptr + token + tl.arange(0, 1), scale)
    scale_inv = tl.sum(scale_inv_from_scale(scale, FAST_MATH, ZERO_GUARD), axis=0)
    for w in range(W):
        shard = base + w * stride_w
        for off in range(0, C, BLOCK):
            cols = off + tl.arange(0, BLOCK)
            mask = cols < C
            v = tl.load(shard + cols, mask=mask, other=0.0).to(tl.float32)
            val = quantize_payload_f32(v, scale_inv, FAST_MATH)
            tl.store(row_out + w * C + cols, val.to(tl.float8e4nv), mask=mask)


@triton.jit
def _merge_two_sources_per_token_quant_fp8_kernel(
    first_ptr,
    second_ptr,
    q_ptr,
    s_ptr,
    C,
    stride_first_row,
    stride_second_row,
    FAST_MATH: tl.constexpr,
    ZERO_GUARD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    first = first_ptr + token * stride_first_row
    second = second_ptr + token * stride_second_row
    row_out = q_ptr + token * (2 * C)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for off in range(0, C, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < C
        a = tl.load(first + cols, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(second + cols, mask=mask, other=0.0).to(tl.float32)
        acc = tl.maximum(acc, amax_abs_f32(a, FAST_MATH))
        acc = tl.maximum(acc, amax_abs_f32(b, FAST_MATH))
    amax = tl.max(acc, axis=0, keep_dims=True)
    scale = scale_from_amax(amax, FAST_MATH)
    tl.store(s_ptr + token + tl.arange(0, 1), scale)
    scale_inv = tl.sum(scale_inv_from_scale(scale, FAST_MATH, ZERO_GUARD), axis=0)
    for off in range(0, C, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < C
        a = tl.load(first + cols, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(second + cols, mask=mask, other=0.0).to(tl.float32)
        val_a = quantize_payload_f32(a, scale_inv, FAST_MATH)
        val_b = quantize_payload_f32(b, scale_inv, FAST_MATH)
        tl.store(row_out + cols, val_a.to(tl.float8e4nv), mask=mask)
        tl.store(row_out + C + cols, val_b.to(tl.float8e4nv), mask=mask)


def can_use_usp_merge_heads_per_token_quant_fp8(x: torch.Tensor) -> bool:
    return (
        x.is_cuda
        and x.dtype is torch.bfloat16
        and x.dim() == 5
        and x.numel() > 0
        and x.is_contiguous()
    )


def usp_merge_heads_per_token_quant_fp8(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``usp_merge_heads`` + per-token FP8 quant in one pass.

    ``x`` is ``[W, S, B, h_local, D]`` contiguous; returns ``(q, s)`` with
    ``q`` fp8_e4m3 of shape ``[B*S, W*h_local*D]`` (row ``b*S + s``, the
    layout ``usp_merge_heads(x).reshape(B*S, -1)`` produces) and ``s`` fp32 of
    shape ``[B*S, 1]``, bitwise equal to quantizing the merged bf16 rows.
    """
    if not can_use_usp_merge_heads_per_token_quant_fp8(x):
        raise RuntimeError("unsupported input for fused USP head merge + fp8 quant")
    world, seq, batch, h_local, head_dim = x.shape
    inner = h_local * head_dim
    tokens = batch * seq
    q = torch.empty((tokens, world * inner), dtype=torch.float8_e4m3fn, device=x.device)
    s = torch.empty((tokens, 1), dtype=torch.float32, device=x.device)
    with torch.cuda.device(x.device):
        _merge_heads_per_token_quant_fp8_kernel[(tokens,)](
            x,
            q,
            s,
            seq,
            inner,
            world,
            seq * batch * inner,
            batch * inner,
            inner,
            FAST_MATH=reference_fast_math(),
            ZERO_GUARD=reference_zero_guard(tokens, x.device),
            # Fastest of the {512..4096} x {2,4,8} sweep on H200 at the
            # production shape [2, 20992, 1, 28, 128]; re-tune on shape change.
            BLOCK=2048,
            num_warps=4,
        )
    return q, s


def can_use_merge_two_sources_per_token_quant_fp8(
    first: torch.Tensor, second: torch.Tensor
) -> bool:
    return (
        first.is_cuda
        and first.dtype is torch.bfloat16
        and second.dtype is torch.bfloat16
        and first.device == second.device
        and first.dim() == 2
        and first.shape == second.shape
        and first.numel() > 0
        and first.stride(-1) == 1
        and second.stride(-1) == 1
        and first.stride(0) >= first.shape[-1]
        and second.stride(0) >= second.shape[-1]
    )


def merge_two_sources_per_token_quant_fp8(
    first: torch.Tensor, second: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate two ``[T, C]`` head-shard sources column-wise and quantize.

    ``first`` fills output columns ``[0, C)`` and ``second`` ``[C, 2C)``;
    returns ``(q, s)`` bitwise equal to quantizing
    ``torch.cat((first, second), dim=1)``.  This is the 2-rank IPC output
    merge with the two halves passed in output-column order.
    """
    if not can_use_merge_two_sources_per_token_quant_fp8(first, second):
        raise RuntimeError("unsupported inputs for fused two-source merge + fp8 quant")
    tokens, inner = first.shape
    q = torch.empty((tokens, 2 * inner), dtype=torch.float8_e4m3fn, device=first.device)
    s = torch.empty((tokens, 1), dtype=torch.float32, device=first.device)
    with torch.cuda.device(first.device):
        _merge_two_sources_per_token_quant_fp8_kernel[(tokens,)](
            first,
            second,
            q,
            s,
            inner,
            first.stride(0),
            second.stride(0),
            FAST_MATH=reference_fast_math(),
            ZERO_GUARD=reference_zero_guard(tokens, first.device),
            BLOCK=2048,
            num_warps=4,
        )
    return q, s
