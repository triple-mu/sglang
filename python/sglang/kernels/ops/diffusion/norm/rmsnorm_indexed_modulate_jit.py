# SPDX-License-Identifier: Apache-2.0
"""Fused RMSNorm + indexed adaLN scale/shift for the MiniMax-H3 DiT (bitexact).

Plan A replaces ``nn.RMSNorm -> indexed_scale_shift_bf16_``; Plan B also
absorbs the preceding ``indexed_gate_bf16_`` residual update. Both replicate
aten's bf16 RMSNorm reduction order and every eager rounding boundary, so the
outputs are bitwise equal to the eager chain on a torch build that dispatches
``nn.RMSNorm`` to ``vectorized_layer_norm_kernel``; the model verifies that on
its first fused call and falls back to eager for good otherwise. C++ only; the
callers gate on ``can_use_*`` and keep the eager path themselves.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, is_hip_runtime, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_INDEX_DTYPES = (torch.int32, torch.int64)
_ALIGN_BYTES = 16
_VEC_ELEMS = 4  # aten's vec_size for bf16 RMSNorm


@cache_once
def _jit_module(hidden_size: int) -> Module:
    args = make_cpp_args(hidden_size)
    kernel = f"rmsnorm_indexed_modulate::RMSNormIndexedModulateKernel<{args}>"
    return load_jit(
        "diffusion_rmsnorm_indexed_modulate",
        *args,
        cuda_files=["diffusion/rmsnorm_indexed_modulate.cuh"],
        cuda_wrappers=[
            ("rmsnorm_indexed_scale_shift", f"{kernel}::run"),
            ("gate_residual_rmsnorm_indexed_scale_shift", f"{kernel}::run_gated"),
        ],
    )


def _modulation_rows_ok(rows: torch.Tensor, hidden: int, stride: int) -> bool:
    return (
        rows.dtype is torch.bfloat16
        and rows.dim() == 2
        and rows.shape[1] == hidden
        and rows.stride(1) == 1
        and rows.stride(0) == stride
        and stride % _VEC_ELEMS == 0
        and rows.data_ptr() % _ALIGN_BYTES == 0
    )


def can_use_rmsnorm_indexed_scale_shift(
    x: torch.Tensor,
    gamma: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    indices: torch.Tensor,
) -> bool:
    """Shapes, dtypes, strides and alignment the kernel accepts; no build is attempted."""
    if is_hip_runtime() or not x.is_cuda:
        return False
    hidden = x.shape[-1]
    return (
        x.dtype is torch.bfloat16
        and x.dim() == 2
        and x.shape[0] > 0
        and hidden % _VEC_ELEMS == 0
        and x.is_contiguous()
        and x.data_ptr() % _ALIGN_BYTES == 0
        and gamma.dtype is torch.bfloat16
        and gamma.shape == (hidden,)
        and gamma.is_contiguous()
        and gamma.data_ptr() % _ALIGN_BYTES == 0
        and _modulation_rows_ok(scale, hidden, scale.stride(0))
        and _modulation_rows_ok(shift, hidden, scale.stride(0))
        and shift.shape == scale.shape
        and indices.dtype in _INDEX_DTYPES
        and indices.is_contiguous()
        and indices.shape == (x.shape[0],)
    )


def can_use_gate_residual_rmsnorm_indexed_scale_shift(
    residual: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    gamma: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    indices: torch.Tensor,
) -> bool:
    return (
        can_use_rmsnorm_indexed_scale_shift(residual, gamma, scale, shift, indices)
        and update.dtype is torch.bfloat16
        and update.shape == residual.shape
        and update.is_contiguous()
        and update.data_ptr() != residual.data_ptr()
        and update.data_ptr() % _ALIGN_BYTES == 0
        and _modulation_rows_ok(gate, residual.shape[-1], scale.stride(0))
        and gate.shape == scale.shape
    )


def rmsnorm_indexed_scale_shift(
    x: torch.Tensor,
    gamma: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    indices: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    """Plan A: ``modulate(rmsnorm(x), scale[indices], shift[indices])`` into a new tensor."""
    out = torch.empty_like(x)
    _jit_module(x.shape[1]).rmsnorm_indexed_scale_shift(
        out, x, gamma, scale, shift, indices, eps
    )
    return out


def gate_residual_rmsnorm_indexed_scale_shift_(
    residual: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    gamma: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    indices: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Plan B: ``residual += gate[indices] * update`` in place, then Plan A on it.

    Returns ``(out, residual)``.
    """
    out = torch.empty_like(residual)
    _jit_module(residual.shape[1]).gate_residual_rmsnorm_indexed_scale_shift(
        out, residual, update, gate, gamma, scale, shift, indices, eps
    )
    return out, residual


__all__ = [
    "can_use_gate_residual_rmsnorm_indexed_scale_shift",
    "can_use_rmsnorm_indexed_scale_shift",
    "gate_residual_rmsnorm_indexed_scale_shift_",
    "rmsnorm_indexed_scale_shift",
]
