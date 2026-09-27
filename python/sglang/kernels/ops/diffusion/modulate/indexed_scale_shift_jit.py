# SPDX-License-Identifier: Apache-2.0
"""Indexed adaLN modulation with a chosen output dtype (MiniMax-H3 final layer).

Bitwise equal to the Triton ``indexed_scale_shift_bf16_`` chain; an fp32 output
folds the ``.to(float32)`` the final layer used to run as a second pass.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, is_hip_runtime, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_INDEX_DTYPES = (torch.int32, torch.int64)
_ALIGN_BYTES = 16
_VEC_ELEMS = 8
# Measured at [20992, 5376] on H200 in the fusion branch; 5376 / 8 = 672 vectors
# = exactly three 224-thread blocks.
_BLOCK_THREADS = 224
_VECS_PER_THREAD = 3
_OUT_TYPES = {torch.bfloat16: "bf16_t", torch.float32: "fp32_t"}


@cache_once
def _jit_module(out_dtype: torch.dtype) -> Module:
    cpp_type = _OUT_TYPES[out_dtype]
    return load_jit(
        "diffusion_indexed_scale_shift",
        cpp_type,
        cuda_files=["diffusion/indexed_scale_shift.cuh"],
        cuda_wrappers=[
            (
                "indexed_scale_shift",
                f"indexed_scale_shift::IndexedScaleShiftKernel<{cpp_type}, "
                f"{make_cpp_args(_BLOCK_THREADS, _VECS_PER_THREAD)}>::run",
            ),
        ],
    )


def can_use_indexed_scale_shift(
    x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, indices: torch.Tensor
) -> bool:
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
        and shift.dtype is torch.bfloat16
        and scale.dtype is torch.bfloat16
        and shift.dim() == 2
        and shift.shape[1] == hidden
        and scale.shape == shift.shape
        and shift.stride(1) == 1
        and scale.stride(1) == 1
        and shift.stride(0) == scale.stride(0)
        and shift.stride(0) % _VEC_ELEMS == 0
        and shift.data_ptr() % _ALIGN_BYTES == 0
        and scale.data_ptr() % _ALIGN_BYTES == 0
        and indices.dtype in _INDEX_DTYPES
        and indices.is_contiguous()
        and indices.shape == (x.shape[0],)
    )


def indexed_scale_shift(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    *,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """``x * (1 + scale[indices]) + shift[indices]`` in the bf16 chain, stored as `out_dtype`."""
    if out_dtype not in _OUT_TYPES:
        raise ValueError(f"indexed_scale_shift stores bf16 or fp32, not {out_dtype}")
    out = x if out_dtype is torch.bfloat16 else torch.empty_like(x, dtype=out_dtype)
    _jit_module(out_dtype).indexed_scale_shift(out, x, shift, scale, indices)
    return out


__all__ = ["can_use_indexed_scale_shift", "indexed_scale_shift"]
