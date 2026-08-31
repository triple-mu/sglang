# SPDX-License-Identifier: Apache-2.0
"""JIT C++ default for the indexed adaLN modulation (MiniMax-H3).

Dispatch order: C++ JIT (CUDA SM90+, bf16 contiguous rows) -> Triton
(``indexed_modulation_triton``). ROCm and pre-SM90 fail closed to Triton.
The eager fallback lives one level up at the model call site
(``_modulate_scale_shift``), which only routes CUDA bf16 inputs here.

Numerical contract: identical to the Triton kernel -- an RNE round to bf16
after ``1 + scale``, after the product, and on the final store, so the two
backends are bitwise interchangeable for finite values.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, is_hip_runtime, load_jit, make_cpp_args
from sglang.kernels.ops.diffusion.modulate.indexed_modulation_triton import (
    indexed_scale_shift_bf16_ as triton_indexed_scale_shift_bf16_,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module


_INDEX_DTYPES = (torch.int32, torch.int64)
_ALIGN_BYTES = 16
_VEC_ELEMS = 8  # 16B of bf16
# Measured at [20992, 5376] on H200 (bench_abab.py); re-tune when the shape
# family changes.
_BLOCK_THREADS = 224
_VECS_PER_THREAD = 3
_FAILED_RUNTIME_DEVICES: set[int | None] = set()

logger = logging.getLogger(__name__)


@cache_once
def _jit_indexed_scale_shift_module() -> Module:
    args = make_cpp_args(_BLOCK_THREADS, _VECS_PER_THREAD)
    return load_jit(
        "diffusion_indexed_scale_shift",
        *args,
        cuda_files=["diffusion/indexed_scale_shift.cuh"],
        cuda_wrappers=[
            (
                "indexed_scale_shift",
                f"indexed_scale_shift::IndexedScaleShiftKernel<{args}>::run",
            ),
        ],
    )


@torch.compiler.assume_constant_result
@cache_once
def _cuda_backend_available() -> bool:
    """SM90+ CUDA with a loadable build; ROCm and older archs fail closed."""
    if is_hip_runtime() or not torch.cuda.is_available():
        return False
    if torch.cuda.get_device_capability()[0] < 9:
        return False
    try:
        _jit_indexed_scale_shift_module()
        return True
    except Exception as exc:
        logger.warning("Failed to load JIT indexed_scale_shift kernel: %s", exc)
        return False


def can_use_indexed_scale_shift_cuda(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
) -> bool:
    return (
        _cuda_backend_available()
        and x.dtype is torch.bfloat16
        and x.is_cuda
        and x.dim() == 2
        and x.shape[0] > 0
        and x.shape[1] % _VEC_ELEMS == 0
        and x.is_contiguous()
        and shift.dtype is torch.bfloat16
        and scale.dtype is torch.bfloat16
        and shift.dim() == 2
        and shift.shape[1] == x.shape[1]
        and scale.shape == shift.shape
        and shift.is_contiguous()
        and scale.is_contiguous()
        and indices.dtype in _INDEX_DTYPES
        and indices.is_contiguous()
        and indices.shape == (x.shape[0],)
        and x.data_ptr() % _ALIGN_BYTES == 0
        and shift.data_ptr() % _ALIGN_BYTES == 0
        and scale.data_ptr() % _ALIGN_BYTES == 0
    )


def indexed_scale_shift_bf16_cuda(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """C++ backend only; raises on unsupported input instead of falling back."""
    if not can_use_indexed_scale_shift_cuda(x, shift, scale, indices):
        raise RuntimeError("unsupported input for indexed_scale_shift CUDA")
    _jit_indexed_scale_shift_module().indexed_scale_shift(x, shift, scale, indices)
    return x


def indexed_scale_shift_bf16_(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """In place ``x = x * (1 + scale[indices]) + shift[indices]`` (bf16 chain)."""
    device_key = x.device.index
    if device_key not in _FAILED_RUNTIME_DEVICES and can_use_indexed_scale_shift_cuda(
        x, shift, scale, indices
    ):
        try:
            _jit_indexed_scale_shift_module().indexed_scale_shift(
                x, shift, scale, indices
            )
            return x
        except Exception as exc:
            if torch.compiler.is_compiling():
                raise
            _FAILED_RUNTIME_DEVICES.add(device_key)
            logger.warning(
                "Disabling indexed_scale_shift CUDA fast path on %s: %s",
                x.device,
                exc,
            )
    return triton_indexed_scale_shift_bf16_(x, shift, scale, indices)


__all__ = [
    "can_use_indexed_scale_shift_cuda",
    "indexed_scale_shift_bf16_",
    "indexed_scale_shift_bf16_cuda",
]
