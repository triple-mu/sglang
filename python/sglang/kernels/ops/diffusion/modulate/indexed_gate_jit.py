# SPDX-License-Identifier: Apache-2.0
"""JIT C++ default for the indexed gated residual (MiniMax-H3).

Dispatch order: C++ JIT (CUDA SM90+, bf16 contiguous rows) -> Triton
(``indexed_modulation_triton``). ROCm and pre-SM90 fail closed to Triton.
The eager fallback lives one level up at the model call site
(``_modulate_gate``), which only routes CUDA bf16 inputs here.

Numerical contract: identical to the Triton kernel -- the product is
RNE-rounded to bf16 precision (``round_bf16(gate[idx] * other)``) and the sum
is stored with one final RNE fp32->bf16 conversion, so the two backends are
bitwise interchangeable for finite values. The fused Plan B kernel in
``norm/rmsnorm_indexed_modulate_triton.py`` asserts bit-equality against this
op's residual write-back.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, is_hip_runtime, load_jit
from sglang.kernels.ops.diffusion.modulate.indexed_modulation_triton import (
    _indexed_gate_bf16 as _triton_indexed_gate_bf16,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module


_INDEX_DTYPES = (torch.int32, torch.int64)
_ALIGN_BYTES = 16
_VEC_ELEMS = 8  # 16B of bf16
_FAILED_RUNTIME_DEVICES: set[int | None] = set()

logger = logging.getLogger(__name__)


@cache_once
def _jit_indexed_gate_module() -> Module:
    return load_jit(
        "diffusion_indexed_gate",
        cuda_files=["diffusion/indexed_gate.cuh"],
        cuda_wrappers=[
            ("indexed_gate", "indexed_gate::IndexedGateKernel::run"),
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
        _jit_indexed_gate_module()
        return True
    except Exception as exc:
        logger.warning("Failed to load JIT indexed_gate kernel: %s", exc)
        return False


def can_use_indexed_gate_bf16_cuda(
    x: torch.Tensor,
    gate: torch.Tensor,
    other: torch.Tensor,
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
        and gate.dtype is torch.bfloat16
        and other.dtype is torch.bfloat16
        and gate.dim() == 2
        and gate.shape[1] == x.shape[1]
        and other.shape == x.shape
        and gate.is_contiguous()
        and other.is_contiguous()
        and indices.dtype in _INDEX_DTYPES
        and indices.is_contiguous()
        and indices.shape == (x.shape[0],)
        and x.data_ptr() % _ALIGN_BYTES == 0
        and gate.data_ptr() % _ALIGN_BYTES == 0
        and other.data_ptr() % _ALIGN_BYTES == 0
    )


def _indexed_gate_bf16_dispatch(
    output: torch.Tensor,
    x: torch.Tensor,
    gate: torch.Tensor,
    other: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    device_key = x.device.index
    if (
        device_key not in _FAILED_RUNTIME_DEVICES
        and (
            output is x
            or (output.is_contiguous() and output.data_ptr() % _ALIGN_BYTES == 0)
        )
        and can_use_indexed_gate_bf16_cuda(x, gate, other, indices)
    ):
        try:
            _jit_indexed_gate_module().indexed_gate(output, x, gate, other, indices)
            return output
        except Exception as exc:
            if torch.compiler.is_compiling():
                raise
            _FAILED_RUNTIME_DEVICES.add(device_key)
            logger.warning(
                "Disabling indexed_gate CUDA fast path on %s: %s", x.device, exc
            )
    return _triton_indexed_gate_bf16(output, x, gate, other, indices)


def indexed_gate_bf16_(
    x: torch.Tensor,
    gate: torch.Tensor,
    other: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """In place ``x = x + round_bf16(gate[indices] * other)`` (bf16 chain)."""
    return _indexed_gate_bf16_dispatch(x, x, gate, other, indices)


def indexed_gate_bf16(
    x: torch.Tensor,
    gate: torch.Tensor,
    other: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Out-of-place form of :func:`indexed_gate_bf16_`."""
    return _indexed_gate_bf16_dispatch(torch.empty_like(x), x, gate, other, indices)


__all__ = [
    "can_use_indexed_gate_bf16_cuda",
    "indexed_gate_bf16",
    "indexed_gate_bf16_",
]
