# SPDX-License-Identifier: Apache-2.0
"""JIT C++ default for the fused RMSNorm + indexed adaLN chain (MiniMax-H3).

One kernel body covers both plans (see ``rmsnorm_indexed_modulate_triton``):
Plan A ``rmsnorm_indexed_scale_shift`` and Plan B
``gate_residual_rmsnorm_indexed_scale_shift_``. Dispatch order: C++ JIT (CUDA
SM90+, bf16 contiguous rows) -> Triton; ROCm and pre-SM90 fail closed to
Triton. The eager fallback lives at the model level -- the fused block loop
only engages on CUDA bf16 contiguous hidden states.

Two norm/modulate output contracts:

- Merged-``w_eff`` entry points (near-lossless): the Plan B residual
  write-back replicates the eager ``indexed_gate_bf16_`` rounding chain
  bitwise; the norm/modulate output keeps the Triton kernel's near-lossless
  contract (fp32 sum of squares, one bf16 round on the store) with an
  implementation-specific reduction tree.
- ``*_bitexact`` entry points (unmerged bf16 ``gamma`` + ``scale``): bitwise
  vs the eager chain ``nn.RMSNorm -> indexed_scale_shift_bf16_`` (plus
  ``indexed_gate_bf16_`` in front for Plan B) by replicating aten's
  ``vectorized_layer_norm_kernel<BFloat16, float, true>`` reduction order and
  every eager bf16 rounding boundary. C++ only -- there is no Triton twin, so
  callers gate on ``can_use_*`` and fall back to the eager kernels themselves.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, is_hip_runtime, load_jit, make_cpp_args
from sglang.kernels.ops.diffusion.norm.rmsnorm_indexed_modulate_triton import (
    gate_residual_rmsnorm_indexed_scale_shift_ as triton_gate_residual_rmsnorm_indexed_scale_shift_,
)
from sglang.kernels.ops.diffusion.norm.rmsnorm_indexed_modulate_triton import (
    rmsnorm_indexed_scale_shift as triton_rmsnorm_indexed_scale_shift,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module


_INDEX_DTYPES = (torch.int32, torch.int64)
_ALIGN_BYTES = 16
_VEC_ELEMS = 8  # 16B of bf16
# Measured at [20992, 5376] on H200 with the packed-bf16x2 kernel body
# ({128,224,256,352,448,512,672} sweep): 352 threads is fastest for Plan B
# (223us vs 264 Triton) and ties 224 for Plan A (115us vs 130); the row
# remainder is handled by the in-kernel bounds check, so exact tiling is not
# required. Rows narrower than the preferred block round down to whole warps.
_PREFERRED_THREADS = 352
_FAILED_RUNTIME_DEVICES: set[int | None] = set()

logger = logging.getLogger(__name__)


def _block_threads(hidden_size: int) -> int:
    vecs = hidden_size // _VEC_ELEMS
    return min(_PREFERRED_THREADS, vecs - vecs % 32)


@cache_once
def _jit_rmsnorm_indexed_modulate_module(hidden_size: int, threads: int) -> Module:
    args = make_cpp_args(hidden_size, threads)
    kernel = f"rmsnorm_indexed_modulate::RMSNormIndexedModulateKernel<{args}>"
    # The aten-order bitexact variant pins its own (32, 4) block geometry, so
    # it is keyed on the hidden size alone.
    aten = f"rmsnorm_indexed_modulate::RMSNormIndexedModulateAtenKernel<{hidden_size}>"
    return load_jit(
        "diffusion_rmsnorm_indexed_modulate",
        *args,
        cuda_files=["diffusion/rmsnorm_indexed_modulate.cuh"],
        cuda_wrappers=[
            ("rmsnorm_indexed_scale_shift", f"{kernel}::run"),
            ("gate_residual_rmsnorm_indexed_scale_shift", f"{kernel}::run_gated"),
            ("rmsnorm_indexed_scale_shift_bitexact", f"{aten}::run"),
            (
                "gate_residual_rmsnorm_indexed_scale_shift_bitexact",
                f"{aten}::run_gated",
            ),
        ],
    )


@torch.compiler.assume_constant_result
@cache_once
def _cuda_backend_available(hidden_size: int, threads: int) -> bool:
    """SM90+ CUDA with a loadable build; ROCm and older archs fail closed."""
    if is_hip_runtime() or not torch.cuda.is_available():
        return False
    if torch.cuda.get_device_capability()[0] < 9:
        return False
    if hidden_size % _VEC_ELEMS != 0:
        return False
    # The block must not out-count the row's 16B vectors (one vector per
    # thread minimum, enforced by a static_assert on the C++ side).
    if threads < 32 or hidden_size // _VEC_ELEMS < threads:
        return False
    try:
        _jit_rmsnorm_indexed_modulate_module(hidden_size, threads)
        return True
    except Exception as exc:
        logger.warning("Failed to load JIT rmsnorm_indexed_modulate kernel: %s", exc)
        return False


def _rows_supported(
    x: torch.Tensor,
    weight_eff: torch.Tensor,
    shift: torch.Tensor,
    indices: torch.Tensor,
    threads: int,
) -> bool:
    return (
        x.dtype is torch.bfloat16
        and x.is_cuda
        and x.dim() == 2
        and x.shape[0] > 0
        and x.is_contiguous()
        and _cuda_backend_available(x.shape[1], threads)
        and weight_eff.dtype is torch.float32
        and weight_eff.dim() == 2
        and weight_eff.shape[1] == x.shape[1]
        and weight_eff.is_contiguous()
        and shift.dtype is torch.bfloat16
        and shift.shape == weight_eff.shape
        and shift.is_contiguous()
        and indices.dtype in _INDEX_DTYPES
        and indices.is_contiguous()
        and indices.shape == (x.shape[0],)
        and x.data_ptr() % _ALIGN_BYTES == 0
        and weight_eff.data_ptr() % _ALIGN_BYTES == 0
        and shift.data_ptr() % _ALIGN_BYTES == 0
    )


def _plan_a_threads(hidden_size: int) -> int:
    return _block_threads(hidden_size)


def _plan_b_threads(hidden_size: int) -> int:
    return _block_threads(hidden_size)


def can_use_rmsnorm_indexed_scale_shift_cuda(
    x: torch.Tensor,
    weight_eff: torch.Tensor,
    shift: torch.Tensor,
    indices: torch.Tensor,
) -> bool:
    return _rows_supported(
        x, weight_eff, shift, indices, threads=_plan_a_threads(x.shape[-1])
    )


def can_use_gate_residual_rmsnorm_indexed_scale_shift_cuda(
    residual: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    weight_eff: torch.Tensor,
    shift: torch.Tensor,
    indices: torch.Tensor,
) -> bool:
    return (
        _rows_supported(
            residual,
            weight_eff,
            shift,
            indices,
            threads=_plan_b_threads(residual.shape[-1]),
        )
        and update.dtype is torch.bfloat16
        and update.shape == residual.shape
        and update.is_contiguous()
        and update.data_ptr() != residual.data_ptr()
        and gate.dtype is torch.bfloat16
        and gate.shape == weight_eff.shape
        and gate.is_contiguous()
        and update.data_ptr() % _ALIGN_BYTES == 0
        and gate.data_ptr() % _ALIGN_BYTES == 0
    )


def rmsnorm_indexed_scale_shift_cuda(
    x: torch.Tensor,
    weight_eff: torch.Tensor,
    shift: torch.Tensor,
    indices: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    """C++ backend only; raises on unsupported input instead of falling back."""
    if not can_use_rmsnorm_indexed_scale_shift_cuda(x, weight_eff, shift, indices):
        raise RuntimeError("unsupported input for rmsnorm_indexed_scale_shift CUDA")
    out = torch.empty_like(x)
    module = _jit_rmsnorm_indexed_modulate_module(
        x.shape[1], _plan_a_threads(x.shape[1])
    )
    module.rmsnorm_indexed_scale_shift(out, x, weight_eff, shift, indices, eps)
    return out


def gate_residual_rmsnorm_indexed_scale_shift_cuda_(
    residual: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    weight_eff: torch.Tensor,
    shift: torch.Tensor,
    indices: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """C++ backend only; raises on unsupported input instead of falling back."""
    if not can_use_gate_residual_rmsnorm_indexed_scale_shift_cuda(
        residual, update, gate, weight_eff, shift, indices
    ):
        raise RuntimeError(
            "unsupported input for gate_residual_rmsnorm_indexed_scale_shift CUDA"
        )
    out = torch.empty_like(residual)
    module = _jit_rmsnorm_indexed_modulate_module(
        residual.shape[1], _plan_b_threads(residual.shape[1])
    )
    module.gate_residual_rmsnorm_indexed_scale_shift(
        out, residual, update, gate, weight_eff, shift, indices, eps
    )
    return out, residual


def _bitexact_rows_supported(
    x: torch.Tensor,
    gamma: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    indices: torch.Tensor,
) -> bool:
    return (
        x.dtype is torch.bfloat16
        and x.is_cuda
        and x.dim() == 2
        and x.shape[0] > 0
        and x.is_contiguous()
        and _cuda_backend_available(x.shape[1], _block_threads(x.shape[1]))
        and gamma.dtype is torch.bfloat16
        and gamma.shape == (x.shape[1],)
        and gamma.is_contiguous()
        and scale.dtype is torch.bfloat16
        and scale.dim() == 2
        and scale.shape[1] == x.shape[1]
        and scale.is_contiguous()
        and shift.dtype is torch.bfloat16
        and shift.shape == scale.shape
        and shift.is_contiguous()
        and indices.dtype in _INDEX_DTYPES
        and indices.is_contiguous()
        and indices.shape == (x.shape[0],)
        and x.data_ptr() % _ALIGN_BYTES == 0
        and gamma.data_ptr() % _ALIGN_BYTES == 0
        and scale.data_ptr() % _ALIGN_BYTES == 0
        and shift.data_ptr() % _ALIGN_BYTES == 0
    )


def can_use_rmsnorm_indexed_scale_shift_bitexact_cuda(
    x: torch.Tensor,
    gamma: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    indices: torch.Tensor,
) -> bool:
    return _bitexact_rows_supported(x, gamma, scale, shift, indices)


def can_use_gate_residual_rmsnorm_indexed_scale_shift_bitexact_cuda(
    residual: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    gamma: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    indices: torch.Tensor,
) -> bool:
    return (
        _bitexact_rows_supported(residual, gamma, scale, shift, indices)
        and update.dtype is torch.bfloat16
        and update.shape == residual.shape
        and update.is_contiguous()
        and update.data_ptr() != residual.data_ptr()
        and gate.dtype is torch.bfloat16
        and gate.shape == scale.shape
        and gate.is_contiguous()
        and update.data_ptr() % _ALIGN_BYTES == 0
        and gate.data_ptr() % _ALIGN_BYTES == 0
    )


def rmsnorm_indexed_scale_shift_bitexact(
    x: torch.Tensor,
    gamma: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    indices: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    """Plan A, bitwise vs eager ``nn.RMSNorm -> indexed_scale_shift_bf16_``.

    C++ backend only; raises on unsupported input instead of falling back --
    callers gate on ``can_use_rmsnorm_indexed_scale_shift_bitexact_cuda``.
    """
    if not can_use_rmsnorm_indexed_scale_shift_bitexact_cuda(
        x, gamma, scale, shift, indices
    ):
        raise RuntimeError(
            "unsupported input for rmsnorm_indexed_scale_shift_bitexact CUDA"
        )
    out = torch.empty_like(x)
    module = _jit_rmsnorm_indexed_modulate_module(
        x.shape[1], _plan_a_threads(x.shape[1])
    )
    module.rmsnorm_indexed_scale_shift_bitexact(
        out, x, gamma, scale, shift, indices, eps
    )
    return out


def gate_residual_rmsnorm_indexed_scale_shift_bitexact_(
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
    """Plan B, bitwise vs eager ``indexed_gate_bf16_`` then the Plan A chain;
    ``residual`` is updated in place and returned as ``(out, residual)``.

    C++ backend only; raises on unsupported input instead of falling back.
    """
    if not can_use_gate_residual_rmsnorm_indexed_scale_shift_bitexact_cuda(
        residual, update, gate, gamma, scale, shift, indices
    ):
        raise RuntimeError(
            "unsupported input for gate_residual_rmsnorm_indexed_scale_shift_bitexact"
            " CUDA"
        )
    out = torch.empty_like(residual)
    module = _jit_rmsnorm_indexed_modulate_module(
        residual.shape[1], _plan_b_threads(residual.shape[1])
    )
    module.gate_residual_rmsnorm_indexed_scale_shift_bitexact(
        out, residual, update, gate, gamma, scale, shift, indices, eps
    )
    return out, residual


def rmsnorm_indexed_scale_shift(
    x: torch.Tensor,
    weight_eff: torch.Tensor,
    shift: torch.Tensor,
    indices: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    """Plan A: ``(x / rms(x)) * w_eff[indices] + shift[indices]``."""
    device_key = x.device.index
    if device_key not in _FAILED_RUNTIME_DEVICES and (
        can_use_rmsnorm_indexed_scale_shift_cuda(x, weight_eff, shift, indices)
    ):
        try:
            return rmsnorm_indexed_scale_shift_cuda(
                x, weight_eff, shift, indices, eps=eps
            )
        except Exception as exc:
            if torch.compiler.is_compiling():
                raise
            _FAILED_RUNTIME_DEVICES.add(device_key)
            logger.warning(
                "Disabling rmsnorm_indexed_modulate CUDA fast path on %s: %s",
                x.device,
                exc,
            )
    return triton_rmsnorm_indexed_scale_shift(x, weight_eff, shift, indices, eps=eps)


def gate_residual_rmsnorm_indexed_scale_shift_(
    residual: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    weight_eff: torch.Tensor,
    shift: torch.Tensor,
    indices: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Plan B: gated residual add (in place, bitwise vs ``indexed_gate_bf16_``)
    fused with the following norm/modulate; returns ``(out, residual)``."""
    device_key = residual.device.index
    if device_key not in _FAILED_RUNTIME_DEVICES and (
        can_use_gate_residual_rmsnorm_indexed_scale_shift_cuda(
            residual, update, gate, weight_eff, shift, indices
        )
    ):
        try:
            return gate_residual_rmsnorm_indexed_scale_shift_cuda_(
                residual, update, gate, weight_eff, shift, indices, eps=eps
            )
        except Exception as exc:
            if torch.compiler.is_compiling():
                raise
            _FAILED_RUNTIME_DEVICES.add(device_key)
            logger.warning(
                "Disabling rmsnorm_indexed_modulate CUDA fast path on %s: %s",
                residual.device,
                exc,
            )
    return triton_gate_residual_rmsnorm_indexed_scale_shift_(
        residual, update, gate, weight_eff, shift, indices, eps=eps
    )


__all__ = [
    "can_use_gate_residual_rmsnorm_indexed_scale_shift_bitexact_cuda",
    "can_use_gate_residual_rmsnorm_indexed_scale_shift_cuda",
    "can_use_rmsnorm_indexed_scale_shift_bitexact_cuda",
    "can_use_rmsnorm_indexed_scale_shift_cuda",
    "gate_residual_rmsnorm_indexed_scale_shift_",
    "gate_residual_rmsnorm_indexed_scale_shift_bitexact_",
    "gate_residual_rmsnorm_indexed_scale_shift_cuda_",
    "rmsnorm_indexed_scale_shift",
    "rmsnorm_indexed_scale_shift_bitexact",
    "rmsnorm_indexed_scale_shift_cuda",
]
