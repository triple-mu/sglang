# SPDX-License-Identifier: Apache-2.0
"""JIT C++ default for the fused packed SwiGLU + per-token FP8 quantization.

Dispatch order: C++ JIT -> Triton (``silu_mul_quant_triton``); ROCm and
non-SM90 CUDA fail closed to Triton because the C++ kernel replicates the
SM90 ``--use_fast_math`` build of the reference quant kernel and nothing
else (``fp8_quant_replica.reference_fast_math``).  The eager fallback stays
at the call site behind ``can_use_fused_silu_mul_per_token_quant_fp8``.

Numerical contract (unchanged from the Triton kernel): payload and scale are
bitwise equal to ``sgl_per_token_quant_fp8(F.silu(gate) * up)``, including
the reference's dispatch-dependent zero-scale semantics, which the C++
launcher re-derives from the row count.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_hip_runtime,
    load_jit,
    make_cpp_args,
)
from sglang.kernels.ops.diffusion.activation.silu_mul_quant_triton import (
    can_use_fused_silu_mul_per_token_quant_fp8,
)
from sglang.kernels.ops.diffusion.activation.silu_mul_quant_triton import (
    fused_silu_mul_per_token_quant_fp8 as triton_fused_silu_mul_per_token_quant_fp8,
)
from sglang.kernels.ops.diffusion.common.fp8_quant_replica import reference_fast_math

if TYPE_CHECKING:
    from tvm_ffi.module import Module

logger = logging.getLogger(__name__)

_VEC_ELEMS = 8  # 16B of bf16 per thread-load on the C++ side
_THREADS = 512  # must match silu_mul_quant::kThreads
_MAX_VECS_PER_THREAD = 8
_ALIGN_BYTES = 16
_FAILED_RUNTIME_DEVICES: set[int | None] = set()


def _vecs_per_thread(hidden: int) -> int:
    return -(-(hidden // _VEC_ELEMS) // _THREADS)


@cache_once
def _jit_silu_mul_quant_module(vecs_per_thread: int) -> Module:
    args = make_cpp_args(vecs_per_thread)
    return load_jit(
        "diffusion_silu_mul_quant",
        *args,
        cuda_files=["diffusion/silu_mul_quant.cuh"],
        cuda_wrappers=[
            (
                "silu_mul_quant",
                f"silu_mul_quant::SiluMulQuantKernel<{args}>::run",
            )
        ],
    )


@torch.compiler.assume_constant_result
@cache_once
def _cuda_backend_available(vecs_per_thread: int) -> bool:
    """SM90 CUDA with a loadable build; ROCm and other archs fail closed.

    The C++ kernel replicates the fast-math reference arithmetic verified on
    SM90; on any arch where the reference quant kernel is a precise-math
    build, the Triton replica (which carries both variants) stays default.
    """
    if is_hip_runtime() or not torch.cuda.is_available():
        return False
    if not reference_fast_math():
        return False
    try:
        _jit_silu_mul_quant_module(vecs_per_thread)
        return True
    except Exception as exc:
        logger.warning("Failed to load JIT silu_mul_quant kernel: %s", exc)
        return False


def can_use_silu_mul_quant_cuda(x: torch.Tensor) -> bool:
    """Whether the C++ backend handles this input (assumes the public
    ``can_use_fused_silu_mul_per_token_quant_fp8`` gate already passed)."""
    hidden = x.shape[-1] // 2
    return (
        hidden % _VEC_ELEMS == 0
        and _vecs_per_thread(hidden) <= _MAX_VECS_PER_THREAD
        and x.stride(0) % _VEC_ELEMS == 0
        and x.data_ptr() % _ALIGN_BYTES == 0
        and _cuda_backend_available(_vecs_per_thread(hidden))
    )


def silu_mul_quant_cuda(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """C++ backend only; raises on unsupported input instead of falling back."""
    if not (
        can_use_fused_silu_mul_per_token_quant_fp8(x) and can_use_silu_mul_quant_cuda(x)
    ):
        raise RuntimeError("unsupported input for fused SiLU-mul + fp8 quant CUDA")
    rows, packed = x.shape
    hidden = packed // 2
    q = torch.empty((rows, hidden), dtype=torch.float8_e4m3fn, device=x.device)
    s = torch.empty((rows, 1), dtype=torch.float32, device=x.device)
    _jit_silu_mul_quant_module(_vecs_per_thread(hidden)).silu_mul_quant(x, q, s)
    return q, s


def fused_silu_mul_per_token_quant_fp8(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``silu(gate) * up`` + per-token FP8 quant over ``x = [gate | up]``.

    Returns ``(q, s)`` with ``q`` fp8_e4m3 of shape ``[M, D]`` and ``s`` fp32
    of shape ``[M, 1]``, bitwise equal to feeding the eager activation into
    ``sgl_per_token_quant_fp8``.  C++ JIT when supported, Triton otherwise.
    """
    if not can_use_fused_silu_mul_per_token_quant_fp8(x):
        raise RuntimeError("unsupported input for fused SiLU-mul + fp8 quant")
    device_key = x.device.index
    if device_key not in _FAILED_RUNTIME_DEVICES and can_use_silu_mul_quant_cuda(x):
        try:
            return silu_mul_quant_cuda(x)
        except Exception as exc:
            if torch.compiler.is_compiling():
                raise
            _FAILED_RUNTIME_DEVICES.add(device_key)
            logger.warning(
                "Disabling silu_mul_quant CUDA fast path on %s: %s", x.device, exc
            )
    return triton_fused_silu_mul_per_token_quant_fp8(x)


__all__ = [
    "can_use_fused_silu_mul_per_token_quant_fp8",
    "can_use_silu_mul_quant_cuda",
    "fused_silu_mul_per_token_quant_fp8",
    "silu_mul_quant_cuda",
]
