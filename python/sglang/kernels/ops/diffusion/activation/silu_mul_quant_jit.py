# SPDX-License-Identifier: Apache-2.0
"""Fused packed SwiGLU + per-token FP8 quantisation for the MiniMax-H3 fp8 MLP.

``x = [gate | up]`` in bf16 becomes ``(q, scale)``: fp8 e4m3 rows and one fp32
scale per row, ready for the per-row-scaled fp8 GEMM. The activation keeps the
eager ``silu(gate) * up`` bf16 rounding; the quantisation is the plain
``amax / 448`` per-token scheme.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, is_hip_runtime, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_VEC_ELEMS = 8
_THREADS = 512  # silu_mul_quant::kThreads
_MAX_VECS_PER_THREAD = 8
_ALIGN_BYTES = 16


def _vecs_per_thread(hidden: int) -> int:
    return -(-(hidden // _VEC_ELEMS) // _THREADS)


@cache_once
def _jit_module(vecs_per_thread: int) -> Module:
    args = make_cpp_args(vecs_per_thread)
    return load_jit(
        "diffusion_silu_mul_quant",
        *args,
        cuda_files=["diffusion/silu_mul_quant.cuh"],
        cuda_wrappers=[
            ("silu_mul_quant", f"silu_mul_quant::SiluMulQuantKernel<{args}>::run")
        ],
    )


def can_use_silu_mul_per_token_quant_fp8(x: torch.Tensor) -> bool:
    if is_hip_runtime() or not x.is_cuda:
        return False
    if x.dtype is not torch.bfloat16 or x.dim() != 2 or x.shape[-1] % (2 * _VEC_ELEMS):
        return False
    hidden = x.shape[-1] // 2
    return (
        x.shape[0] > 0
        and x.stride(1) == 1
        and x.stride(0) % _VEC_ELEMS == 0
        and x.data_ptr() % _ALIGN_BYTES == 0
        and _vecs_per_thread(hidden) <= _MAX_VECS_PER_THREAD
    )


def silu_mul_per_token_quant_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(q [M, D] fp8_e4m3, scale [M, 1] fp32)`` from ``x = [gate | up]`` of width 2D."""
    rows, packed = x.shape
    hidden = packed // 2
    q = torch.empty((rows, hidden), dtype=torch.float8_e4m3fn, device=x.device)
    s = torch.empty((rows, 1), dtype=torch.float32, device=x.device)
    _jit_module(_vecs_per_thread(hidden)).silu_mul_quant(x, q, s)
    return q, s


__all__ = ["can_use_silu_mul_per_token_quant_fp8", "silu_mul_per_token_quant_fp8"]
