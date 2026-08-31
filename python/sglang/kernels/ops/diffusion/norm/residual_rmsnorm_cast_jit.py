# SPDX-License-Identifier: Apache-2.0
"""JIT C++ kernel for the MiniMax-H3 VAE ViT decoder residual triple.

One launch replaces the eager three-kernel chain per block site:

    y = residual_fp32 + branch          (aten mixed-dtype add, fp32)
    n = RMSNorm_fp32(y)                 (aten vectorized rms kernel)
    out = n.to(autocast_dtype)          (the cast the next Linear would issue)

``residual`` is updated to ``y`` in place; the returned tensor is the
normalized output already in the autocast compute dtype. The reduction
replicates aten's ``vectorized_layer_norm_kernel<float, float, true>``
bit-for-bit (see residual_rmsnorm_cast.cuh), so callers may rely on
``torch.equal`` vs the eager chain. There is no Triton fallback -- the model
level falls back to the eager chain when this kernel does not apply.
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

if TYPE_CHECKING:
    from tvm_ffi.module import Module

logger = logging.getLogger(__name__)

_BRANCH_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_OUT_DTYPES = (torch.float16, torch.bfloat16)
_ALIGN_BYTES = 16


@cache_once
def _jit_residual_rmsnorm_cast_module(
    hidden_size: int, branch_dtype: torch.dtype, out_dtype: torch.dtype
) -> Module:
    args = make_cpp_args(hidden_size, branch_dtype, out_dtype)
    return load_jit(
        "diffusion_residual_rmsnorm_cast",
        *args,
        cuda_files=["diffusion/residual_rmsnorm_cast.cuh"],
        cuda_wrappers=[
            (
                "residual_rmsnorm_cast",
                f"residual_rmsnorm_cast::ResidualRMSNormCastKernel<{args}>::run",
            )
        ],
    )


@torch.compiler.assume_constant_result
@cache_once
def can_use_fused_residual_rmsnorm_cast(
    hidden_size: int, branch_dtype: torch.dtype, out_dtype: torch.dtype
) -> bool:
    """CUDA-only: the kernel replicates aten's CUDA rms reduction; ROCm's aten
    dispatches a different kernel, so it fails closed to the eager chain."""
    if is_hip_runtime() or not torch.cuda.is_available():
        return False
    # aten's vectorized rms kernel (the replicated oracle) requires % 4 rows.
    if hidden_size <= 0 or hidden_size % 4 != 0:
        return False
    if branch_dtype not in _BRANCH_DTYPES or out_dtype not in _OUT_DTYPES:
        return False
    try:
        _jit_residual_rmsnorm_cast_module(hidden_size, branch_dtype, out_dtype)
        return True
    except Exception as exc:
        logger.warning("Failed to load JIT residual_rmsnorm_cast kernel: %s", exc)
        return False


def fused_residual_rmsnorm_cast_(
    residual: torch.Tensor,
    branch: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Update ``residual += branch`` in place and return the normalized rows
    cast to ``out_dtype``. Tensor invariants are enforced by the C++ launcher;
    callers gate applicability via ``can_use_fused_residual_rmsnorm_cast``."""
    hidden_size = residual.shape[-1]
    out = torch.empty(residual.shape, dtype=out_dtype, device=residual.device)
    module = _jit_residual_rmsnorm_cast_module(hidden_size, branch.dtype, out_dtype)
    module.residual_rmsnorm_cast(
        out.view(-1, hidden_size),
        residual.view(-1, hidden_size),
        branch.view(-1, hidden_size),
        weight,
        eps,
    )
    return out


__all__ = [
    "can_use_fused_residual_rmsnorm_cast",
    "fused_residual_rmsnorm_cast_",
]
