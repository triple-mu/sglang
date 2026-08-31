# SPDX-License-Identifier: Apache-2.0
"""Channels-last two-pass GroupNorm(+SiLU): C++ JIT default, Triton fallback.

Same public surface and numerical contract as
``group_norm_silu_twopass_triton`` (fp32 statistics, folded affine, optional
SiLU epilogue, tolerance vs the eager oracle); the support predicates are
re-exported unchanged.  Dispatch order is C++ JIT -> Triton: ROCm, unsupported
channel widths, and JIT build failures fail closed to the Triton kernels.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.kernels.ops.diffusion.norm.group_norm_silu_twopass_triton import (
    _gn_silu_rows as _triton_gn_silu_rows,
)
from sglang.kernels.ops.diffusion.norm.group_norm_silu_twopass_triton import (
    can_use_group_norm_silu_4d,
    can_use_group_norm_silu_rows,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

logger = logging.getLogger(__name__)

_SUPPORTED_DTYPES = (torch.float32, torch.bfloat16, torch.float16)
_ALIGN_BYTES = 16
_FAILED_RUNTIME_KEYS: set[tuple[int | None, torch.dtype]] = set()


@cache_once
def _jit_group_norm_silu_module(dtype: torch.dtype) -> Module:
    if torch.version.hip is not None:
        raise RuntimeError("group_norm_silu twopass CUDA kernel is CUDA-only")
    if dtype not in _SUPPORTED_DTYPES:
        raise RuntimeError(f"Unsupported group_norm_silu dtype: {dtype}")
    args = make_cpp_args(dtype)
    return load_jit(
        "diffusion_group_norm_silu_twopass",
        *args,
        cuda_files=["diffusion/group_norm_silu_twopass.cuh"],
        cuda_wrappers=[
            (
                "group_norm_silu_rows",
                f"group_norm_silu_twopass::GroupNormSiluTwopassKernel<{args}>::run",
            ),
        ],
    )


def _can_use_cpp(x3: torch.Tensor) -> bool:
    # The C++ kernel's extra constraints on top of the shared twopass
    # predicate: 16B-vector channel alignment, grid.y batch bound, and a
    # loadable module (ROCm and build failures fail closed to Triton).
    runtime_key = (x3.device.index, x3.dtype)
    if runtime_key in _FAILED_RUNTIME_KEYS:
        return False
    vec = _ALIGN_BYTES // x3.element_size()
    if (
        torch.version.hip is not None
        or torch.compiler.is_compiling()
        or x3.shape[-1] % vec != 0
        or x3.shape[0] > 65535
        or x3.data_ptr() % _ALIGN_BYTES != 0
    ):
        return False
    try:
        _jit_group_norm_silu_module(x3.dtype)
        return True
    except Exception as exc:
        _FAILED_RUNTIME_KEYS.add(runtime_key)
        logger.warning(
            "Disabling group_norm_silu twopass CUDA fast path on %s/%s: %s",
            x3.device,
            x3.dtype,
            exc,
        )
        return False


def _gn_silu_rows_cpp(x3, weight, bias, num_groups, eps, apply_silu):
    y3 = torch.empty_like(x3)
    module = _jit_group_norm_silu_module(x3.dtype)
    module.group_norm_silu_rows(x3, y3, weight, bias, num_groups, eps, apply_silu)
    return y3


def _gn_silu_rows_dispatch(x3, weight, bias, num_groups, eps, apply_silu):
    if _can_use_cpp(x3):
        try:
            return _gn_silu_rows_cpp(x3, weight, bias, num_groups, eps, apply_silu)
        except Exception as exc:
            _FAILED_RUNTIME_KEYS.add((x3.device.index, x3.dtype))
            logger.warning(
                "group_norm_silu twopass CUDA fast path failed on %s/%s, "
                "falling back to Triton: %s",
                x3.device,
                x3.dtype,
                exc,
            )
    return _triton_gn_silu_rows(x3, weight, bias, num_groups, eps, apply_silu)


def group_norm_silu_4d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    num_groups: int,
    eps: float,
    apply_silu: bool = True,
) -> torch.Tensor:
    """Fused GroupNorm(+SiLU) for a channels_last 4D (N, C, H, W) activation.

    Runs on the free (N, H*W, C) view (no layout copy) and preserves the
    channels_last output layout.  Guard with
    :func:`can_use_group_norm_silu_4d`.
    """
    if not can_use_group_norm_silu_4d(x, weight, bias, num_groups):
        raise ValueError("unsupported input for group_norm_silu_4d")
    n_batch, c, h, w = x.shape
    x3 = x.permute(0, 2, 3, 1).reshape(n_batch, h * w, c)
    y3 = _gn_silu_rows_dispatch(x3, weight, bias, num_groups, eps, apply_silu)
    return y3.reshape(n_batch, h, w, c).permute(0, 3, 1, 2)


def group_norm_silu_rows(
    x3: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    num_groups: int,
    eps: float,
    apply_silu: bool = True,
) -> torch.Tensor:
    """Fused GroupNorm(+SiLU) over (N, L, C) rows (C = channels, innermost).

    Guard with :func:`can_use_group_norm_silu_rows`.
    """
    if not can_use_group_norm_silu_rows(x3, weight, bias, num_groups):
        raise ValueError("unsupported input for group_norm_silu_rows")
    return _gn_silu_rows_dispatch(x3, weight, bias, num_groups, eps, apply_silu)


__all__ = [
    "can_use_group_norm_silu_4d",
    "can_use_group_norm_silu_rows",
    "group_norm_silu_4d",
    "group_norm_silu_rows",
]
