# SPDX-License-Identifier: Apache-2.0
"""USP output head-merge + per-token FP8 quant: C++ JIT default, Triton fallback.

Same public surface and numerical contract as ``usp_merge_quant_triton``:
payload and scale are **bitwise equal to the separated chain**
(``sgl_per_token_quant_fp8`` over the merged bf16 rows).  Both entry points
run register-staged one-CTA-per-token kernels from ``usp_merge_quant_fp8.cuh``
that replicate the SM90 fast-math reference's machine ops explicitly (FTZ bit
ops, ``rcp.approx.f32``), gather each token's shards into registers once and
quantize from there -- one DRAM read instead of the Triton replica's two.

Anything the kernels do not cover (ROCm, non-SM90 archs, unaligned rows, rows
wider than the register budget, JIT build failures) fails closed to the Triton
fast-math replica, which carries the contract on every platform the reference
supports.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.kernels.ops.diffusion.common.fp8_quant_replica import (
    reference_fast_math,
    reference_zero_guard,
)
from sglang.kernels.ops.diffusion.layout.usp_merge_quant_triton import (
    can_use_merge_two_sources_per_token_quant_fp8,
    can_use_usp_merge_heads_per_token_quant_fp8,
)
from sglang.kernels.ops.diffusion.layout.usp_merge_quant_triton import (
    merge_two_sources_per_token_quant_fp8 as _triton_merge_two_sources,
)
from sglang.kernels.ops.diffusion.layout.usp_merge_quant_triton import (
    usp_merge_heads_per_token_quant_fp8 as _triton_merge_heads,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

logger = logging.getLogger(__name__)

_ALIGN_BYTES = 16
_VEC_ELEMS = 8  # 16B / bf16
_BLOCK_SIZE = 256
# Retained 8-element chunks per thread; bounds the merged row at 16384 elements.
_MAX_CHUNKS = 8


# --- head-merge (NCCL all-to-all output) form


@cache_once
def _jit_merge_heads_module(chunks_per_thread: int) -> Module:
    args = make_cpp_args(_BLOCK_SIZE, chunks_per_thread, is_arch_support_pdl())
    return load_jit(
        "diffusion_usp_merge_heads_quant_fp8",
        *args,
        cuda_files=["diffusion/usp_merge_quant_fp8.cuh"],
        cuda_wrappers=[
            (
                "usp_merge_heads_per_token_quant_fp8",
                "usp_merge_quant_fp8::" f"MergeHeadsQuantKernel<{args}>::run",
            )
        ],
    )


@cache_once
def _merge_heads_module_or_none(chunks_per_thread: int) -> Module | None:
    """The compiled head-merge module, or None where its replica does not apply."""
    # The kernel replicates the SM90 fast-math reference explicitly; other
    # arches (and ROCm) run a different reference build, so they fail closed
    # to Triton, whose replica follows the reference build per-arch.
    if torch.version.hip is not None or not reference_fast_math():
        return None
    try:
        return _jit_merge_heads_module(chunks_per_thread)
    except Exception as exc:
        logger.warning("Failed to load JIT head-merge+quant kernel: %s", exc)
        return None


def _cpp_merge_heads_quant_fp8(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Run the C++ head-merge kernel, or None when this input needs Triton.

    Every invariant the C++ launcher CHECKs is pre-cleared here, so an
    exception past this gate is a real bug and intentionally propagates
    instead of degrading to Triton.
    """
    if torch.compiler.is_compiling():
        return None
    world, seq, batch, h_local, head_dim = x.shape
    inner = h_local * head_dim
    if inner % _VEC_ELEMS or x.data_ptr() % _ALIGN_BYTES:
        return None
    chunks_per_thread = -(-(world * inner) // (_VEC_ELEMS * _BLOCK_SIZE))
    if chunks_per_thread > _MAX_CHUNKS:
        return None
    module = _merge_heads_module_or_none(chunks_per_thread)
    if module is None:
        return None
    tokens = batch * seq
    q = torch.empty((tokens, world * inner), dtype=torch.float8_e4m3fn, device=x.device)
    s = torch.empty((tokens, 1), dtype=torch.float32, device=x.device)
    module.usp_merge_heads_per_token_quant_fp8(
        q, s, x, reference_zero_guard(tokens, x.device)
    )
    return q, s


def usp_merge_heads_per_token_quant_fp8(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``usp_merge_heads`` + per-token FP8 quant in one pass.

    ``x`` is ``[W, S, B, h_local, D]`` contiguous; returns ``(q, s)`` with
    ``q`` fp8_e4m3 of shape ``[B*S, W*h_local*D]`` (row ``b*S + s``) and ``s``
    fp32 of shape ``[B*S, 1]``, bitwise equal to quantizing the merged bf16
    rows.
    """
    if not can_use_usp_merge_heads_per_token_quant_fp8(x):
        raise RuntimeError("unsupported input for fused USP head merge + fp8 quant")
    fused = _cpp_merge_heads_quant_fp8(x)
    if fused is not None:
        return fused
    return _triton_merge_heads(x)


# --- two-source (IPC output) form


@cache_once
def _jit_merge_two_sources_module(chunks_per_thread: int) -> Module:
    args = make_cpp_args(_BLOCK_SIZE, chunks_per_thread, is_arch_support_pdl())
    return load_jit(
        "diffusion_usp_merge_two_quant_fp8",
        *args,
        cuda_files=["diffusion/usp_merge_quant_fp8.cuh"],
        cuda_wrappers=[
            (
                "merge_two_sources_per_token_quant_fp8",
                "usp_merge_quant_fp8::" f"MergeTwoSourcesQuantKernel<{args}>::run",
            )
        ],
    )


@cache_once
def _merge_two_module_or_none(chunks_per_thread: int) -> Module | None:
    """The compiled two-source module, or None where its replica does not apply."""
    if torch.version.hip is not None or not reference_fast_math():
        return None
    try:
        return _jit_merge_two_sources_module(chunks_per_thread)
    except Exception as exc:
        logger.warning("Failed to load JIT two-source merge+quant kernel: %s", exc)
        return None


def _cpp_merge_two_sources_quant_fp8(
    first: torch.Tensor, second: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Run the C++ two-source kernel, or None when this input needs Triton.

    Every invariant the C++ launcher CHECKs is pre-cleared here, so an
    exception past this gate is a real bug and intentionally propagates
    instead of degrading to Triton.
    """
    if torch.compiler.is_compiling():
        return None
    tokens, inner = first.shape
    if inner % _VEC_ELEMS:
        return None
    chunks_per_thread = -(-(2 * inner) // (_VEC_ELEMS * _BLOCK_SIZE))
    if chunks_per_thread > _MAX_CHUNKS:
        return None
    if (
        first.stride(0) % _VEC_ELEMS
        or second.stride(0) % _VEC_ELEMS
        or first.data_ptr() % _ALIGN_BYTES
        or second.data_ptr() % _ALIGN_BYTES
    ):
        return None
    module = _merge_two_module_or_none(chunks_per_thread)
    if module is None:
        return None
    q = torch.empty((tokens, 2 * inner), dtype=torch.float8_e4m3fn, device=first.device)
    s = torch.empty((tokens, 1), dtype=torch.float32, device=first.device)
    module.merge_two_sources_per_token_quant_fp8(
        q, s, first, second, reference_zero_guard(tokens, first.device)
    )
    return q, s


def merge_two_sources_per_token_quant_fp8(
    first: torch.Tensor, second: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate two ``[T, C]`` head-shard sources column-wise and quantize.

    ``first`` fills output columns ``[0, C)`` and ``second`` ``[C, 2C)``;
    returns ``(q, s)`` bitwise equal to quantizing
    ``torch.cat((first, second), dim=1)``.
    """
    if not can_use_merge_two_sources_per_token_quant_fp8(first, second):
        raise RuntimeError("unsupported inputs for fused two-source merge + fp8 quant")
    fused = _cpp_merge_two_sources_quant_fp8(first, second)
    if fused is not None:
        return fused
    return _triton_merge_two_sources(first, second)


__all__ = [
    "can_use_merge_two_sources_per_token_quant_fp8",
    "can_use_usp_merge_heads_per_token_quant_fp8",
    "merge_two_sources_per_token_quant_fp8",
    "usp_merge_heads_per_token_quant_fp8",
]
