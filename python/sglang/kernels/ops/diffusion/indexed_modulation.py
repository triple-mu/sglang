# SPDX-License-Identifier: Apache-2.0
"""Indexed AdaLN modulation for the MiniMax-H3 DiT.

The CUDA JIT kernels are bit-exact with the Triton ones in
``sglang.kernels.ops.diffusion.triton.indexed_modulation`` (which stays as the
A/B reference); they only replace the ``next_power_of_2(hidden)`` block shape
with fixed 128-bit vector accesses.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.kernels.ops.diffusion.triton import indexed_modulation as triton_impl
from sglang.srt.environ import envs

if TYPE_CHECKING:
    from tvm_ffi.module import Module


# Rows handled per CTA; each row is one more independent 128-bit load in flight.
# 2 measures fastest on H200 at the production shape (1 starves the memory
# pipe, 4 spills enough registers to lose occupancy).
_ROWS_PER_BLOCK = 2


@cache_once
def _jit_indexed_modulation_module() -> Module:
    args = make_cpp_args(_ROWS_PER_BLOCK)
    return load_jit(
        "diffusion_indexed_modulation",
        *args,
        cuda_files=["diffusion/indexed_modulation.cuh"],
        cuda_wrappers=[
            (
                "indexed_scale_shift",
                f"sglang::indexed_modulation::IndexedScaleShiftKernel<{args}>::run",
            ),
            (
                "indexed_gate",
                f"sglang::indexed_modulation::IndexedGateKernel<{args}>::run",
            ),
        ],
    )


@cache_once
def _use_jit() -> bool:
    return envs.SGLANG_OPT_USE_JIT_INDEXED_MODULATION.get()


def indexed_scale_shift_bf16_(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """In-place ``x = round_bf16(x * round_bf16(1 + scale[idx])) + shift[idx]``."""
    if not _use_jit():
        return triton_impl.indexed_scale_shift_bf16_(x, shift, scale, indices)
    _jit_indexed_modulation_module().indexed_scale_shift(x, shift, scale, indices)
    return x


def indexed_gate_bf16_(
    x: torch.Tensor,
    gate: torch.Tensor,
    other: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """In-place ``x = x + round_bf16(gate[idx] * other)``."""
    if not _use_jit():
        return triton_impl.indexed_gate_bf16_(x, gate, other, indices)
    _jit_indexed_modulation_module().indexed_gate(x, gate, other, indices)
    return x
