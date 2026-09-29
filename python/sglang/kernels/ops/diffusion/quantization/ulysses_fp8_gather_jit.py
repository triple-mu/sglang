# SPDX-License-Identifier: Apache-2.0
"""Per-token fp8 rows for the Ulysses attention-output gather.

See `csrc/diffusion/ulysses_fp8_gather.cuh`: each source quantises its heads
of every token to e4m3 with one scale and ships `[codes | fp32 scale | pad]`
rows; the destination merges the rows of all sources into the `(q, s)` pair a
per-token-scaled fp8 GEMM consumes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_ROW_ALIGN = 16
_WARP_ELEMS = 128
_MAX_GROUP = 2048
_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)


def ulysses_fp8_gather_row_bytes(group: int) -> int:
    """Bytes per payload row: `group` codes, the fp32 scale, zero pad to 16 bytes."""
    return -(-(group + 4) // _ROW_ALIGN) * _ROW_ALIGN


@cache_once
def _jit_module(dtype: torch.dtype) -> Module:
    if dtype not in _SUPPORTED_DTYPES:
        raise RuntimeError(f"ulysses_fp8_gather: unsupported dtype {dtype}")
    args = make_cpp_args(dtype)
    return load_jit(
        "diffusion_ulysses_fp8_gather",
        *args,
        cuda_files=["diffusion/ulysses_fp8_gather.cuh"],
        cuda_wrappers=[
            ("quant", f"ulysses_fp8_gather::Kernels<{args}>::quant"),
            ("requant", f"ulysses_fp8_gather::Kernels<{args}>::requant"),
        ],
    )


def can_use_ulysses_fp8_gather(x: torch.Tensor) -> bool:
    """`[S, G]` bf16/fp16 CUDA rows with G a multiple of 128 up to 2048."""
    return (
        x.is_cuda
        and x.dtype in _SUPPORTED_DTYPES
        and x.dim() == 2
        and x.is_contiguous()
        and 0 < x.shape[1] <= _MAX_GROUP
        and x.shape[1] % _WARP_ELEMS == 0
    )


def ulysses_fp8_gather_quant(x: torch.Tensor, *, out: torch.Tensor) -> torch.Tensor:
    """`x [S, G]` -> per-token fp8 rows in `out [S, row_bytes(G)]` uint8."""
    _jit_module(x.dtype).quant(x, out)
    return out


def ulysses_fp8_gather_requant(
    payload: torch.Tensor, *, group: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """`[S_local, W, R]` rows from W sources -> `(q [S_local, W * G] e4m3, s [S_local, 1] fp32)`.

    Every source's codes are rescaled to the token's largest scale; the source
    holding that scale keeps its codes bit for bit.
    """
    s_local, world, _ = payload.shape
    q = torch.empty(
        (s_local, world * group), dtype=torch.float8_e4m3fn, device=payload.device
    )
    s = torch.empty((s_local, 1), dtype=torch.float32, device=payload.device)
    _jit_module(torch.bfloat16).requant(payload, q, s)
    return q, s


__all__ = [
    "can_use_ulysses_fp8_gather",
    "ulysses_fp8_gather_quant",
    "ulysses_fp8_gather_requant",
    "ulysses_fp8_gather_row_bytes",
]
