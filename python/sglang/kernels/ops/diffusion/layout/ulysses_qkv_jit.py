# SPDX-License-Identifier: Apache-2.0
"""Ulysses destination-major QKV pack: cpp-jit first, Triton fallback.

Pure data movement, so every backend is bitwise identical to the unpacked
aten copy chain; the dispatch here only changes which kernel moves the bytes.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    is_hip_runtime,
    load_jit,
    make_cpp_args,
)
from sglang.kernels.ops.diffusion.layout.ulysses_qkv_triton import (
    pack_qkv_destination_major as _pack_qkv_destination_major_triton,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

logger = logging.getLogger(__name__)

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_FAILED_RUNTIME_DEVICES: set[int | None] = set()


@cache_once
def _jit_ulysses_qkv_pack_module(dtype: torch.dtype) -> Module:
    if dtype not in _SUPPORTED_DTYPES:
        raise RuntimeError(f"Unsupported pack_qkv_destination_major dtype: {dtype}")
    args = make_cpp_args(dtype, is_arch_support_pdl())
    return load_jit(
        "diffusion_ulysses_qkv_pack",
        *args,
        cuda_files=["diffusion/ulysses_qkv_pack.cuh"],
        cuda_wrappers=[
            (
                "pack_qkv_destination_major",
                f"ulysses_qkv_pack::PackQkvDestinationMajorKernel<{args}>::run",
            ),
        ],
    )


@torch.compiler.assume_constant_result
@cache_once
def can_use_pack_qkv_destination_major_cuda(dtype: torch.dtype) -> bool:
    """Whether the cpp-jit pack is available; ROCm and pre-SM90 fail closed."""
    if is_hip_runtime() or not torch.cuda.is_available():
        return False
    if dtype not in _SUPPORTED_DTYPES:
        return False
    if torch.cuda.get_device_capability()[0] < 9:
        return False
    try:
        _jit_ulysses_qkv_pack_module(dtype)
        return True
    except Exception as exc:
        logger.warning("Failed to load JIT ulysses qkv pack kernel: %s", exc)
        return False


def pack_qkv_destination_major(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    world_size: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pack matching ``[rows, global_heads, head_size]`` Q/K/V tensors."""
    # Validation stays here (not per-backend) so both paths raise the same
    # ValueErrors; the messages are part of the tested contract.
    if q.dim() != 3 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, and v must have the same 3D shape")
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("q, k, and v must be CUDA tensors")
    if not (q.device == k.device == v.device and q.dtype == k.dtype == v.dtype):
        raise ValueError("q, k, and v must have the same device and dtype")
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise ValueError("q, k, and v must be contiguous in head_size")
    if world_size < 1 or q.shape[1] % world_size != 0:
        raise ValueError("world_size must be positive and divide global_heads")

    rows, global_heads, head_size = q.shape
    local_heads = global_heads // world_size
    expected_shape = (world_size, rows, local_heads, 3 * head_size)
    if out is not None:
        if not (
            out.shape == expected_shape
            and out.is_contiguous()
            and out.dtype == q.dtype
            and out.device == q.device
        ):
            raise ValueError(
                "out must be a contiguous tensor with the expected shape, "
                "device, and dtype"
            )
        output = out
    else:
        output = torch.empty(expected_shape, dtype=q.dtype, device=q.device)
    if rows * global_heads * head_size == 0:
        return output

    device_key = q.device.index
    if (
        device_key not in _FAILED_RUNTIME_DEVICES
        and not torch.compiler.is_compiling()
        and can_use_pack_qkv_destination_major_cuda(q.dtype)
    ):
        try:
            module = _jit_ulysses_qkv_pack_module(q.dtype)
            module.pack_qkv_destination_major(output, q, k, v)
            return output
        except Exception as exc:
            _FAILED_RUNTIME_DEVICES.add(device_key)
            logger.warning(
                "Disabling ulysses qkv pack CUDA fast path on %s: %s", q.device, exc
            )
    return _pack_qkv_destination_major_triton(q, k, v, world_size, out=output)


__all__ = [
    "can_use_pack_qkv_destination_major_cuda",
    "pack_qkv_destination_major",
]
