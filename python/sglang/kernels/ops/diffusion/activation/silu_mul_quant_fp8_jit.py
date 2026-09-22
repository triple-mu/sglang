"""Gated-FFN activation producer: ``silu(gate) * up`` in fp32 with a dynamic
per-token FP8 E4M3 epilogue (CUDA JIT).

Contract (close, not bit-exact): both halves of the ``[gate | up]`` row are
widened to fp32, SiLU uses the precise ``expf`` like aten, the product is never
rounded to 16 bits, and the FP8 epilogue matches ``per_token_quant_fp8`` (scale
= amax / 448, codes clamped to +-448, an all-zero row gives scale 0 and zero
codes). Verified against ``F.silu(gate.float()) * up.float()`` for fp16 and
bf16 rows of width 2 * 8192 (MiniMax-H3's FFN) at 1 / 7 / 257 rows: >= 99.9%
identical FP8 bins and a dequantized error within half an FP8 ulp. The output
width is ``x.shape[-1] // 2``; ``x`` must be contiguous and 16-byte aligned.
Deployment-gated by the MiniMax-H3 FP8 decoder option, not by the request
quality tier.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.kernels.ops.diffusion.common.platform import is_cuda_sm_at_least
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_MIN_SM = (10, 0)
_ALIGNMENT = 16
_DTYPES = (torch.float16, torch.bfloat16)
# kBlockSize * kVecSize in silu_mul_quant_fp8.cuh for 16-bit inputs.
_WIDTH_MULTIPLE = 2048
# The fp32 products live in registers (width / 256 floats per thread); 8192 is
# the verified MiniMax-H3 FFN width and already 32 floats per thread.
_MAX_WIDTH = 8192


@cache_once
def _jit_silu_mul_quant_fp8_module(dtype: torch.dtype, width: int) -> Module:
    if dtype not in _DTYPES:
        raise RuntimeError(f"Unsupported dtype {dtype}; expected float16 or bfloat16")
    if width <= 0 or width % _WIDTH_MULTIPLE != 0 or width > _MAX_WIDTH:
        raise RuntimeError(
            f"Unsupported output width {width}; expected a multiple of "
            f"{_WIDTH_MULTIPLE} up to {_MAX_WIDTH}"
        )
    args = make_cpp_args(dtype, width)
    return load_jit(
        "diffusion_silu_mul_quant_fp8",
        *args,
        cuda_files=["diffusion/silu_mul_quant_fp8.cuh"],
        cuda_wrappers=[
            (
                "silu_mul_quant_fp8",
                f"silu_mul_quant_fp8::SiluMulQuantFp8Kernel<{args}>::run",
            )
        ],
    )


@cache_once
def _jit_silu_mul_quant_fp4_module(dtype: torch.dtype, width: int) -> Module:
    if dtype not in _DTYPES:
        raise RuntimeError(f"Unsupported dtype {dtype}; expected float16 or bfloat16")
    if width <= 0 or width % _WIDTH_MULTIPLE != 0 or width > _MAX_WIDTH:
        raise RuntimeError(
            f"Unsupported output width {width}; expected a multiple of "
            f"{_WIDTH_MULTIPLE} up to {_MAX_WIDTH}"
        )
    args = make_cpp_args(dtype, width)
    return load_jit(
        "diffusion_silu_mul_quant_fp4",
        *args,
        cuda_files=["diffusion/silu_mul_quant_fp8.cuh"],
        cuda_wrappers=[
            (
                "silu_mul_quant_fp4",
                f"silu_mul_quant_fp8::SiluMulQuantFp4Kernel<{args}>::run",
            )
        ],
    )


@register_custom_op(mutates_args=["out_q4", "out_sf"])
def silu_mul_quant_fp4_raw(
    x: torch.Tensor,
    gate_bias: torch.Tensor,
    up_bias: torch.Tensor,
    out_q4: torch.Tensor,
    out_sf: torch.Tensor,
    global_scale: float,
) -> None:
    """``out_q4, out_sf = nvfp4(silu(gate + gate_bias) * (up + up_bias))`` on ``[gate | up]`` rows."""
    module = _jit_silu_mul_quant_fp4_module(x.dtype, out_q4.shape[-1] * 2)
    module.silu_mul_quant_fp4(x, gate_bias, up_bias, out_q4, out_sf, global_scale)


def silu_mul_quant_fp4(
    x: torch.Tensor,
    *,
    global_scale: float,
    gate_bias: torch.Tensor,
    up_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(q4, sf)``: packed E2M1 codes ``[M, width / 2]`` (uint8) and
    128x4-swizzled E4M3 block scales ``[ceil(M / 128) * 128, width / 16]`` (uint8)
    for ``flashinfer.mm_fp4``; the up-projection biases are folded in."""
    width = x.shape[-1] // 2
    rows = x.view(-1, 2 * width)
    out_q4 = torch.empty((rows.shape[0], width // 2), dtype=torch.uint8, device=x.device)
    scale_rows = (rows.shape[0] + 127) // 128 * 128
    out_sf = torch.empty((scale_rows, width // 16), dtype=torch.uint8, device=x.device)
    silu_mul_quant_fp4_raw(rows, gate_bias, up_bias, out_q4, out_sf, float(global_scale))
    return out_q4, out_sf


def can_use_silu_mul_quant_fp4(
    x: torch.Tensor, gate_bias: torch.Tensor, up_bias: torch.Tensor, *, global_scale: float
) -> bool:
    width = x.shape[-1] // 2 if x.dim() >= 2 else 0
    return (
        can_use_silu_mul_quant_fp8(x)
        and width % 64 == 0
        and global_scale > 0
        and all(
            b.shape == (width,) and b.dtype == torch.float32 and b.device == x.device
            and b.is_contiguous() and b.data_ptr() % _ALIGNMENT == 0
            for b in (gate_bias, up_bias)
        )
    )


@register_custom_op(mutates_args=["out_q", "out_s"])
def silu_mul_quant_fp8_raw(
    x: torch.Tensor, out_q: torch.Tensor, out_s: torch.Tensor
) -> None:
    """``out_q, out_s = fp8(silu(gate) * up)`` on 2-D ``[gate | up]`` rows."""
    module = _jit_silu_mul_quant_fp8_module(x.dtype, out_q.shape[-1])
    module.silu_mul_quant_fp8(x, out_q, out_s)


def silu_mul_quant_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(q, scales)``: fp8 e4m3 ``[M, width]`` and fp32 ``[M, 1]``."""
    width = x.shape[-1] // 2
    rows = x.view(-1, 2 * width)
    out_q = torch.empty(
        (rows.shape[0], width), dtype=torch.float8_e4m3fn, device=x.device
    )
    out_s = torch.empty((rows.shape[0], 1), dtype=torch.float32, device=x.device)
    silu_mul_quant_fp8_raw(rows, out_q, out_s)
    return out_q, out_s


def can_use_silu_mul_quant_fp8(x: torch.Tensor) -> bool:
    return (
        not torch.compiler.is_compiling()
        and x.is_cuda
        and x.dim() >= 2
        and x.dtype in _DTYPES
        and 0 < x.shape[-1] <= 2 * _MAX_WIDTH
        and x.shape[-1] % (2 * _WIDTH_MULTIPLE) == 0
        and x.numel() > 0
        and x.is_contiguous()
        and x.data_ptr() % _ALIGNMENT == 0
        and is_cuda_sm_at_least(_MIN_SM, x.device)
    )


__all__ = [
    "can_use_silu_mul_quant_fp4",
    "can_use_silu_mul_quant_fp8",
    "silu_mul_quant_fp4",
    "silu_mul_quant_fp4_raw",
    "silu_mul_quant_fp8",
    "silu_mul_quant_fp8_raw",
]
