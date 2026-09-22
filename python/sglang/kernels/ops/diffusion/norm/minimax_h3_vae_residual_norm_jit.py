"""MiniMax-H3 VAE decoder row producers: fp32 residual update + RMSNorm /
LayerNorm with a dynamic per-token FP8 E4M3 epilogue (CUDA JIT).

Contract (close, not bit-exact): the residual update is one fp32 fma, the
statistics accumulate in fp32 over the fp32 row, both norms scale by
``rsqrt(var + eps)`` (LayerNorm with the two-pass centered variance), and the
FP8 epilogue matches ``per_token_quant_fp8`` (scale = amax / 448, codes clamped
to +-448, an all-zero row gives scale 0 and zero codes). Verified against
``F.rms_norm`` / ``F.layer_norm`` on ``[2, 7, 2048]`` rows with ``x`` in
fp32/fp16 and ``projected`` in fp16/fp32: >= 99.9% identical FP8 bins and a
dequantized error within half an FP8 ulp. The row width is ``x.shape[-1]``
(MiniMax-H3 uses 2048); every row tensor must be contiguous and 16-byte
aligned, the affine vectors are fp32. Deployment-gated by the MiniMax-H3 FP8
decoder option, not by the request quality tier.
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
_ROW_DTYPES = (torch.float32, torch.float16)
# kBlockSize * kVecSize in minimax_h3_vae_residual_norm.cuh.
_WIDTH_MULTIPLE = 1024
# The row lives in registers (width / 256 floats per thread); 2048 is the
# verified MiniMax-H3 width, 4096 keeps the per-thread budget at 16 floats.
_MAX_WIDTH = 4096

_LAUNCHERS = {
    "rmsnorm_fp8": "RmsNormFp8Kernel::run",
    "residual_rmsnorm_fp8": "ResidualRmsNormFp8Kernel::run",
    "residual_rmsnorm_fp8_bias": "ResidualRmsNormFp8Kernel::run_bias",
    "residual_layernorm": "ResidualLayerNormKernel::run",
    "residual_layernorm_bias": "ResidualLayerNormKernel::run_bias",
    "residual_rmsnorm_fp4": "ResidualRmsNormFp4Kernel::run",
}
# NVFP4: 16 elements per E4M3 block scale; the swizzled scale tensor pads rows to 128.
NVFP4_BLOCK = 16
NVFP4_SCALE_ROW_TILE = 128


@cache_once
def _jit_minimax_h3_vae_residual_norm_module(
    mode: str, dtype: torch.dtype, projected_dtype: torch.dtype | None, width: int
) -> Module:
    if mode not in _LAUNCHERS:
        raise RuntimeError(f"Unknown MiniMax-H3 residual norm mode {mode!r}")
    if dtype not in _ROW_DTYPES:
        raise RuntimeError(
            f"Unsupported row dtype {dtype}; expected float32 or float16"
        )
    if (projected_dtype is None) != (mode == "rmsnorm_fp8"):
        raise RuntimeError(f"mode {mode!r} takes a projected dtype iff residual")
    if projected_dtype is not None and projected_dtype not in _ROW_DTYPES:
        raise RuntimeError(
            f"Unsupported projected dtype {projected_dtype}; "
            "expected float32 or float16"
        )
    if width <= 0 or width % _WIDTH_MULTIPLE != 0 or width > _MAX_WIDTH:
        raise RuntimeError(
            f"Unsupported row width {width}; expected a multiple of "
            f"{_WIDTH_MULTIPLE} up to {_MAX_WIDTH}"
        )
    dtypes = (dtype,) if projected_dtype is None else (dtype, projected_dtype)
    args = make_cpp_args(*dtypes, width)
    struct, method = _LAUNCHERS[mode].split("::")
    launcher = f"minimax_h3_vae_residual_norm::{struct}<{args}>::{method}"
    return load_jit(
        "diffusion_minimax_h3_vae_residual_norm",
        mode,
        *args,
        cuda_files=["diffusion/minimax_h3_vae_residual_norm.cuh"],
        cuda_wrappers=[(mode, launcher)],
    )


@register_custom_op(mutates_args=["out_q", "out_s"])
def minimax_h3_vae_rmsnorm_fp8_raw(
    x: torch.Tensor,
    weight: torch.Tensor,
    out_q: torch.Tensor,
    out_s: torch.Tensor,
    eps: float,
) -> None:
    """``out_q, out_s = fp8(RMSNorm(x) * weight)`` on 2-D rows."""
    module = _jit_minimax_h3_vae_residual_norm_module(
        "rmsnorm_fp8", x.dtype, None, x.shape[-1]
    )
    module.rmsnorm_fp8(x, weight, out_q, out_s, eps)


@register_custom_op(mutates_args=["residual_out", "out_q", "out_s"])
def minimax_h3_vae_residual_rmsnorm_fp8_raw(
    x: torch.Tensor,
    projected: torch.Tensor,
    layer_scale: torch.Tensor,
    weight: torch.Tensor,
    residual_out: torch.Tensor,
    out_q: torch.Tensor,
    out_s: torch.Tensor,
    eps: float,
) -> None:
    """``residual_out = fma(projected, layer_scale, x)``, then RMSNorm + FP8."""
    module = _jit_minimax_h3_vae_residual_norm_module(
        "residual_rmsnorm_fp8", x.dtype, projected.dtype, x.shape[-1]
    )
    module.residual_rmsnorm_fp8(
        x, projected, layer_scale, weight, residual_out, out_q, out_s, eps
    )


@register_custom_op(mutates_args=["residual_out", "out_q", "out_s"])
def minimax_h3_vae_residual_rmsnorm_fp8_bias_raw(
    x: torch.Tensor,
    projected: torch.Tensor,
    projected_bias: torch.Tensor,
    layer_scale: torch.Tensor,
    weight: torch.Tensor,
    residual_out: torch.Tensor,
    out_q: torch.Tensor,
    out_s: torch.Tensor,
    eps: float,
) -> None:
    """Like ``minimax_h3_vae_residual_rmsnorm_fp8_raw`` with ``projected + projected_bias``."""
    module = _jit_minimax_h3_vae_residual_norm_module(
        "residual_rmsnorm_fp8_bias", x.dtype, projected.dtype, x.shape[-1]
    )
    module.residual_rmsnorm_fp8_bias(
        x, projected, projected_bias, layer_scale, weight, residual_out, out_q, out_s, eps
    )


@register_custom_op(mutates_args=["residual_out", "out_q4", "out_sf"])
def minimax_h3_vae_residual_rmsnorm_fp4_raw(
    x: torch.Tensor,
    projected: torch.Tensor,
    layer_scale: torch.Tensor,
    weight: torch.Tensor,
    residual_out: torch.Tensor,
    out_q4: torch.Tensor,
    out_sf: torch.Tensor,
    global_scale: float,
    eps: float,
) -> None:
    """``residual_out = fma(projected, layer_scale, x)``, then RMSNorm + NVFP4 (E2M1 codes,
    128x4-swizzled E4M3 block scales) for ``flashinfer.mm_fp4``."""
    module = _jit_minimax_h3_vae_residual_norm_module(
        "residual_rmsnorm_fp4", x.dtype, projected.dtype, x.shape[-1]
    )
    module.residual_rmsnorm_fp4(
        x, projected, layer_scale, weight, residual_out, out_q4, out_sf, global_scale, eps
    )


@register_custom_op(mutates_args=["out"])
def minimax_h3_vae_residual_layernorm_bias_raw(
    x: torch.Tensor,
    projected: torch.Tensor,
    projected_bias: torch.Tensor,
    layer_scale: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
    eps: float,
) -> None:
    """Like ``minimax_h3_vae_residual_layernorm_raw`` with ``projected + projected_bias``."""
    module = _jit_minimax_h3_vae_residual_norm_module(
        "residual_layernorm_bias", x.dtype, projected.dtype, x.shape[-1]
    )
    module.residual_layernorm_bias(
        x, projected, projected_bias, layer_scale, weight, bias, out, eps
    )


@register_custom_op(mutates_args=["out"])
def minimax_h3_vae_residual_layernorm_raw(
    x: torch.Tensor,
    projected: torch.Tensor,
    layer_scale: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
    eps: float,
) -> None:
    """``out = LayerNorm(fma(projected, layer_scale, x)) * weight + bias``."""
    module = _jit_minimax_h3_vae_residual_norm_module(
        "residual_layernorm", x.dtype, projected.dtype, x.shape[-1]
    )
    module.residual_layernorm(x, projected, layer_scale, weight, bias, out, eps)


def minimax_h3_vae_rmsnorm_fp8(
    x: torch.Tensor, weight: torch.Tensor, *, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(q, scales)``: fp8 e4m3 ``[M, width]`` and fp32 ``[M, 1]``."""
    rows = x.view(-1, x.shape[-1])
    out_q = torch.empty_like(rows, dtype=torch.float8_e4m3fn)
    out_s = torch.empty((rows.shape[0], 1), dtype=torch.float32, device=x.device)
    minimax_h3_vae_rmsnorm_fp8_raw(rows, weight, out_q, out_s, float(eps))
    return out_q, out_s


def minimax_h3_vae_residual_rmsnorm_fp8(
    x: torch.Tensor,
    projected: torch.Tensor,
    layer_scale: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float,
    projected_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(residual, q, scales)``; ``residual`` is fp32 with ``x.shape``.

    ``projected_bias`` (fp32 ``[width]``) is added to ``projected`` first; it
    carries the bias of an NVFP4 GEMM whose epilogue could not add it.
    """
    rows = x.view(-1, x.shape[-1])
    residual = torch.empty_like(rows, dtype=torch.float32)
    out_q = torch.empty_like(rows, dtype=torch.float8_e4m3fn)
    out_s = torch.empty((rows.shape[0], 1), dtype=torch.float32, device=x.device)
    if projected_bias is None:
        minimax_h3_vae_residual_rmsnorm_fp8_raw(
            rows, projected.view_as(rows), layer_scale, weight, residual, out_q, out_s, float(eps)
        )
    else:
        minimax_h3_vae_residual_rmsnorm_fp8_bias_raw(
            rows, projected.view_as(rows), projected_bias, layer_scale, weight,
            residual, out_q, out_s, float(eps),
        )
    return residual.view(x.shape), out_q, out_s


def nvfp4_scale_rows(rows: int) -> int:
    """Rows of the swizzled NVFP4 scale tensor for ``rows`` activation rows."""
    return (rows + NVFP4_SCALE_ROW_TILE - 1) // NVFP4_SCALE_ROW_TILE * NVFP4_SCALE_ROW_TILE


def minimax_h3_vae_residual_rmsnorm_fp4(
    x: torch.Tensor,
    projected: torch.Tensor,
    layer_scale: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float,
    global_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(residual, q4, sf)``: fp32 residual with ``x.shape``, packed E2M1
    codes ``[M, width / 2]`` (uint8) and 128x4-swizzled E4M3 block scales
    ``[nvfp4_scale_rows(M), width / 16]`` (uint8) for ``flashinfer.mm_fp4``.
    ``global_scale`` is the consuming linear's calibrated input scale
    (``448 * 6 / amax``)."""
    rows = x.view(-1, x.shape[-1])
    width = rows.shape[-1]
    residual = torch.empty_like(rows, dtype=torch.float32)
    out_q4 = torch.empty((rows.shape[0], width // 2), dtype=torch.uint8, device=x.device)
    out_sf = torch.empty(
        (nvfp4_scale_rows(rows.shape[0]), width // NVFP4_BLOCK), dtype=torch.uint8, device=x.device
    )
    minimax_h3_vae_residual_rmsnorm_fp4_raw(
        rows, projected.view_as(rows), layer_scale, weight, residual, out_q4, out_sf,
        float(global_scale), float(eps),
    )
    return residual.view(x.shape), out_q4, out_sf


def minimax_h3_vae_residual_layernorm(
    x: torch.Tensor,
    projected: torch.Tensor,
    layer_scale: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    eps: float,
    projected_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the fp32 LayerNorm of the updated residual, shaped like ``x``.

    ``projected_bias`` (fp32 ``[width]``) is added to ``projected`` first.
    """
    rows = x.view(-1, x.shape[-1])
    out = torch.empty_like(rows, dtype=torch.float32)
    if projected_bias is None:
        minimax_h3_vae_residual_layernorm_raw(
            rows, projected.view_as(rows), layer_scale, weight, bias, out, float(eps)
        )
    else:
        minimax_h3_vae_residual_layernorm_bias_raw(
            rows, projected.view_as(rows), projected_bias, layer_scale, weight, bias, out, float(eps)
        )
    return out.view(x.shape)


def _row_ok(x: torch.Tensor) -> bool:
    return (
        x.is_cuda
        and x.dim() >= 2
        and x.dtype in _ROW_DTYPES
        and 0 < x.shape[-1] <= _MAX_WIDTH
        and x.shape[-1] % _WIDTH_MULTIPLE == 0
        and x.numel() > 0
        and x.is_contiguous()
        and x.data_ptr() % _ALIGNMENT == 0
    )


def _projected_ok(projected: torch.Tensor, x: torch.Tensor) -> bool:
    return (
        projected.shape == x.shape
        and projected.dtype in _ROW_DTYPES
        and projected.device == x.device
        and projected.is_contiguous()
        and projected.data_ptr() % _ALIGNMENT == 0
    )


def _vector_ok(vector: torch.Tensor, x: torch.Tensor) -> bool:
    return (
        vector.shape == (x.shape[-1],)
        and vector.dtype == torch.float32
        and vector.device == x.device
        and vector.is_contiguous()
        and vector.data_ptr() % _ALIGNMENT == 0
    )


def can_use_minimax_h3_vae_rmsnorm_fp8(
    x: torch.Tensor, weight: torch.Tensor, *, eps: float
) -> bool:
    return (
        not torch.compiler.is_compiling()
        and _row_ok(x)
        and _vector_ok(weight, x)
        and eps > 0
        and is_cuda_sm_at_least(_MIN_SM, x.device)
    )


def can_use_minimax_h3_vae_residual_rmsnorm_fp8(
    x: torch.Tensor,
    projected: torch.Tensor,
    layer_scale: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float,
) -> bool:
    return (
        not torch.compiler.is_compiling()
        and _row_ok(x)
        and _projected_ok(projected, x)
        and _vector_ok(layer_scale, x)
        and _vector_ok(weight, x)
        and eps > 0
        and is_cuda_sm_at_least(_MIN_SM, x.device)
    )


def can_use_minimax_h3_vae_residual_rmsnorm_fp4(
    x: torch.Tensor,
    projected: torch.Tensor,
    layer_scale: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float,
    global_scale: float,
) -> bool:
    return (
        can_use_minimax_h3_vae_residual_rmsnorm_fp8(
            x, projected, layer_scale, weight, eps=eps
        )
        and x.shape[-1] % (4 * NVFP4_BLOCK) == 0
        and global_scale > 0
    )


def can_use_minimax_h3_vae_residual_layernorm(
    x: torch.Tensor,
    projected: torch.Tensor,
    layer_scale: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    eps: float,
) -> bool:
    return (
        not torch.compiler.is_compiling()
        and _row_ok(x)
        and _projected_ok(projected, x)
        and _vector_ok(layer_scale, x)
        and _vector_ok(weight, x)
        and _vector_ok(bias, x)
        and eps > 0
        and is_cuda_sm_at_least(_MIN_SM, x.device)
    )


__all__ = [
    "NVFP4_BLOCK",
    "can_use_minimax_h3_vae_residual_layernorm",
    "can_use_minimax_h3_vae_residual_rmsnorm_fp4",
    "can_use_minimax_h3_vae_residual_rmsnorm_fp8",
    "minimax_h3_vae_residual_layernorm_bias_raw",
    "minimax_h3_vae_residual_rmsnorm_fp4",
    "minimax_h3_vae_residual_rmsnorm_fp4_raw",
    "minimax_h3_vae_residual_rmsnorm_fp8_bias_raw",
    "nvfp4_scale_rows",
    "can_use_minimax_h3_vae_rmsnorm_fp8",
    "minimax_h3_vae_residual_layernorm",
    "minimax_h3_vae_residual_layernorm_raw",
    "minimax_h3_vae_residual_rmsnorm_fp8",
    "minimax_h3_vae_residual_rmsnorm_fp8_raw",
    "minimax_h3_vae_rmsnorm_fp8",
    "minimax_h3_vae_rmsnorm_fp8_raw",
]
