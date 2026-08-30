# SPDX-License-Identifier: Apache-2.0
"""Triton replica of the reference per-token FP8 quantization arithmetic.

Producer-fused quant kernels (SwiGLU + quant, Ulysses head-merge + quant) are
lossless only *relative to the separated chain*: producer output bytes fed to
``sgl_per_token_quant_fp8`` (``kernels/jit/csrc/gemm/per_token_quant_fp8.cuh``)
must yield the identical fp8 payload and fp32 scale.  That reference is built
with ``--use_fast_math`` on SM90 and precise math elsewhere (see
``ops/quantization/per_token_quant_fp8.py``), so the replica carries a
``FAST_MATH`` constexpr mirroring the same predicate.

SM90 fast-math build (verified against SASS on H200, sm_90a):

- amax:      ``FMNMX.FTZ`` over ``|fp32(x)|`` -- denormal inputs count as zero
- scale:     ``FMUL.FTZ(amax, 0x3B124925)`` -- compile-time rn(1/448), output
             denormals flushed to sign-preserving zero
- scale_inv: ``MUFU.RCP(scale)`` (``rcp.approx.f32``, not a correctly rounded
             division); the warp-dispatch kernel guards ``scale == 0`` to 0,
             the CTA-dispatch kernel does not (0 -> +inf)
- payload:   ``FMUL.FTZ(fp32(x), scale_inv)`` then ``FMNMX.FTZ`` clamp to
             [-448, 448] (IEEE minNum/maxNum NaN semantics) then
             ``F2FP.SATFINITE.E4M3``

Precise build (non-SM90): same dataflow with ``div.rn.f32`` for the scale and
its reciprocal and no denormal flushing.

The zero-scale guard differs between the reference's two dispatch variants, so
callers must mirror the launcher's dispatch predicate via
:func:`reference_zero_guard`.
"""

from __future__ import annotations

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore

from sglang.kernels.jit.utils import get_jit_cuda_arch
from sglang.kernels.ops.diffusion.common.numerics import div_rn_f32

FP8_E4M3_MAX = 448.0
# rn(1/448) = 0x3B124925, the compile-time constant the fast-math reference
# multiplies by instead of dividing by 448.
RECIP_FP8_E4M3_MAX = 1.0 / 448.0

_SM_COUNT_CACHE: dict[int, int] = {}


def reference_fast_math() -> bool:
    """Whether the reference JIT quant kernel is a ``--use_fast_math`` build.

    Must stay in lockstep with ``_jit_per_token_quant_fp8_module`` in
    ``ops/quantization/per_token_quant_fp8.py``.
    """
    arch = get_jit_cuda_arch()
    return (arch.major, arch.minor) == (9, 0)


def reference_zero_guard(num_tokens: int, device: torch.device) -> bool:
    """Whether the reference dispatch guards ``scale == 0`` for this row count.

    Mirrors ``launch_per_token_quant_fp8`` in
    ``kernels/jit/csrc/gemm/per_token_quant_fp8.cuh``: the warp-per-token
    kernel (chosen for ``num_tokens >= sm_count * 2 * 8``) zeroes the
    reciprocal of a zero scale; the CTA-per-token kernel divides by it.
    """
    index = device.index if device.index is not None else torch.cuda.current_device()
    sm_count = _SM_COUNT_CACHE.get(index)
    if sm_count is None:
        sm_count = torch.cuda.get_device_properties(index).multi_processor_count
        _SM_COUNT_CACHE[index] = sm_count
    return num_tokens >= sm_count * 2 * 8


@triton.jit
def ftz_f32(x):
    """Flush fp32 denormals to sign-preserving zero (FTZ input semantics)."""
    u = x.to(tl.uint32, bitcast=True)
    is_denormal = (u & 0x7F800000) == 0
    return tl.where(is_denormal, u & 0x80000000, u).to(tl.float32, bitcast=True)


@triton.jit
def rcp_approx_f32(x):
    """``rcp.approx.f32`` (MUFU.RCP), matching the fast-math reciprocal."""
    return tl.inline_asm_elementwise(
        asm="rcp.approx.f32 $0, $1;",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def amax_abs_f32(v, FAST_MATH: tl.constexpr):
    """One tile's contribution to the row amax: ``|v|`` with FTZ under fast math."""
    av = tl.abs(v)
    if FAST_MATH:
        av = ftz_f32(av)
    return av


@triton.jit
def scale_from_amax(amax, FAST_MATH: tl.constexpr):
    """Row scale ``amax / 448`` with the reference build's rounding."""
    if FAST_MATH:
        return ftz_f32(amax * 0.0022321429569274187)  # rn(1/448), FMUL.FTZ
    return div_rn_f32(amax, 448.0)


@triton.jit
def scale_inv_from_scale(scale, FAST_MATH: tl.constexpr, ZERO_GUARD: tl.constexpr):
    """Reference reciprocal of the scale; operates on a (1,)-shaped tensor."""
    if FAST_MATH:
        inv = rcp_approx_f32(scale)
    else:
        inv = div_rn_f32(1.0 + tl.zeros_like(scale), scale)
    if ZERO_GUARD:
        inv = tl.where(scale != 0.0, inv, 0.0)
    return inv


@triton.jit
def quantize_payload_f32(v, scale_inv, FAST_MATH: tl.constexpr):
    """Clamped fp32 payload, ready for an ``.to(tl.float8e4nv)`` store."""
    if FAST_MATH:
        v = ftz_f32(v)  # FMUL.FTZ flushes denormal inputs before multiplying
    val = v * scale_inv
    val = tl.minimum(val, 448.0)
    val = tl.maximum(val, -448.0)
    return val
