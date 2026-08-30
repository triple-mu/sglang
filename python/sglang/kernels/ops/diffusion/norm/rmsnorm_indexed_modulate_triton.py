# SPDX-License-Identifier: Apache-2.0
"""Fused RMSNorm + indexed adaLN scale/shift for the MiniMax-H3 DiT.

Plan A replaces the eager pair

    ``nn.RMSNorm(x)`` -> ``indexed_scale_shift_bf16_``          (2 kernels)

with one kernel computing ``out = (x / rms(x)) * w_eff[g] + b[g]`` per row,
where ``g`` is the row's modulation group (``combined_indices``) and
``w_eff[g] = norm_weight * (1 + scale[g])`` is merged once per denoise step in
fp32 (the RMSNorm gamma and the ``1 + scale`` factor commute multiplicatively).

Plan B additionally absorbs the preceding indexed gated-residual add

    ``y = residual + gate[g] * update``      (written back: the next residual)
    ``out = (y / rms(y)) * w_eff[g] + b[g]``

into the same kernel with two outputs.

Numerics: the gated-residual arithmetic replicates the eager
``indexed_gate_bf16_`` rounding chain exactly (``round_bf16(gate * update)``,
then a bf16 store of the add), so the residual stream stays bit-identical to
the eager path. The norm/modulate output is near-lossless rather than
bit-exact: the sum of squares uses the natural Triton tree reduction and the
affine epilogue rounds to bf16 once instead of three times. Measured at
[20992, 5376] on H200: <= 4 bf16 ulp of the pre-shift operand magnitude vs
the eager chain (p99 <= 2 ulp), and ~2.8x closer to the fp64 truth than the
eager chain itself; see ``test_minimax_h3_fused_adaln.py``.
"""

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore

from sglang.kernels.ops.diffusion.common.numerics import round_bf16_to_fp32


@triton.jit
def _rmsnorm_indexed_modulate_kernel(
    out_ptr,
    res_out_ptr,
    x_ptr,
    update_ptr,
    gate_ptr,
    weight_ptr,  # fp32 merged w_eff rows
    shift_ptr,
    indices_ptr,
    hidden_size,
    eps,
    stride_out_row,
    stride_res_out_row,
    stride_x_row,
    stride_update_row,
    stride_gate_row,
    stride_weight_row,
    stride_shift_row,
    stride_indices,
    BLOCK_N: tl.constexpr,
    HAS_GATE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    columns = tl.arange(0, BLOCK_N)
    mask = columns < hidden_size
    index = tl.load(indices_ptr + row * stride_indices)

    x = tl.load(x_ptr + row * stride_x_row + columns, mask=mask, other=0.0).to(
        tl.float32
    )
    if HAS_GATE:
        update = tl.load(
            update_ptr + row * stride_update_row + columns, mask=mask, other=0.0
        ).to(tl.float32)
        gate = tl.load(
            gate_ptr + index * stride_gate_row + columns, mask=mask, other=0.0
        ).to(tl.float32)
        # eager pair: bf16 round after gate*update and after the residual add
        x = round_bf16_to_fp32(x + round_bf16_to_fp32(gate * update))
        tl.store(res_out_ptr + row * stride_res_out_row + columns, x, mask=mask)

    rstd = tl.rsqrt(tl.sum(x * x, axis=0) / hidden_size + eps)
    weight = tl.load(
        weight_ptr + index * stride_weight_row + columns, mask=mask, other=0.0
    )
    shift = tl.load(
        shift_ptr + index * stride_shift_row + columns, mask=mask, other=0.0
    ).to(tl.float32)
    tl.store(
        out_ptr + row * stride_out_row + columns,
        x * rstd * weight + shift,  # store rounds to bf16 once
        mask=mask,
    )


def _validate_rows(x: torch.Tensor, weight_eff: torch.Tensor, shift: torch.Tensor):
    if x.dtype is not torch.bfloat16 or not x.is_cuda or x.stride(-1) != 1:
        raise ValueError("fused rmsnorm_indexed input must be CUDA bf16 rows")
    if weight_eff.dtype is not torch.float32 or weight_eff.stride(-1) != 1:
        raise ValueError("fused rmsnorm_indexed w_eff must be fp32 rows")
    if shift.dtype is not torch.bfloat16 or shift.stride(-1) != 1:
        raise ValueError("fused rmsnorm_indexed shift must be bf16 rows")


def rmsnorm_indexed_scale_shift(
    x: torch.Tensor,
    weight_eff: torch.Tensor,
    shift: torch.Tensor,
    indices: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    """Plan A: ``(x / rms(x)) * w_eff[indices] + shift[indices]``."""
    _validate_rows(x, weight_eff, shift)
    rows, hidden_size = x.shape
    out = torch.empty_like(x)
    if rows == 0:
        return out
    _rmsnorm_indexed_modulate_kernel[(rows,)](
        out,
        out,
        x,
        x,
        x,
        weight_eff,
        shift,
        indices,
        hidden_size,
        eps,
        out.stride(0),
        0,
        x.stride(0),
        0,
        0,
        weight_eff.stride(0),
        shift.stride(0),
        indices.stride(0),
        BLOCK_N=triton.next_power_of_2(hidden_size),
        HAS_GATE=False,
        num_warps=8,
    )
    return out


def gate_residual_rmsnorm_indexed_scale_shift_(
    residual: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    weight_eff: torch.Tensor,
    shift: torch.Tensor,
    indices: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Plan B: gated residual add fused with the following norm/modulate.

    ``residual`` is updated in place to ``y = residual + gate[idx] * update``
    (bit-exact vs ``indexed_gate_bf16_``); returns ``(out, residual)`` with
    ``out = (y / rms(y)) * w_eff[idx] + shift[idx]``.
    """
    _validate_rows(residual, weight_eff, shift)
    if update.dtype is not torch.bfloat16 or update.stride(-1) != 1:
        raise ValueError("fused rmsnorm_indexed update must be bf16 rows")
    if gate.dtype is not torch.bfloat16 or gate.stride(-1) != 1:
        raise ValueError("fused rmsnorm_indexed gate must be bf16 rows")
    rows, hidden_size = residual.shape
    out = torch.empty_like(residual)
    if rows == 0:
        return out, residual
    _rmsnorm_indexed_modulate_kernel[(rows,)](
        out,
        residual,
        residual,
        update,
        gate,
        weight_eff,
        shift,
        indices,
        hidden_size,
        eps,
        out.stride(0),
        residual.stride(0),
        residual.stride(0),
        update.stride(0),
        gate.stride(0),
        weight_eff.stride(0),
        shift.stride(0),
        indices.stride(0),
        BLOCK_N=triton.next_power_of_2(hidden_size),
        HAS_GATE=True,
        num_warps=8,
    )
    return out, residual
