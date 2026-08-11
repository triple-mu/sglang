from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


logger = logging.getLogger(__name__)

_NAMESPACE = "sglang::indexed_modulation_norm"
# Mirrors kVecSize in diffusion/indexed_modulation_norm.cuh.
_VEC_SIZE = 4


@cache_once
def _jit_indexed_modulation_norm_module(hidden_size: int) -> Module:
    if hidden_size % _VEC_SIZE:
        raise RuntimeError(
            f"hidden_size={hidden_size} must be a multiple of {_VEC_SIZE}"
        )
    args = make_cpp_args(hidden_size)
    return load_jit(
        "diffusion_indexed_modulation_norm",
        *args,
        cuda_files=["diffusion/indexed_modulation_norm.cuh"],
        cuda_wrappers=[
            (
                "indexed_norm_scale_shift",
                f"{_NAMESPACE}::IndexedNormScaleShiftKernel<{args}>::run",
            ),
            (
                "indexed_gate_norm_scale_shift",
                f"{_NAMESPACE}::IndexedGateNormScaleShiftKernel<{args}>::run",
            ),
        ],
    )


@torch.compiler.assume_constant_result
@cache_once
def can_use_indexed_modulation_norm(hidden_size: int) -> bool:
    """Whether the fused kernel compiles for this row width."""
    try:
        _jit_indexed_modulation_norm_module(hidden_size)
        return True
    except Exception as e:
        logger.warning(f"Failed to load JIT fused indexed modulation+norm kernel: {e}")
        return False


def _norm_scale_shift_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    indices: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return torch.empty_like(x)


@register_custom_op(
    op_name="diffusion_indexed_norm_scale_shift",
    mutates_args=[],
    fake_impl=_norm_scale_shift_fake,
)
def indexed_norm_scale_shift(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    indices: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """RMSNorm(x) followed by the indexed affine `n * (1 + scale[i]) + shift[i]`.

    Bit-exact with `nn.RMSNorm(x)` -> `indexed_scale_shift_bf16_`. All tensors
    are BF16; `x` is [T, H] contiguous, `scale` / `shift` are [P, H] rows of an
    AdaLN projection, `indices` is an int64 [T] gather map.
    """
    out = torch.empty_like(x)
    module = _jit_indexed_modulation_norm_module(x.shape[-1])
    module.indexed_norm_scale_shift(out, x, weight, scale, shift, indices, eps)
    return out


def _gate_norm_scale_shift_fake(
    residual: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    indices: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return torch.empty_like(residual)


@register_custom_op(
    op_name="diffusion_indexed_gate_norm_scale_shift",
    mutates_args=["residual"],
    fake_impl=_gate_norm_scale_shift_fake,
)
def indexed_gate_norm_scale_shift(
    residual: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    indices: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """In-place gated residual, then RMSNorm, then the indexed affine.

    `residual` is updated in place to `residual + gate[i] * update`, matching
    the eager `indexed_gate_bf16_`; the return value is the normalized and
    modulated activation. Bit-exact with `indexed_gate_bf16_` -> `nn.RMSNorm`
    -> `indexed_scale_shift_bf16_`.
    """
    out = torch.empty_like(residual)
    module = _jit_indexed_modulation_norm_module(residual.shape[-1])
    module.indexed_gate_norm_scale_shift(
        out, residual, update, gate, weight, scale, shift, indices, eps
    )
    return out
