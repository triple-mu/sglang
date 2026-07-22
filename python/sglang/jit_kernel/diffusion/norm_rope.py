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
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


logger = logging.getLogger(__name__)


@cache_once
def _jit_norm_rope_module(
    head_dim: int,
    cross_head: bool,
    interleaved: bool,
    dtype: torch.dtype,
) -> Module:
    args = make_cpp_args(
        head_dim, cross_head, interleaved, is_arch_support_pdl(), dtype
    )
    return load_jit(
        "norm_rope",
        *args,
        cuda_files=["diffusion/norm_rope.cuh"],
        cuda_wrappers=[("norm_rope", f"NormRopeKernel<{args}>::run")],
    )


@torch.compiler.assume_constant_result
@cache_once
def can_use_fused_inplace_norm_rope(
    head_dim: int,
    num_heads: int,
    cross_head: bool,
    interleaved: bool,
    dtype: torch.dtype,
) -> bool:
    if dtype not in (torch.float16, torch.bfloat16):
        logger.warning(f"Unsupported dtype={dtype} for JIT fused Norm+RoPE")
        return False
    if head_dim < 2 or head_dim > 1024 or head_dim & (head_dim - 1):
        logger.warning(
            f"head_dim={head_dim} must be a power of two in [2, 1024] for fused Norm+RoPE"
        )
        return False
    if cross_head:
        # The cross-head kernel stages the token's num_heads*head_dim fp32 values in
        # dynamic shared memory; leave a margin for the kernel's static smem.
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        cap = getattr(props, "shared_memory_per_block_optin", 49152) - 1024
        if num_heads * head_dim * 4 > cap:
            logger.warning(
                "cross_head fused Norm+RoPE needs %s B shared memory > device cap %s B",
                num_heads * head_dim * 4,
                cap,
            )
            return False
    try:
        _jit_norm_rope_module(head_dim, cross_head, interleaved, dtype)
        return True
    except Exception as e:
        logger.warning(f"Failed to load JIT fused Norm+RoPE kernel: {e}")
        return False


@register_custom_op(mutates_args=["x"])
def fused_inplace_norm_rope(
    x: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    cross_head: bool,
    interleaved: bool = True,
    eps: float = 1e-6,
) -> None:
    """In-place fused RMSNorm + RoPE on ONE tensor (q or k), Wan semantics: fp32 math,
    eps inside rsqrt, per-channel fp32 weight, RoPE over the full head_dim.

    x: [b, seq, n, head_dim] contiguous fp16/bf16. weight: fp32, [head_dim] (per-head)
    or [n*head_dim] (cross_head: one RMS scalar per token). cos/sin: fp32 [seq, head_dim//2],
    pre-sliced to this rank's positions. interleaved=True is the GPT-J pair convention.
    """
    module = _jit_norm_rope_module(x.size(-1), cross_head, interleaved, x.dtype)
    module.norm_rope(x, weight, cos, sin, eps)
