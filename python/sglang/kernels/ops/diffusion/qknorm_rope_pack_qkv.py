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
def _jit_qknorm_rope_pack_qkv_module(
    head_dim: int,
    rope_dim: int,
    is_neox: bool,
    dtype: torch.dtype,
    cache_dtype: torch.dtype,
) -> Module:
    args = make_cpp_args(
        head_dim,
        rope_dim,
        is_neox,
        is_arch_support_pdl(),
        dtype,
        cache_dtype,
    )
    return load_jit(
        "qknorm_rope_pack_qkv",
        *args,
        cuda_files=["diffusion/qknorm_rope_pack_qkv.cuh"],
        cuda_wrappers=[
            (
                "qknorm_rope_pack_qkv",
                f"sglang::qknorm_rope_pack_qkv::QKNormRopePackQKVKernel<{args}>::run",
            ),
        ],
    )


@torch.compiler.assume_constant_result
@cache_once
def can_use_fused_qknorm_rope_pack_qkv(
    head_dim: int,
    rope_dim: int,
    is_neox: bool,
    dtype: torch.dtype,
    cache_dtype: torch.dtype,
) -> bool:
    if head_dim not in (64, 128, 256):
        logger.warning(f"Unsupported head_dim={head_dim} for fused QKNorm+RoPE+pack")
        return False
    if rope_dim <= 0 or rope_dim > head_dim:
        logger.warning(
            f"Unsupported rope_dim={rope_dim} for head_dim={head_dim} in fused QKNorm+RoPE+pack"
        )
        return False
    elems_per_thread = head_dim // 32
    if rope_dim % elems_per_thread != 0:
        logger.warning(
            "rope_dim=%s must be divisible by per-thread width=%s for fused QKNorm+RoPE+pack",
            rope_dim,
            elems_per_thread,
        )
        return False
    if is_neox:
        rotary_lanes = rope_dim // elems_per_thread
        if rotary_lanes < 2 or rotary_lanes % 2:
            logger.warning(
                "rope_dim=%s yields invalid rotary_lanes=%s for neox fused QKNorm+RoPE+pack",
                rope_dim,
                rotary_lanes,
            )
            return False
    # The kernel only implements the rounded (bf16 boundary before RoPE)
    # contract, which requires the cache to live in the activation dtype.
    if cache_dtype != dtype:
        logger.warning(
            "Fused QKNorm+RoPE+pack requires cache dtype %s to match activation dtype %s",
            cache_dtype,
            dtype,
        )
        return False
    try:
        _jit_qknorm_rope_pack_qkv_module(
            head_dim,
            rope_dim,
            is_neox,
            dtype,
            cache_dtype,
        )
        return True
    except Exception as e:
        logger.warning(f"Failed to load JIT fused QKNorm+RoPE+pack kernel: {e}")
        return False


def _fake_qknorm_rope_pack_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    *,
    world_size: int,
    is_neox: bool,
    eps: float,
    head_dim: int = 0,
    rope_dim: int = 0,
) -> torch.Tensor:
    rows, global_heads, head_size = q.shape
    return q.new_empty((world_size, rows, global_heads // world_size, 3 * head_size))


@register_custom_op(
    op_name="diffusion_qknorm_rope_pack_qkv",
    mutates_args=[],
    fake_impl=_fake_qknorm_rope_pack_qkv,
)
def fused_qknorm_rope_pack_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    *,
    world_size: int,
    is_neox: bool,
    eps: float,
    head_dim: int = 0,
    rope_dim: int = 0,
) -> torch.Tensor:
    """Per-head QKNorm + RoPE on q/k fused into the destination-major QKV pack.

    Returns ``[world_size, rows, heads // world_size, 3 * head_dim]``, the
    Ulysses input all-to-all payload with ``q | k | v`` on the last axis. Q and K
    carry the rounded (bf16 boundary before RoPE) contract; V is copied as is.
    Bit-exact with ``fused_inplace_qknorm_rope`` followed by
    ``pack_qkv_destination_major``, in one pass over q/k/v.
    """
    rows, global_heads, head_size = q.shape
    packed = q.new_empty((world_size, rows, global_heads // world_size, 3 * head_size))
    module = _jit_qknorm_rope_pack_qkv_module(
        head_dim or head_size,
        rope_dim or cos_sin_cache.size(-1),
        is_neox,
        q.dtype,
        cos_sin_cache.dtype,
    )
    module.qknorm_rope_pack_qkv(
        packed, q, k, v, q_weight, k_weight, cos_sin_cache, positions, eps
    )
    return packed
