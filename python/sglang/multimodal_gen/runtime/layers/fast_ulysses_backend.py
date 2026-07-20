"""Experimental fast_ulysses (NVSHMEM + NVLink P2P) backend for Ulysses SP a2a.

Enabled via SGLANG_DIFFUSION_ENABLE_FAST_ULYSSES (plus
SGLANG_DIFFUSION_ENABLE_FAST_ULYSSES_QK_FUSION for the fused QK norm+RoPE
input a2a, SGLANG_DIFFUSION_ENABLE_FAST_ULYSSES_ASYNC_V_A2A for the async v
input a2a that overlaps the q/k projections, and
SGLANG_DIFFUSION_ENABLE_FAST_ULYSSES_ASYNC_QK_A2A for async q/k input a2a on
the unfused path). Every entry point returns None / False when the fast path
cannot serve the call, and callers must then run the NCCL path.

Collective safety: every decision here (env flags, group topology, tensor
shape/dtype) is identical across SP ranks under SPMD execution, so for a given
call site either all ranks take the fast path or none does.
"""

import functools
import logging
from typing import TYPE_CHECKING

import msgspec
import torch
import torch.distributed as dist

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_sp_group,
    get_ulysses_parallel_world_size,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    is_in_breakable_cuda_graph,
)

if TYPE_CHECKING:
    from fast_ulysses import AsyncA2AHandle

logger = logging.getLogger(__name__)

# fast_ulysses supports single-node NVLink P2P, world sizes 1..8; below 2
# there is nothing to communicate.
_MIN_WORLD_SIZE = 2
_MAX_WORLD_SIZE = 8

# Calls actually served by the fast path; tests assert on these to catch
# silent fallbacks.
fast_call_counts: dict[str, int] = {
    "a2a": 0,
    "qk2": 0,
    "a2a_async": 0,
    "a2a_async_qk": 0,
}


class QKFusedCtx(msgspec.Struct):
    """Inputs for the fused cross-head QK RMSNorm + RoPE + input a2a."""

    weight_q: torch.Tensor  # fp32 [num_heads * head_dim]
    weight_k: torch.Tensor  # fp32 [num_heads * head_dim]
    cos: torch.Tensor  # fp32 [s_local, head_dim // 2]
    sin: torch.Tensor  # fp32 [s_local, head_dim // 2]
    eps: float


@functools.cache
def _a2a_enabled() -> bool:
    return bool(envs.SGLANG_DIFFUSION_ENABLE_FAST_ULYSSES)


@functools.cache
def _qk_fusion_enabled() -> bool:
    return _a2a_enabled() and bool(envs.SGLANG_DIFFUSION_ENABLE_FAST_ULYSSES_QK_FUSION)


@functools.cache
def _async_v_enabled() -> bool:
    return _a2a_enabled() and bool(
        envs.SGLANG_DIFFUSION_ENABLE_FAST_ULYSSES_ASYNC_V_A2A
    )


@functools.cache
def _async_qk_enabled() -> bool:
    return _a2a_enabled() and bool(
        envs.SGLANG_DIFFUSION_ENABLE_FAST_ULYSSES_ASYNC_QK_A2A
    )


@functools.cache
def _get_group():
    """Build the process-wide UlyssesGroup once; None means permanent fallback.

    Lazy construction is collective-safe: under SPMD every SP rank reaches the
    first fast-path call in lockstep, and UlyssesGroup.__init__ runs its own
    broadcast + barriers on the bootstrap group.
    """
    if not _a2a_enabled():
        return None
    try:
        from fast_ulysses import UlyssesGroup
    except ImportError:
        logger.warning(
            "SGLANG_DIFFUSION_ENABLE_FAST_ULYSSES is set but fast_ulysses is "
            "not importable; keeping the NCCL all-to-all path."
        )
        return None
    ulysses_pg = get_sp_group().ulysses_group
    ulysses_world = dist.get_world_size(group=ulysses_pg)
    # Single-node NVLink P2P only: a multi-node pure-Ulysses layout passes the
    # world-size check below but fails loudly at NVSHMEM init.
    if ulysses_world != dist.get_world_size():
        # cfg-parallel / tp / ring layouts would need one NVSHMEM domain per
        # sub-group, which is unvalidated; keep NCCL there.
        logger.warning(
            "fast_ulysses disabled: ulysses group has %d ranks but world has "
            "%d; only pure-Ulysses layouts are supported.",
            ulysses_world,
            dist.get_world_size(),
        )
        return None
    if not (_MIN_WORLD_SIZE <= ulysses_world <= _MAX_WORLD_SIZE):
        logger.warning(
            "fast_ulysses disabled: ulysses world size %d outside [%d, %d].",
            ulysses_world,
            _MIN_WORLD_SIZE,
            _MAX_WORLD_SIZE,
        )
        return None
    group = UlyssesGroup(process_group=ulysses_pg)
    logger.info("fast_ulysses UlyssesGroup initialized (world=%d).", ulysses_world)
    return group


def _shape_ok(x: torch.Tensor, *, split_dim: int, world_size: int) -> bool:
    return (
        x.ndim == 4
        and x.is_cuda
        and x.dtype in (torch.bfloat16, torch.float16)
        and x.shape[split_dim] % world_size == 0
        and (x.shape[3] * x.element_size()) % 16 == 0
    )


def maybe_all_to_all_4d(x: torch.Tensor, *, mode: int, tag: str) -> torch.Tensor | None:
    """Fast-path replacement for the uniform Ulysses a2a on [b, s, h, d].

    mode 0: (b, s_local, h, d) -> (b, s_global, h_local, d)   into attention
    mode 1: (b, s_global, h_local, d) -> (b, s_local, h, d)   out of attention

    Returns None whenever the fast path cannot serve the call. ``tag`` must be
    unique among results alive at the same time (symmetric-heap output buffers
    are reused per tag); an empty tag disables the fast path.
    """
    if not tag or not _a2a_enabled():
        return None
    group = _get_group()
    if group is None:
        return None
    split_dim = 2 if mode == 0 else 1
    if not _shape_ok(
        x, split_dim=split_dim, world_size=get_ulysses_parallel_world_size()
    ):
        return None
    out = group.all_to_all_single_4d(x, mode=mode, tag=tag)
    fast_call_counts["a2a"] += 1
    return out


def maybe_async_input_a2a_v(v: torch.Tensor) -> "AsyncA2AHandle | None":
    """Async mode0 a2a for v on the group's comm stream; overlaps the caller's
    subsequent q/k projections.

    The caller MUST wait() the handle before issuing the next fast_ulysses
    collective on the main stream: the per-group barrier epoch is a single
    monotonic counter, so barrier kernels must execute in submission order
    (see fast_ulysses comm.py). Tag "usp_v" matches the sync path, so the same
    symmetric buffer is reused; cross-layer reuse stays safe because the
    handle's ready-event trails the previous layer's output a2a, whose flag
    barrier orders every peer's reads of this buffer before the new writes.

    Returns None whenever the fast path cannot serve the call.
    """
    if not _async_v_enabled():
        return None
    group = _get_group()
    if group is None:
        return None
    if not _shape_ok(v, split_dim=2, world_size=get_ulysses_parallel_world_size()):
        return None
    if is_in_breakable_cuda_graph():
        # The async launch uses Tensor.record_stream + cross-stream events,
        # both illegal under CUDA graph capture; the BCG scope flag flips in
        # SPMD lockstep, so this stays rank-uniform.
        return None
    if fast_call_counts["a2a_async"] == 0:
        logger.info("fast_ulysses async v input a2a path active.")
    handle = group.all_to_all_single_4d_async(v, mode=0, tag="usp_v")
    fast_call_counts["a2a_async"] += 1
    return handle


def maybe_async_input_a2a_qk(
    q: torch.Tensor, k: torch.Tensor
) -> "tuple[AsyncA2AHandle, AsyncA2AHandle] | None":
    """Async mode0 a2a for q and k: both collectives are in flight on the
    group's comm stream at once and the caller waits the two handles together.

    Unfused path only -- the fused qk2 collective has no async variant. Same
    ordering contract as maybe_async_input_a2a_v: the caller MUST wait BOTH
    handles before issuing the next sync fast_ulysses collective (e.g. a sync
    v a2a must run after the waits, not between issue and wait).

    Returns None whenever the fast path cannot serve the call.
    """
    if not _async_qk_enabled():
        return None
    group = _get_group()
    if group is None:
        return None
    world_size = get_ulysses_parallel_world_size()
    if not (
        _shape_ok(q, split_dim=2, world_size=world_size)
        and _shape_ok(k, split_dim=2, world_size=world_size)
    ):
        return None
    if is_in_breakable_cuda_graph():
        # Same capture hazard as maybe_async_input_a2a_v.
        return None
    if fast_call_counts["a2a_async_qk"] == 0:
        logger.info("fast_ulysses async q/k input a2a path active.")
    hq = group.all_to_all_single_4d_async(q, mode=0, tag="usp_q")
    hk = group.all_to_all_single_4d_async(k, mode=0, tag="usp_k")
    fast_call_counts["a2a_async_qk"] += 2
    return hq, hk


def can_use_qk2(*, num_heads: int, head_dim: int, dtype: torch.dtype) -> bool:
    """Rank-invariant pre-check so the model can skip its own norm/rope."""
    if not _qk_fusion_enabled():
        return False
    if _get_group() is None:
        return False
    world_size = get_ulysses_parallel_world_size()
    return (
        dtype in (torch.bfloat16, torch.float16)
        and num_heads % world_size == 0
        and head_dim >= 2
        and (head_dim & (head_dim - 1)) == 0
        and head_dim * dtype.itemsize >= 32
    )


def qk2_input_a2a(
    q: torch.Tensor, k: torch.Tensor, *, ctx: QKFusedCtx
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused cross-head RMSNorm + RoPE + mode0 a2a for q and k in one
    collective. Caller must have checked can_use_qk2()."""
    group = _get_group()
    assert group is not None, "qk2_input_a2a requires can_use_qk2() == True"
    if fast_call_counts["qk2"] == 0:
        logger.info("fast_ulysses fused QK norm+RoPE+a2a path active.")
    out_q, out_k = group.all_to_all_single_4d_qk2(
        q,
        k,
        ctx.weight_q,
        ctx.weight_k,
        ctx.cos,
        ctx.sin,
        mode="cross_head",
        interleaved=True,
        eps=ctx.eps,
        tag="usp_qk",
    )
    fast_call_counts["qk2"] += 1
    return out_q, out_k
