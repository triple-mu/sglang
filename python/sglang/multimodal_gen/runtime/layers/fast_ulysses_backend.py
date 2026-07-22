"""Experimental fast_ulysses (NVSHMEM + NVLink P2P) backend for Ulysses SP a2a.

Env surface (see envs.py):
- SGLANG_DIFFUSION_ENABLE_FAST_ULYSSES: route the uniform Ulysses a2a through
  fast_ulysses.
- SGLANG_DIFFUSION_FAST_ULYSSES_PIPELINE_QKV: fully pipelined QKV input a2a
  (per-tensor norm+rope, one async a2a per tensor, v issued first; q's a2a
  overlaps k's projection).
- SGLANG_DIFFUSION_FAST_ULYSSES_TYPE: transfer path,
  none (library default routing) / base (SM scatter) / tma / ce (copy engine).

Every entry point returns None / False when the fast path cannot serve the
call, and callers must then run the NCCL path.

Collective safety: every decision here (env flags, group topology, tensor
shape/dtype) is identical across SP ranks under SPMD execution, so for a given
call site either all ranks take the fast path or none does.
"""

import functools
import logging
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_sp_group,
    get_ulysses_parallel_world_size,
)
from sglang.multimodal_gen.runtime.utils.nvtx_pytorch_hooks import maybe_nvtx_range
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
fast_call_counts: dict[str, int] = {"a2a": 0, "a2a_async": 0}


@functools.cache
def _a2a_enabled() -> bool:
    return bool(envs.SGLANG_DIFFUSION_ENABLE_FAST_ULYSSES)


@functools.cache
def _pipeline_qkv_enabled() -> bool:
    return _a2a_enabled() and bool(envs.SGLANG_DIFFUSION_FAST_ULYSSES_PIPELINE_QKV)


@functools.cache
def _a2a_type() -> str:
    """Transfer path for the non-fused a2a calls: none/base/tma/ce."""
    a2a_type = str(envs.SGLANG_DIFFUSION_FAST_ULYSSES_TYPE or "none")
    assert a2a_type in ("none", "base", "tma", "ce"), (
        "SGLANG_DIFFUSION_FAST_ULYSSES_TYPE must be none, base, tma or ce, "
        f"got {a2a_type!r}"
    )
    return a2a_type


# use_tma argument for the kernel-path calls per _a2a_type() ("ce" is routed
# to the dedicated CE entry points instead).
_USE_TMA_BY_TYPE = {"none": None, "base": False, "tma": True}


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

    The transfer path follows SGLANG_DIFFUSION_FAST_ULYSSES_TYPE. Returns None
    whenever the fast path cannot serve the call. ``tag`` must be unique among
    results alive at the same time (symmetric-heap output buffers are reused
    per tag); an empty tag disables the fast path.
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
    a2a_type = _a2a_type()
    with maybe_nvtx_range(f"fu::a2a[{tag}] mode={mode} type={a2a_type}"):
        if a2a_type == "ce":
            out = group.all_to_all_single_4d_ce(x, mode=mode, tag=tag)
        else:
            out = group.all_to_all_single_4d(
                x, mode=mode, tag=tag, use_tma=_USE_TMA_BY_TYPE[a2a_type]
            )
    fast_call_counts["a2a"] += 1
    return out


def can_use_pipelined_qkv(*, num_heads: int, head_dim: int, dtype: torch.dtype) -> bool:
    """Rank-invariant pre-check for the pipelined QKV path: the plain-a2a
    shape/dtype envelope plus a usable group and no CUDA graph capture (the
    async launch uses Tensor.record_stream + cross-stream events, both illegal
    under capture; the BCG scope flag flips in SPMD lockstep, so this stays
    rank-uniform)."""
    if not _pipeline_qkv_enabled():
        return False
    if _get_group() is None:
        return False
    if is_in_breakable_cuda_graph():
        return False
    world_size = get_ulysses_parallel_world_size()
    return (
        dtype in (torch.bfloat16, torch.float16)
        and num_heads % world_size == 0
        and (head_dim * dtype.itemsize) % 16 == 0
    )


def pipelined_input_a2a(
    x: torch.Tensor, *, tag: str, barrier: bool
) -> "AsyncA2AHandle":
    """Async mode0 a2a on the transfer path selected by
    SGLANG_DIFFUSION_FAST_ULYSSES_TYPE. Caller must have checked
    can_use_pipelined_qkv(); production issue order is v, then q, then k, with
    barrier=True only on the last call -- the whole group shares ONE
    completion handshake, and remote data is guaranteed only after the
    barrier-carrying handle's wait() (all three are waited before attention).

    The caller MUST wait() every handle before issuing the next sync
    fast_ulysses collective on the main stream: the per-group barrier epoch is
    a single monotonic counter, so barrier kernels must execute in submission
    order (see fast_ulysses comm.py). Tags match the sync path, so the same
    symmetric buffers are reused; cross-layer reuse stays safe because the
    handle's ready-event trails the previous layer's output a2a, whose flag
    barrier orders every peer's reads of these buffers before the new writes.
    """
    group = _get_group()
    assert group is not None, "pipelined_input_a2a requires can_use_pipelined_qkv()"
    a2a_type = _a2a_type()
    if fast_call_counts["a2a_async"] == 0:
        logger.info("fast_ulysses pipelined qkv input a2a path active (%s).", a2a_type)
    with maybe_nvtx_range(f"fu::a2a_async[{tag}] barrier={barrier} type={a2a_type}"):
        if a2a_type == "ce":
            handle = group.all_to_all_single_4d_ce_async(
                x, mode=0, tag=tag, barrier=barrier
            )
        else:
            handle = group.all_to_all_single_4d_async(
                x, mode=0, tag=tag, use_tma=_USE_TMA_BY_TYPE[a2a_type], barrier=barrier
            )
    fast_call_counts["a2a_async"] += 1
    return handle
