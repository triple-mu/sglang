"""Custom NVSHMEM Ulysses all-to-all wrapper for SGLang Diffusion.

Optional drop-in for the torch all_to_all in USPAttention's Ulysses path.
Gated by SGLANG_DIFFUSION_USE_CUSTOM_ULYSSES_A2A; falls back to torch when the
op is unavailable or the parallel config is unsupported.
"""

from __future__ import annotations

import logging

import torch

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_sp_group,
    get_ulysses_parallel_world_size,
)

logger = logging.getLogger(__name__)

# custom_ulysses_op supports single-node NVLink P2P for these world sizes only.
_SUPPORTED_WORLD_SIZES = (2, 4, 8)

# id(ulysses_pg) -> custom_ulysses_op.UlyssesGroup (collective; built lazily on
# first use while all ranks run the same attention forward in lockstep).
_groups: dict[int, object] = {}

_import_checked = False
_import_ok = False


def _custom_op_available() -> bool:
    global _import_checked, _import_ok
    if not _import_checked:
        _import_checked = True
        try:
            import custom_ulysses_op  # noqa: F401  (triggers TORCH_LIBRARY register)

            _import_ok = True
        except Exception as e:  # ImportError or .so dlopen failure
            _import_ok = False
            logger.warning(
                "custom_ulysses_op unavailable, falling back to torch a2a: %s", e
            )
    return _import_ok


def is_custom_ulysses_a2a_enabled() -> bool:
    """All-rank-consistent gate: env on + supported ulysses ws + op importable."""
    if not envs.SGLANG_DIFFUSION_USE_CUSTOM_ULYSSES_A2A:
        return False
    if get_ulysses_parallel_world_size() not in _SUPPORTED_WORLD_SIZES:
        return False
    return _custom_op_available()


def _get_group():
    ulysses_pg = get_sp_group().ulysses_group
    assert ulysses_pg is not None, "Ulysses process group is not initialized."
    key = id(ulysses_pg)
    group = _groups.get(key)
    if group is None:
        from custom_ulysses_op import UlyssesGroup

        group = UlyssesGroup(
            process_group=ulysses_pg,
            initial_pool_bytes=int(envs.SGLANG_DIFFUSION_CUSTOM_ULYSSES_POOL_BYTES),
        )
        _groups[key] = group
    return group


def custom_ulysses_a2a(x: torch.Tensor, *, mode: int, tag: str) -> torch.Tensor:
    """4D Ulysses all-to-all via the custom NVSHMEM op.

    mode 0: (b, s_local, n_global, d) -> (b, s_global, n_local, d)
    mode 1: inverse.
    Every rank MUST call with the same (shape, mode, use_tma) sequence, and tag
    must be unique per concurrently-live tensor (same tag reuses one heap buffer).
    """
    group = _get_group()
    use_tma = bool(envs.SGLANG_DIFFUSION_CUSTOM_ULYSSES_TMA)
    return group.all_to_all_single_4d(x, mode=mode, tag=tag, use_tma=use_tma)
