# SPDX-License-Identifier: Apache-2.0
"""On-disk persistence for MiniMax H3 online-rebuilt AdaLN plans.

Every plan-cache miss under ``--minimax-h3-adaln-online`` costs a streaming
read of all adaln_proj layers from the original checkpoint (~24 GiB). The
persist store keyed by (checkpoint fingerprint, tp size, exact fp32 plan key)
turns that into a per-plan slab read on later process starts. Loading is
all-or-nothing per rebuild pass and any mismatch falls back to the checkpoint
rebuild, so a stale or foreign store can only cost time, never correctness.
"""

from __future__ import annotations

import hashlib
import os
import struct
import tempfile
from collections.abc import Iterable

import msgspec
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

_PERSIST_FORMAT_VERSION = "1"
# Cheap content fingerprint: per shard, size plus a head/tail sample. A
# retrained checkpoint of identical structure differs essentially everywhere,
# so sampled bytes are sufficient without hashing the full 24 GiB.
_FINGERPRINT_SAMPLE_BYTES = 1 << 20


def adaln_plan_key(timesteps: torch.Tensor) -> tuple[int, ...]:
    """One denoise step's unique timesteps as their exact fp32 bit patterns."""
    return tuple(
        struct.unpack("<I", struct.pack("<f", float(value)))[0]
        for value in timesteps.tolist()
    )


class MiniMaxH3AdalnPersistSpec(msgspec.Struct, frozen=True, kw_only=True):
    """Resolved location and validation contract of one persist store."""

    directory: str
    checkpoint_fingerprint: str
    tp_size: int
    num_layers: int
    block_width: int
    final_width: int


def make_adaln_persist_spec(
    *,
    directory: str,
    weight_files: list[str],
    tp_size: int,
    num_layers: int,
    block_width: int,
    final_width: int,
) -> MiniMaxH3AdalnPersistSpec:
    fingerprint = _checkpoint_fingerprint(weight_files)
    # tp_size is part of the store identity: the rebuild GEMM is sharded per
    # rank and all-gathered, so plans built at a different tp are not
    # bit-identical to what this configuration would build.
    store = os.path.join(directory, f"{fingerprint[:16]}-tp{tp_size}")
    return MiniMaxH3AdalnPersistSpec(
        directory=store,
        checkpoint_fingerprint=fingerprint,
        tp_size=tp_size,
        num_layers=num_layers,
        block_width=block_width,
        final_width=final_width,
    )


def load_adaln_plans(
    spec: MiniMaxH3AdalnPersistSpec,
    plan_keys: list[tuple[int, ...]],
    *,
    device: torch.device,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None:
    """Load every requested plan or None; a partial hit saves nothing.

    Any later miss re-reads the full checkpoint regardless of how many plans
    it needs, so the caller only skips the rebuild when the whole set loads.
    Returns per key ``(plan_timesteps [L], block_params [L, layers, block_w],
    final_params [L, final_w])`` on ``device``.
    """
    loaded = []
    for key in plan_keys:
        entry = _load_one_plan(spec, key, device=device)
        if entry is None:
            return None
        loaded.append(entry)
    return loaded


def save_adaln_plans(
    spec: MiniMaxH3AdalnPersistSpec,
    plans: Iterable[tuple[tuple[int, ...], torch.Tensor, torch.Tensor, torch.Tensor]],
) -> int:
    """Best-effort persist of built plans; returns how many files were written.

    Existing files are kept as-is (same key implies the same bytes). Writes go
    through a temp file plus ``os.replace`` so concurrent readers only ever
    see complete files.
    """
    os.makedirs(spec.directory, exist_ok=True)
    saved = 0
    for key, plan_timesteps, block_params, final_params in plans:
        path = _plan_path(spec, key)
        if os.path.exists(path):
            continue
        try:
            _write_one_plan(
                spec,
                path,
                key=key,
                plan_timesteps=plan_timesteps,
                block_params=block_params,
                final_params=final_params,
            )
        except Exception:
            logger.warning(
                "MiniMax H3 AdaLN persist: writing %s failed; continuing",
                path,
                exc_info=True,
            )
            continue
        saved += 1
    return saved


def _checkpoint_fingerprint(weight_files: list[str]) -> str:
    digest = hashlib.sha256()
    digest.update(f"minimax-h3-adaln-persist-v{_PERSIST_FORMAT_VERSION}".encode())
    for path in sorted(weight_files):
        size = os.path.getsize(path)
        digest.update(f"\n{os.path.basename(path)}\t{size}\t".encode())
        with open(path, "rb") as f:
            digest.update(f.read(_FINGERPRINT_SAMPLE_BYTES))
            if size > 2 * _FINGERPRINT_SAMPLE_BYTES:
                f.seek(size - _FINGERPRINT_SAMPLE_BYTES)
                digest.update(f.read(_FINGERPRINT_SAMPLE_BYTES))
    return digest.hexdigest()


def _plan_path(spec: MiniMaxH3AdalnPersistSpec, key: tuple[int, ...]) -> str:
    plan_hash = hashlib.sha256(",".join(str(bits) for bits in key).encode()).hexdigest()
    return os.path.join(spec.directory, f"plan-{plan_hash[:24]}.safetensors")


def _expected_metadata(spec: MiniMaxH3AdalnPersistSpec, key: tuple[int, ...]):
    return {
        "format_version": _PERSIST_FORMAT_VERSION,
        "checkpoint_fingerprint": spec.checkpoint_fingerprint,
        "tp_size": str(spec.tp_size),
        "num_layers": str(spec.num_layers),
        "block_width": str(spec.block_width),
        "final_width": str(spec.final_width),
        "plan_key": ",".join(str(bits) for bits in key),
    }


def _load_one_plan(
    spec: MiniMaxH3AdalnPersistSpec,
    key: tuple[int, ...],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    path = _plan_path(spec, key)
    if not os.path.isfile(path):
        return None
    expected = _expected_metadata(spec, key)
    try:
        with safe_open(path, framework="pt", device=str(device)) as plan_file:
            metadata = plan_file.metadata() or {}
            if {k: metadata.get(k) for k in expected} != expected:
                logger.warning(
                    "MiniMax H3 AdaLN persist: %s metadata mismatch; rebuilding",
                    path,
                )
                return None
            plan_timesteps = plan_file.get_tensor("plan_timesteps")
            block_params = plan_file.get_tensor("block_params")
            final_params = plan_file.get_tensor("final_params")
    except Exception:
        logger.warning(
            "MiniMax H3 AdaLN persist: reading %s failed; rebuilding",
            path,
            exc_info=True,
        )
        return None
    length = len(key)
    if (
        plan_timesteps.dtype != torch.float32
        or plan_timesteps.shape != (length,)
        or adaln_plan_key(plan_timesteps) != key
        or block_params.dtype != torch.bfloat16
        or block_params.shape != (length, spec.num_layers, spec.block_width)
        or final_params.dtype != torch.bfloat16
        or final_params.shape != (length, spec.final_width)
    ):
        logger.warning(
            "MiniMax H3 AdaLN persist: %s tensor contract mismatch; rebuilding",
            path,
        )
        return None
    return plan_timesteps, block_params, final_params


def _write_one_plan(
    spec: MiniMaxH3AdalnPersistSpec,
    path: str,
    *,
    key: tuple[int, ...],
    plan_timesteps: torch.Tensor,
    block_params: torch.Tensor,
    final_params: torch.Tensor,
) -> None:
    fd, tmp_path = tempfile.mkstemp(
        dir=spec.directory, prefix=".tmp-", suffix=".safetensors"
    )
    os.close(fd)
    try:
        save_file(
            {
                "plan_timesteps": plan_timesteps.detach().cpu().contiguous(),
                "block_params": block_params.detach().cpu().contiguous(),
                "final_params": final_params.detach().cpu().contiguous(),
            },
            tmp_path,
            metadata=_expected_metadata(spec, key),
        )
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
