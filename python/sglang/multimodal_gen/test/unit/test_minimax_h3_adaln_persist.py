# SPDX-License-Identifier: Apache-2.0
"""Persist-store contract for the online AdaLN cache: a fresh process must
reload previously built plans bit-identically, and any store mismatch must
fall back to the checkpoint rebuild instead of failing the request."""

from pathlib import Path

import torch
from safetensors.torch import save_file

from sglang.multimodal_gen.configs.models.dits.minimax_h3 import (
    MINIMAX_H3_ADALN_MODALITY_NUM,
    MiniMaxH3DiTArchConfig,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import MiniMaxH3AdalnCache
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    ensure_distributed_env_defaults,
)

_ARCH = MiniMaxH3DiTArchConfig(
    num_layers=2,
    hidden_size=4,
    time_embed_dim=3,
)
_BLOCK_WIDTH = 6 * MINIMAX_H3_ADALN_MODALITY_NUM * _ARCH.hidden_size
_FINAL_WIDTH = 2 * _ARCH.hidden_size


def _ensure_single_process_parallel_runtime() -> None:
    if model_parallel_is_initialized():
        return
    ensure_distributed_env_defaults()
    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)


def _write_online_weights(path: Path, *, seed: int = 7) -> None:
    generator = torch.Generator().manual_seed(seed)
    tensors: dict[str, torch.Tensor] = {}
    for layer in range(_ARCH.num_layers):
        prefix = f"blocks.{layer}.adaln_proj.linear"
        tensors[f"{prefix}.weight"] = torch.randn(
            _BLOCK_WIDTH, _ARCH.time_embed_dim, generator=generator
        )
        tensors[f"{prefix}.bias"] = torch.randn(_BLOCK_WIDTH, generator=generator)
    prefix = "final_layer.adaln_proj.linear"
    tensors[f"{prefix}.weight"] = torch.randn(
        _FINAL_WIDTH, _ARCH.time_embed_dim, generator=generator
    )
    tensors[f"{prefix}.bias"] = torch.randn(_FINAL_WIDTH, generator=generator)
    save_file(tensors, path)


def _online_cache(weight_path: Path) -> MiniMaxH3AdalnCache:
    _ensure_single_process_parallel_runtime()
    cache = MiniMaxH3AdalnCache(
        _ARCH,
        weight_files=[str(weight_path)],
        max_plans=4,
        max_plan_width=2,
    )
    cache.load(torch.device("cpu"))
    return cache


def _embed(timesteps: torch.Tensor) -> torch.Tensor:
    scale = torch.arange(1, _ARCH.time_embed_dim + 1, dtype=torch.float32)
    return timesteps[:, None] * scale + 0.25


def _slab_rows(cache: MiniMaxH3AdalnCache, plan: torch.Tensor):
    slot = int(cache.lookup(plan))
    length = plan.numel()
    return (
        cache.plan_timesteps[slot, :length],
        cache.block_params[slot, :length],
        cache.final_params[slot, :length],
    )


def test_persist_round_trip_loads_bit_identical_plans(tmp_path, monkeypatch):
    monkeypatch.setenv("MINIMAX_H3_ADALN_PERSIST_DIR", str(tmp_path / "store"))
    weight_path = tmp_path / "model.safetensors"
    _write_online_weights(weight_path)
    plan_a = torch.tensor([0.125, 0.999])
    plan_b = torch.tensor([0.5])

    first = _online_cache(weight_path)
    first.build([plan_a, plan_b], embed=_embed)
    assert first.rebuilds == 1 and first.persist_loads == 0

    fresh = _online_cache(weight_path)
    fresh.build([plan_a, plan_b], embed=_embed)
    assert fresh.rebuilds == 0 and fresh.persist_loads == 1
    for plan in (plan_a, plan_b):
        for got, want in zip(_slab_rows(fresh, plan), _slab_rows(first, plan)):
            assert torch.equal(got, want)


def test_persist_partial_store_falls_back_to_rebuild_and_backfills(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("MINIMAX_H3_ADALN_PERSIST_DIR", str(tmp_path / "store"))
    weight_path = tmp_path / "model.safetensors"
    _write_online_weights(weight_path)
    plan_a = torch.tensor([0.125, 0.999])
    plan_b = torch.tensor([0.5])

    first = _online_cache(weight_path)
    first.build([plan_a], embed=_embed)

    fresh = _online_cache(weight_path)
    fresh.build([plan_a, plan_b], embed=_embed)
    assert fresh.rebuilds == 1 and fresh.persist_loads == 0

    # The rebuild backfills the store, so the next start loads both plans.
    third = _online_cache(weight_path)
    third.build([plan_a, plan_b], embed=_embed)
    assert third.rebuilds == 0 and third.persist_loads == 1


def test_persist_checkpoint_change_falls_back_to_rebuild(tmp_path, monkeypatch):
    monkeypatch.setenv("MINIMAX_H3_ADALN_PERSIST_DIR", str(tmp_path / "store"))
    weight_path = tmp_path / "model.safetensors"
    _write_online_weights(weight_path, seed=7)
    plan_a = torch.tensor([0.125, 0.999])

    first = _online_cache(weight_path)
    first.build([plan_a], embed=_embed)

    # A different checkpoint fingerprints into a different store, so the old
    # plans are invisible and the pass rebuilds from the new weights.
    _write_online_weights(weight_path, seed=11)
    fresh = _online_cache(weight_path)
    fresh.build([plan_a], embed=_embed)
    assert fresh.rebuilds == 1 and fresh.persist_loads == 0
    _, first_block, _ = _slab_rows(first, plan_a)
    _, fresh_block, _ = _slab_rows(fresh, plan_a)
    assert not torch.equal(first_block, fresh_block)


def test_persist_tampered_plan_file_falls_back_to_rebuild(tmp_path, monkeypatch):
    store = tmp_path / "store"
    monkeypatch.setenv("MINIMAX_H3_ADALN_PERSIST_DIR", str(store))
    weight_path = tmp_path / "model.safetensors"
    _write_online_weights(weight_path)
    plan_a = torch.tensor([0.125, 0.999])

    first = _online_cache(weight_path)
    first.build([plan_a], embed=_embed)
    plan_files = list(store.rglob("plan-*.safetensors"))
    assert len(plan_files) == 1
    save_file({"junk": torch.zeros(1)}, plan_files[0], metadata={"format_version": "0"})

    fresh = _online_cache(weight_path)
    fresh.build([plan_a], embed=_embed)
    assert fresh.rebuilds == 1 and fresh.persist_loads == 0
    fresh.lookup(plan_a)


def test_persist_disabled_without_env(tmp_path, monkeypatch):
    monkeypatch.delenv("MINIMAX_H3_ADALN_PERSIST_DIR", raising=False)
    weight_path = tmp_path / "model.safetensors"
    _write_online_weights(weight_path)

    cache = _online_cache(weight_path)
    cache.build([torch.tensor([0.5])], embed=_embed)
    assert cache.rebuilds == 1
    assert not list(tmp_path.rglob("plan-*.safetensors"))
