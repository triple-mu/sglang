# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
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
from sglang.multimodal_gen.runtime.models.dits.minimax_h3_adaln_cache import (
    MiniMaxH3AdalnCache,
)
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


def _weights_fill(*shape: int, scale: float) -> torch.Tensor:
    values = torch.arange(int(torch.tensor(shape).prod()), dtype=torch.float32)
    return ((values % 7) * 0.01 * scale).reshape(shape)


def _write_online_weights(
    path: Path,
    *,
    omit: str | None = None,
    scale: float = 0.0,
) -> None:
    # State-machine tests only need checkpoint-compatible shapes (scale 0);
    # value-equality tests pass a nonzero scale for distinguishable outputs.
    def _fill(*shape: int) -> torch.Tensor:
        return _weights_fill(*shape, scale=scale)

    tensors: dict[str, torch.Tensor] = {}
    for layer in range(_ARCH.num_layers):
        prefix = f"blocks.{layer}.adaln_proj.linear"
        tensors[f"{prefix}.weight"] = _fill(_BLOCK_WIDTH, _ARCH.time_embed_dim)
        tensors[f"{prefix}.bias"] = _fill(_BLOCK_WIDTH)
    prefix = "final_layer.adaln_proj.linear"
    tensors[f"{prefix}.weight"] = _fill(_FINAL_WIDTH, _ARCH.time_embed_dim)
    tensors[f"{prefix}.bias"] = _fill(_FINAL_WIDTH)
    if omit is not None:
        tensors.pop(omit)
    save_file(tensors, path)


def _online_cache(
    tmp_path: Path,
    *,
    max_plans: int = 2,
    max_plan_width: int = 2,
    omit: str | None = None,
    host_cache_bytes: int = 0,
    scale: float = 0.0,
) -> MiniMaxH3AdalnCache:
    _ensure_single_process_parallel_runtime()
    weight_path = tmp_path / "model.safetensors"
    _write_online_weights(weight_path, omit=omit, scale=scale)
    cache = MiniMaxH3AdalnCache(
        _ARCH,
        weight_files=[str(weight_path)],
        max_plans=max_plans,
        max_plan_width=max_plan_width,
        host_cache_bytes=host_cache_bytes,
    )
    cache.load(torch.device("cpu"))
    return cache


# One host-tier page for the tiny arch, in bytes (see MiniMaxH3AdalnHostTier).
_PAGE_BYTES = (_ARCH.num_layers * _BLOCK_WIDTH + _FINAL_WIDTH) * 2


def _reference_block(cache, index, plan, num_timesteps):
    # Local oracle for block_all: explicit slab indexing, kept out of the
    # production class so a layout bug cannot "fix itself" in both sides.
    params = cache.block_params[plan, :num_timesteps, index]
    return tuple(params.reshape(-1, 6, _ARCH.hidden_size).unbind(dim=1))


def _embed(timesteps: torch.Tensor) -> torch.Tensor:
    return timesteps[:, None].expand(-1, _ARCH.time_embed_dim)


def test_minimax_h3_adaln_cache_matches_bf16_embedding(tmp_path):
    cache_path = tmp_path / "adaln.safetensors"
    plan_timesteps = torch.tensor([[0.0, 0.0], [1.0, 2.0]])
    plan_lengths = torch.tensor([1, 2], dtype=torch.int64)
    block_params = (
        torch.arange(2 * 2 * 2 * _BLOCK_WIDTH, dtype=torch.float32)
        .reshape(2, 2, 2, _BLOCK_WIDTH)
        .bfloat16()
    )
    final_params = (
        torch.arange(2 * 2 * _FINAL_WIDTH, dtype=torch.float32)
        .reshape(2, 2, _FINAL_WIDTH)
        .bfloat16()
    )
    save_file(
        {
            "plan_timesteps": plan_timesteps,
            "plan_lengths": plan_lengths,
            "block_params": block_params,
            "final_params": final_params,
        },
        cache_path,
        metadata={"format_version": "2", "model_variant": "fl2va"},
    )

    cache = MiniMaxH3AdalnCache(
        _ARCH,
        path=str(cache_path),
        model_variant="fl2va",
    )
    cache.load(torch.device("cpu"))

    cache_plan_index = cache.lookup(plan_timesteps[1])
    block = _reference_block(cache, 1, cache_plan_index, 2)
    final = cache.final(cache_plan_index, 2)

    # block() hands the forward pass six [num_timesteps * modality, hidden]
    # chunks, while the checkpoint stores a plan as one flat
    # [num_timesteps, 6 * modality * hidden] row -- same elements, and the
    # modality axis folds into the leading one rather than staying separate.
    assert torch.equal(
        torch.cat(block, dim=-1).reshape(block_params[1, :, 1].shape),
        block_params[1, :, 1],
    )
    assert torch.equal(torch.cat(final, dim=-1), final_params[1])


def test_sidecar_resolve_slots_and_block_all_match_per_step_paths(tmp_path):
    """Host-resolved slots and the batched gather must mirror lookup/block."""
    cache_path = tmp_path / "adaln.safetensors"
    plan_timesteps = torch.tensor([[0.5, 0.0], [1.0, 2.0]])
    plan_lengths = torch.tensor([1, 2], dtype=torch.int64)
    block_params = (
        torch.arange(2 * 2 * 2 * _BLOCK_WIDTH, dtype=torch.float32)
        .reshape(2, 2, 2, _BLOCK_WIDTH)
        .bfloat16()
    )
    final_params = torch.zeros(2, 2, _FINAL_WIDTH, dtype=torch.bfloat16)
    save_file(
        {
            "plan_timesteps": plan_timesteps,
            "plan_lengths": plan_lengths,
            "block_params": block_params,
            "final_params": final_params,
        },
        cache_path,
        metadata={"format_version": "2", "model_variant": "fl2va"},
    )
    cache = MiniMaxH3AdalnCache(_ARCH, path=str(cache_path), model_variant="fl2va")
    cache.load(torch.device("cpu"))

    slots = cache.resolve_slots([torch.tensor([0.5]), torch.tensor([1.0, 2.0])])
    assert slots.dtype == torch.int64
    assert slots.tolist() == [0, 1]
    assert int(cache.lookup(torch.tensor([1.0, 2.0]))) == int(slots[1])

    stacked = cache.block_all(cache_plan_index=slots[1], num_timesteps=2)
    assert len(stacked) == _ARCH.num_layers
    for index in range(_ARCH.num_layers):
        expected = _reference_block(cache, index, slots[1], 2)
        for got, want in zip(stacked[index], expected):
            assert torch.equal(got, want)
            assert got.stride() == want.stride()

    with pytest.raises(ValueError, match="does not cover"):
        cache.resolve_slots([torch.tensor([9.0])])


def test_online_cache_resolve_slots_after_build(tmp_path):
    cache = _online_cache(tmp_path, max_plan_width=2)
    plan_a = torch.tensor([1.0])
    plan_b = torch.tensor([2.0, 3.0])

    cache.build([plan_a, plan_b, plan_a], embed=_embed)
    slots = cache.resolve_slots([plan_a, plan_b, plan_a])
    assert slots.tolist()[0] == slots.tolist()[2]
    assert int(cache.lookup(plan_b)) == int(slots[1])


def test_slab_buffers_stay_out_of_state_dict(tmp_path):
    cache = _online_cache(tmp_path)
    assert not any("params" in key or "plan" in key for key in cache.state_dict())


def test_host_tier_swap_in_restores_evicted_plans_bit_exactly(tmp_path):
    """A GPU-evicted plan set must return from the host tier byte-identical,
    without another checkpoint pass."""
    cache = _online_cache(
        tmp_path,
        max_plans=2,
        max_plan_width=1,
        host_cache_bytes=64 * _PAGE_BYTES,
        scale=1.0,
    )
    set_a = [torch.tensor([1.0]), torch.tensor([2.0])]
    set_b = [torch.tensor([3.0]), torch.tensor([4.0])]

    cache.build(set_a, embed=_embed)
    slots_a = cache.resolve_slots(set_a)
    snapshot = [
        (
            [t.clone() for t in _reference_block(cache, 0, slots_a[i], 1)],
            [t.clone() for t in cache.final(slots_a[i], 1)],
        )
        for i in range(2)
    ]
    assert cache.stats.built_plans == 2

    cache.build(set_b, embed=_embed)  # evicts set_a from the GPU slab
    passes = cache.rebuilds
    cache.build(set_a, embed=_embed)  # swaps back in from the host tier
    assert cache.rebuilds == passes
    assert cache.stats.host_hit_plans == 2

    slots_a = cache.resolve_slots(set_a)
    for i in range(2):
        blocks, finals = snapshot[i]
        for got, want in zip(_reference_block(cache, 0, slots_a[i], 1), blocks):
            assert torch.equal(got, want)
        for got, want in zip(cache.final(slots_a[i], 1), finals):
            assert torch.equal(got, want)
        assert float(cache.plan_timesteps[slots_a[i], 0]) == float(set_a[i][0])


def test_host_tier_over_capacity_group_recomputes(tmp_path):
    """A group that cannot fit is skipped (never raises) and rebuilt later."""
    cache = _online_cache(
        tmp_path,
        max_plans=2,
        max_plan_width=1,
        host_cache_bytes=1 * _PAGE_BYTES,  # one page: a 2-plan group never fits
    )
    set_a = [torch.tensor([1.0]), torch.tensor([2.0])]
    set_b = [torch.tensor([3.0]), torch.tensor([4.0])]

    cache.build(set_a, embed=_embed)
    assert cache.stats.host_pressure_skips == 1
    cache.build(set_b, embed=_embed)
    passes = cache.rebuilds
    cache.build(set_a, embed=_embed)  # host tier empty: full rebuild again
    assert cache.rebuilds == passes + 1


def test_host_tier_lru_eviction_and_shared_plan_refcount(tmp_path):
    cache = _online_cache(
        tmp_path,
        max_plans=4,
        max_plan_width=1,
        host_cache_bytes=3 * _PAGE_BYTES,
    )
    plan_a = torch.tensor([1.0])
    plan_b = torch.tensor([2.0])
    plan_c = torch.tensor([3.0])

    cache.build([plan_a, plan_b], embed=_embed)  # group 1 {a, b}: 2 pages
    cache.build([plan_a, plan_c], embed=_embed)  # group 2 {a, c}: +1 page (a shared)
    tier = cache._host_tier
    assert tier is not None
    assert len(tier._plans) == 3 and len(tier._free_pages) == 0

    # A third group needs a page; group 1 is LRU. Its shared plan_a must
    # survive because group 2 still references it.
    plan_d = torch.tensor([4.0])
    cache.build([plan_a, plan_d], embed=_embed)
    assert cache.stats.host_evicted_groups == 1
    import struct

    keys = {
        tuple(struct.unpack("<f", struct.pack("<I", bits))[0] for bits in key)
        for key in tier._plans
    }
    assert keys == {(1.0,), (3.0,), (4.0,)}


def test_precision_fp32_projects_in_fp32_then_stores_bf16(tmp_path):
    _ensure_single_process_parallel_runtime()
    weight_path = tmp_path / "model.safetensors"
    _write_online_weights(weight_path, scale=1.0)
    cache = MiniMaxH3AdalnCache(
        _ARCH,
        weight_files=[str(weight_path)],
        max_plans=2,
        max_plan_width=1,
        precision="fp32",
    )
    cache.load(torch.device("cpu"))
    plan = torch.tensor([0.31])

    def bf16_embed(timesteps: torch.Tensor) -> torch.Tensor:
        # The production embed stays fp32 in this mode; a bf16 input here
        # proves the cache upcasts before the projection (a 'match'-style
        # bf16 x fp32 GEMM would fail on dtype mismatch).
        return _embed(timesteps).bfloat16()

    cache.build([plan], embed=bf16_embed)
    slot = cache.resolve_slots([plan])[0]

    adaln_input = bf16_embed(plan).float()
    weight = _weights_fill(_BLOCK_WIDTH, _ARCH.time_embed_dim, scale=1.0)
    bias = _weights_fill(_BLOCK_WIDTH, scale=1.0)
    for layer in range(_ARCH.num_layers):
        expected = torch.nn.functional.linear(adaln_input, weight, bias).bfloat16()
        got = torch.cat(_reference_block(cache, layer, slot, 1), dim=-1).reshape(
            1, _BLOCK_WIDTH
        )
        assert torch.equal(got, expected)

    with pytest.raises(ValueError, match="precision"):
        MiniMaxH3AdalnCache(_ARCH, weight_files=[str(weight_path)], precision="fp64")


def test_invalidate_drops_all_tiers_and_allows_rebuild(tmp_path):
    cache = _online_cache(
        tmp_path,
        max_plans=2,
        max_plan_width=1,
        host_cache_bytes=64 * _PAGE_BYTES,
    )
    plan_a = torch.tensor([1.0])
    cache.build([plan_a], embed=_embed)

    cache.invalidate()
    with pytest.raises(ValueError, match="does not cover"):
        cache.lookup(plan_a)
    with pytest.raises(ValueError, match="does not cover"):
        cache.resolve_slots([plan_a])

    passes = cache.rebuilds
    cache.build([plan_a], embed=_embed)  # host tier was cleared too
    assert cache.rebuilds == passes + 1
    cache.lookup(plan_a)


def test_online_cache_eviction_preserves_in_flight_request_plans(tmp_path):
    """Capacity eviction must never drop plans reused by the current request."""
    cache = _online_cache(tmp_path, max_plan_width=1)
    plan_a = torch.tensor([1.0])
    plan_b = torch.tensor([2.0])
    plan_c = torch.tensor([3.0])

    cache.build([plan_a, plan_b], embed=_embed)
    cache.build([plan_a, plan_c], embed=_embed)

    cache.lookup(plan_a)
    cache.lookup(plan_c)
    with pytest.raises(ValueError, match="does not cover"):
        cache.lookup(plan_b)


def test_online_cache_lru_keeps_alternating_plan_sets_resident(tmp_path):
    """Two alternating schedules must both stay resident once built.

    The pre-LRU slab did a full reset whenever slots overflowed, so two
    alternating plan sets re-read the whole checkpoint on every request.
    """
    cache = _online_cache(tmp_path, max_plans=4, max_plan_width=1)
    set_a = [torch.tensor([1.0]), torch.tensor([2.0])]
    set_b = [torch.tensor([3.0]), torch.tensor([4.0])]

    cache.build(set_a, embed=_embed)
    cache.build(set_b, embed=_embed)
    passes = cache.rebuilds
    cache.build(set_a, embed=_embed)
    cache.build(set_b, embed=_embed)
    assert cache.rebuilds == passes

    slots_a = cache.resolve_slots(set_a)
    slots_b = cache.resolve_slots(set_b)
    assert sorted(slots_a.tolist() + slots_b.tolist()) == [0, 1, 2, 3]


def test_online_cache_evicts_least_recently_used_plan_first(tmp_path):
    cache = _online_cache(tmp_path, max_plans=2, max_plan_width=1)
    plan_a = torch.tensor([1.0])
    plan_b = torch.tensor([2.0])
    plan_c = torch.tensor([3.0])

    cache.build([plan_a], embed=_embed)
    cache.build([plan_b], embed=_embed)
    # Touch plan_a so plan_b becomes the LRU entry, then overflow with plan_c.
    cache.build([plan_a, plan_c], embed=_embed)

    cache.lookup(plan_a)
    cache.lookup(plan_c)
    with pytest.raises(ValueError, match="does not cover"):
        cache.lookup(plan_b)
    with pytest.raises(ValueError, match="does not cover"):
        cache.resolve_slots([plan_b])


def test_online_cache_failed_rebuild_can_be_retried(tmp_path):
    """A failed rebuild must not publish a cache hit that blocks its retry."""
    missing_name = "final_layer.adaln_proj.linear.bias"
    cache = _online_cache(tmp_path, omit=missing_name)
    plan_a = torch.tensor([1.0])

    with pytest.raises(KeyError, match=missing_name):
        cache.build([plan_a], embed=_embed)

    _write_online_weights(tmp_path / "model.safetensors")
    cache.build([plan_a], embed=_embed)
    cache.lookup(plan_a)


def test_online_cache_width_rejection_preserves_resident_plans(tmp_path):
    """Rejecting an over-width plan must not evict usable resident plans."""
    cache = _online_cache(tmp_path, max_plan_width=1)
    plan_a = torch.tensor([1.0])
    plan_b = torch.tensor([2.0])
    wide_plan = torch.tensor([3.0, 4.0])

    cache.build([plan_a, plan_b], embed=_embed)
    with pytest.raises(ValueError, match="--minimax-h3-adaln-plan-width"):
        cache.build([wide_plan], embed=_embed)

    cache.lookup(plan_a)
    cache.lookup(plan_b)


_ADALN_PREFIXES = [
    f"blocks.{layer}.adaln_proj.linear" for layer in range(_ARCH.num_layers)
] + ["final_layer.adaln_proj.linear"]


def _adaln_width(prefix: str) -> int:
    return _FINAL_WIDTH if prefix.startswith("final_layer") else _BLOCK_WIDTH


def _dyadic(generator: torch.Generator, *shape: int) -> torch.Tensor:
    # Multiples of 1/4 keep every product and sum below exactly representable
    # in bf16, so the slab compares bit-for-bit against the merged-weight oracle.
    return torch.randint(0, 3, shape, generator=generator).float() * 0.25


def _online_lora_fixture(
    tmp_path: Path, *, precision: str = "match", rank: int = 2, scale: float = 0.5
) -> tuple[MiniMaxH3AdalnCache, dict[str, torch.Tensor], dict]:
    """An online cache over dyadic weights plus matching dyadic LoRA deltas.

    'match' stores the checkpoint in bf16 so the rebuild runs the production
    bf16 GEMM plus fp32 delta path; 'fp32' keeps fp32 weights.
    """
    _ensure_single_process_parallel_runtime()
    generator = torch.Generator().manual_seed(0)
    weights: dict[str, torch.Tensor] = {}
    deltas: dict[str, list[tuple[torch.Tensor, torch.Tensor, float]]] = {}
    weight_dtype = torch.bfloat16 if precision == "match" else torch.float32
    for prefix in _ADALN_PREFIXES:
        width = _adaln_width(prefix)
        weights[f"{prefix}.weight"] = _dyadic(
            generator, width, _ARCH.time_embed_dim
        ).to(weight_dtype)
        weights[f"{prefix}.bias"] = _dyadic(generator, width).to(weight_dtype)
        deltas[prefix] = [
            (
                _dyadic(generator, rank, _ARCH.time_embed_dim),
                _dyadic(generator, width, rank),
                scale,
            )
        ]
    weight_path = tmp_path / "model.safetensors"
    save_file(weights, weight_path)
    cache = MiniMaxH3AdalnCache(
        _ARCH,
        weight_files=[str(weight_path)],
        max_plans=2,
        max_plan_width=1,
        precision=precision,
    )
    cache.load(torch.device("cpu"))
    return cache, weights, deltas


def _merged_weight_reference(weights, deltas, prefix: str, x: torch.Tensor):
    weight = weights[f"{prefix}.weight"].float()
    for lora_a, lora_b, scale in deltas.get(prefix, []):
        weight = weight + scale * (lora_b @ lora_a)
    bias = weights[f"{prefix}.bias"].float()
    return torch.nn.functional.linear(x.float(), weight, bias).bfloat16()


def _embed_like_checkpoint(weights) -> callable:
    dtype = next(iter(weights.values())).dtype
    return lambda timesteps: _embed(timesteps).to(dtype)


def _plan_row(cache, slot, prefix: str) -> torch.Tensor:
    if prefix.startswith("final_layer"):
        return torch.cat(cache.final(slot, 1), dim=-1).reshape(1, _FINAL_WIDTH)
    layer = int(prefix.split(".")[1])
    return torch.cat(_reference_block(cache, layer, slot, 1), dim=-1).reshape(
        1, _BLOCK_WIDTH
    )


@pytest.mark.parametrize("precision", ["match", "fp32"])
def test_online_cache_applies_adaln_lora_deltas(tmp_path, precision):
    """A rebuilt plan must equal F.linear(x, W + scale * B @ A, bias)."""
    cache, weights, deltas = _online_lora_fixture(tmp_path, precision=precision)
    cache.set_lora_deltas(deltas)
    plan = torch.tensor([2.0])
    embed = _embed_like_checkpoint(weights)
    cache.build([plan], embed=embed)
    slot = cache.resolve_slots([plan])[0]
    x = embed(plan)
    assert x.dtype == (torch.bfloat16 if precision == "match" else torch.float32)

    for prefix in _ADALN_PREFIXES:
        got = _plan_row(cache, slot, prefix)
        assert torch.equal(got, _merged_weight_reference(weights, deltas, prefix, x))
        # The oracle must be sensitive to the delta or the test proves nothing.
        assert not torch.equal(got, _merged_weight_reference(weights, {}, prefix, x))


def test_online_cache_lora_deltas_invalidate_plans(tmp_path):
    cache, weights, deltas = _online_lora_fixture(tmp_path)
    embed = _embed_like_checkpoint(weights)
    plan = torch.tensor([2.0])
    cache.build([plan], embed=embed)
    baseline = cache.block_params[cache.resolve_slots([plan])[0]].clone()

    # Clearing deltas that were never set is a no-op: a LoRA without adaln
    # keys must not throw away resident plans.
    cache.set_lora_deltas({})
    assert cache._slots

    cache.set_lora_deltas(deltas)
    assert not cache._slots and int(cache.plan_lengths.sum()) == 0
    with pytest.raises(ValueError, match="does not cover"):
        cache.lookup(plan)
    cache.build([plan], embed=embed)
    with_lora = cache.block_params[cache.resolve_slots([plan])[0]].clone()
    assert not torch.equal(with_lora, baseline)

    # merge_lora_weights re-hands the same adapters at the same strength on
    # every generate_with_lora call; identical deltas must keep the plans.
    cache.set_lora_deltas({prefix: list(entries) for prefix, entries in deltas.items()})
    assert cache._slots
    lora_a, lora_b, scale = deltas[_ADALN_PREFIXES[0]][0]
    rescaled = dict(deltas)
    rescaled[_ADALN_PREFIXES[0]] = [(lora_a, lora_b, scale * 2)]
    cache.set_lora_deltas(rescaled)
    assert not cache._slots

    cache.set_lora_deltas(None)
    assert not cache._slots
    cache.build([plan], embed=embed)
    restored = cache.block_params[cache.resolve_slots([plan])[0]]
    assert torch.equal(restored, baseline)


def test_online_cache_lora_delta_shape_validation(tmp_path):
    cache = _online_cache(tmp_path)
    lora_a = torch.zeros(2, _ARCH.time_embed_dim)
    lora_b = torch.zeros(_BLOCK_WIDTH, 2)
    block = "blocks.0.adaln_proj.linear"
    final = "final_layer.adaln_proj.linear"

    with pytest.raises(ValueError, match="no adaln_proj layer"):
        cache.set_lora_deltas({"blocks.9.adaln_proj.linear": [(lora_a, lora_b, 1.0)]})
    with pytest.raises(ValueError, match="lora_A"):
        cache.set_lora_deltas({block: [(torch.zeros(2, 5), lora_b, 1.0)]})
    with pytest.raises(ValueError, match="lora_B"):
        cache.set_lora_deltas({final: [(lora_a, lora_b, 1.0)]})
    with pytest.raises(ValueError, match="lora_B"):
        cache.set_lora_deltas({block: [(lora_a, torch.zeros(_BLOCK_WIDTH, 3), 1.0)]})
    assert cache._lora_deltas == {}

    with pytest.raises(ValueError, match="sidecar"):
        _sidecar_cache(tmp_path).set_lora_deltas({block: [(lora_a, lora_b, 1.0)]})


def _sidecar_cache(tmp_path: Path) -> MiniMaxH3AdalnCache:
    # The weight-update guards only read which tier the cache was built as, so
    # the sidecar never has to be loaded here.
    return MiniMaxH3AdalnCache(_ARCH, path=str(tmp_path / "adaln.safetensors"))


def _cache_mode_model(cache: MiniMaxH3AdalnCache | None):
    from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
        MiniMaxH3DiTModel,
    )

    model = MiniMaxH3DiTModel.__new__(MiniMaxH3DiTModel)
    torch.nn.Module.__init__(model)
    model._adaln_precomputed = True
    model.adaln_cache = cache
    return model


def test_sidecar_mode_rejects_weight_updates(tmp_path):
    """A sidecar is built offline; no update can keep it in step."""
    model = _cache_mode_model(_sidecar_cache(tmp_path))
    for weights_path in (str(tmp_path), None):
        with pytest.raises(ValueError, match="sidecar"):
            model.validate_weight_update_source(weights_path=weights_path)


def test_online_cache_rejects_tensor_weight_updates(tmp_path):
    """Tensor RPC carries no directory the rebuild could stream adaln from."""
    model = _cache_mode_model(_online_cache(tmp_path))
    with pytest.raises(ValueError, match="update_weights_from_disk"):
        model.validate_weight_update_source(weights_path=None)


def test_online_cache_rejects_update_source_without_native_adaln(tmp_path):
    model = _cache_mode_model(_online_cache(tmp_path))
    diffusers_layout = tmp_path / "diffusers"
    diffusers_layout.mkdir()
    save_file({"unrelated": torch.zeros(1)}, diffusers_layout / "model.safetensors")

    for weights_path in (str(diffusers_layout), str(tmp_path / "absent")):
        with pytest.raises(ValueError, match="no native adaln_proj"):
            model.validate_weight_update_source(weights_path=weights_path)


def test_disk_update_retargets_rebuild_source_and_drops_plans(tmp_path):
    cache = _online_cache(tmp_path, max_plan_width=1)
    model = _cache_mode_model(cache)
    plan = torch.tensor([1.0])
    cache.build([plan], embed=_embed)
    updated = tmp_path / "updated"
    updated.mkdir()
    _write_online_weights(updated / "model.safetensors")

    model.validate_weight_update_source(weights_path=str(updated))
    model.refresh_weight_derived_caches(weights_path=str(updated))

    assert cache.weight_files == [str(updated / "model.safetensors")]
    with pytest.raises(ValueError, match="does not cover"):
        cache.lookup(plan)


def test_sidecar_lora_guard_rejects_adaln_keys(tmp_path):
    model = _cache_mode_model(_sidecar_cache(tmp_path))
    adapter = {"blocks.0.adaln_proj.linear.lora_A": torch.zeros(1)}
    with pytest.raises(ValueError, match="sidecar"):
        model.prepare_lora_adapter(adapter)


def test_online_lora_guard_keeps_adaln_keys(tmp_path):
    """Online mode leaves adaln keys for apply_lora_extra_targets to consume."""
    model = _cache_mode_model(_online_cache(tmp_path))
    model.arch = _ARCH
    adapter = {
        "blocks.0.adaln_proj.linear.lora_A": torch.zeros(2, _ARCH.time_embed_dim),
        "blocks.0.adaln_proj.linear.lora_B": torch.zeros(_BLOCK_WIDTH, 2),
    }
    assert model.prepare_lora_adapter(adapter) is adapter


def test_online_lora_guard_validates_before_layers_are_touched(tmp_path):
    """Every rejection the cache would raise surfaces from prepare_lora_adapter."""
    model = _cache_mode_model(_online_cache(tmp_path))
    model.arch = _ARCH
    prefix = "blocks.0.adaln_proj.linear"

    with pytest.raises(ValueError, match="missing"):
        model.prepare_lora_adapter(
            {f"{prefix}.lora_A": torch.zeros(2, _ARCH.time_embed_dim)}
        )
    with pytest.raises(ValueError, match="lora_B"):
        model.prepare_lora_adapter(
            {
                f"{prefix}.lora_A": torch.zeros(2, _ARCH.time_embed_dim),
                f"{prefix}.lora_B": torch.zeros(_FINAL_WIDTH, 2),
            }
        )

    curve_arch = MiniMaxH3DiTArchConfig(
        num_layers=_ARCH.num_layers,
        hidden_size=_ARCH.hidden_size,
        time_embed_dim=_ARCH.time_embed_dim,
        adaln_affine_input_dim=7,
    )
    model.arch = curve_arch
    with pytest.raises(ValueError, match="released-checkpoint"):
        model.prepare_lora_adapter(
            {
                f"{prefix}.lora_A": torch.zeros(2, 7),
                f"{prefix}.lora_B": torch.zeros(_BLOCK_WIDTH, 2),
            }
        )


def _adaln_adapter(prefix: str, rank: int) -> dict[str, torch.Tensor]:
    return {
        f"{prefix}.lora_A": torch.zeros(rank, _ARCH.time_embed_dim),
        f"{prefix}.lora_B": torch.zeros(_adaln_width(prefix), rank),
    }


def test_apply_lora_extra_targets_scales_like_layers(tmp_path, monkeypatch):
    """Per-layer alpha, adapter alpha, then rank: the BaseLayerWithLoRA order."""
    cache = _online_cache(tmp_path)
    model = _cache_mode_model(cache)
    handed_over = []
    set_lora_deltas = cache.set_lora_deltas
    monkeypatch.setattr(
        cache,
        "set_lora_deltas",
        lambda deltas: (handed_over.append(deltas), set_lora_deltas(deltas)),
    )
    prefix = "blocks.1.adaln_proj.linear"
    with_layer_alpha = _adaln_adapter(prefix, 16)
    with_layer_alpha[f"{prefix}.alpha"] = torch.tensor(32.0)
    plain = _adaln_adapter(prefix, 16)

    covered = model.apply_lora_extra_targets(
        [
            ("layer_alpha", with_layer_alpha, 0.5, 8),
            ("adapter_alpha", plain, 0.5, 8),
            ("bare", plain, 0.5, None),
        ],
        merge=True,
    )

    assert covered == 1 and len(handed_over) == 1
    assert [scale for _, _, scale in cache._lora_deltas[prefix]] == [1.0, 0.25, 0.5]
    assert model.apply_lora_extra_targets([], merge=True) == 0
    assert cache._lora_deltas == {}

    pruned = dict(plain)
    pruned[f"{prefix}.lora_output_offset"] = torch.zeros(_BLOCK_WIDTH)
    with pytest.raises(ValueError, match="pruned-AdaLN"):
        model.apply_lora_extra_targets([("pruned", pruned, 1.0, None)], merge=True)

    sidecar_model = _cache_mode_model(_sidecar_cache(tmp_path))
    assert (
        sidecar_model.apply_lora_extra_targets([("bare", plain, 1.0, None)], merge=True)
        == 0
    )


def test_lora_pipeline_routes_extra_targets_to_the_dit(tmp_path):
    """set_lora's target names reach the DiT that owns them and nothing else."""
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.pipelines_core.lora.pipeline import (
        LoRAPipeline,
    )

    cache = _online_cache(tmp_path)
    model = _cache_mode_model(cache)
    prefix = "final_layer.adaln_proj.linear"
    pipeline = SimpleNamespace(
        modules={"transformer": model, "fake_score_transformer": torch.nn.Linear(1, 1)},
        lora_adapters={"turbo": _adaln_adapter(prefix, 4)},
        loaded_adapter_alphas={"turbo": None},
    )

    def route(module_name: str) -> int:
        return LoRAPipeline._apply_lora_extra_targets(
            pipeline, module_name, ["turbo"], [1.0], merge=True
        )

    assert route("transformer") == 1
    assert list(cache._lora_deltas) == [prefix]
    assert route("critic") == 0
    assert route("transformer_2") == 0


class _FakeLoraLayer:
    """The wrapped-layer surface merge/unmerge/deactivate and the IPC path touch."""

    def __init__(self, merged: bool):
        self.lora_A = torch.zeros(1)
        self.weight = torch.zeros(1)
        self.can_merge_base_weight = True
        self.merged = merged
        self.disable_lora = False
        self.strength = 1.0
        self.set_calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def merge_lora_weights(self, strength: float) -> None:
        self.merged = True
        self.strength = strength

    def unmerge_lora_weights(self) -> None:
        self.merged = False

    def set_lora_weights(self, lora_a, lora_b, **_) -> None:
        self.set_calls.append((lora_a, lora_b))


def _lora_pipeline_with_turbo(tmp_path: Path, *, merge_mode: str):
    """A LoRAPipeline whose only wrapped layer and adaLN cache carry 'turbo'."""
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.pipelines_core.lora.pipeline import (
        LoRAPipeline,
    )

    class _Pipeline(LoRAPipeline):
        def create_pipeline_stages(self, server_args):
            return None

    cache = _online_cache(tmp_path)
    model = _cache_mode_model(cache)
    model.layerwise_offload_managers = []
    layer = _FakeLoraLayer(merged=merge_mode == "merge")
    pipeline = object.__new__(_Pipeline)
    pipeline.modules = {"transformer": model}
    pipeline.lora_layers = {"blocks.0.attn.qkv_proj": layer}
    pipeline.lora_layers_transformer_2 = {}
    pipeline.lora_layers_critic = {}
    pipeline.lora_initialized = True
    pipeline.lora_adapters = {"turbo": _adaln_adapter(_ADALN_PREFIXES[0], 4)}
    pipeline.loaded_adapter_paths = {"turbo": "/turbo"}
    pipeline.loaded_adapter_alphas = {"turbo": None}
    pipeline.server_args = SimpleNamespace(lora_merge_mode=merge_mode)
    pipeline.cur_adapter_name = {"transformer": "turbo"}
    pipeline.cur_adapter_strength = {"transformer": 1.0}
    pipeline.cur_adapter_config = {"transformer": (["turbo"], [1.0])}
    pipeline.is_lora_merged = {"transformer": merge_mode == "merge"}
    pipeline._apply_lora_extra_targets("transformer", ["turbo"], [1.0], merge=True)
    cache.build([torch.tensor([2.0])], embed=_embed)
    assert cache._slots and cache._lora_deltas
    return pipeline, cache, layer


@pytest.mark.parametrize("merge_mode", ["dynamic", "merge"])
def test_lora_pipeline_unmerge_clears_adaln_deltas(tmp_path, merge_mode):
    pipeline, cache, layer = _lora_pipeline_with_turbo(tmp_path, merge_mode=merge_mode)

    pipeline.unmerge_lora_weights("transformer")

    assert layer.disable_lora and not layer.merged
    assert cache._lora_deltas == {} and not cache._slots


def test_lora_pipeline_deactivate_clears_adaln_deltas(tmp_path):
    pipeline, cache, layer = _lora_pipeline_with_turbo(tmp_path, merge_mode="merge")

    pipeline.deactivate_lora_weights("transformer")

    assert layer.disable_lora and not layer.merged
    assert cache._lora_deltas == {} and not cache._slots


@pytest.mark.parametrize("merge_mode", ["dynamic", "merge"])
def test_lora_pipeline_merge_strength_reaches_adaln_deltas(tmp_path, merge_mode):
    """merge_lora_weights(strength) rescales the cache like the wrapped layers."""
    pipeline, cache, layer = _lora_pipeline_with_turbo(tmp_path, merge_mode=merge_mode)
    prefix = _ADALN_PREFIXES[0]

    pipeline.merge_lora_weights("transformer", strength=0.5)
    assert layer.strength == 0.5
    assert [scale for _, _, scale in cache._lora_deltas[prefix]] == [0.5]
    assert not cache._slots

    # After unmerge the wrapped layers still hold the adapter and a bare merge
    # re-activates it; the cache follows through the set adapter name.
    pipeline.unmerge_lora_weights("transformer")
    assert cache._lora_deltas == {}
    pipeline.merge_lora_weights("transformer", strength=2.0)
    assert [scale for _, _, scale in cache._lora_deltas[prefix]] == [2.0]


def test_lora_ipc_update_clears_adaln_deltas_from_lora_path(tmp_path):
    """An IPC adapter replaces the wrapped layers; a prior disk adapter's adaLN deltas go too."""
    from sglang.multimodal_gen.runtime.post_training.weights_updater import (
        LORA_MERGE_WEIGHT_UPDATE_MODE,
        WeightsUpdater,
    )

    pipeline, cache, layer = _lora_pipeline_with_turbo(tmp_path, merge_mode="dynamic")
    payload = [
        ("blocks.0.attn.qkv_proj.lora_A.weight", torch.zeros(2, 3)),
        ("blocks.0.attn.qkv_proj.lora_B.weight", torch.zeros(5, 2)),
    ]

    ok, message = WeightsUpdater(pipeline).update_weights_from_tensor(
        {"transformer": payload},
        target_modules=["transformer"],
        weight_update_mode=LORA_MERGE_WEIGHT_UPDATE_MODE,
    )

    assert ok, message
    assert len(layer.set_calls) == 1
    assert cache._lora_deltas == {} and not cache._slots


def test_lora_ipc_layer_guard_rejects_adaln_in_cache_mode():
    """IPC passes module prefixes, not the '.lora_A' keys the disk path sees."""
    model = _cache_mode_model(None)
    model.validate_lora_layers(["blocks.0.attn.qkv_proj"])
    with pytest.raises(ValueError, match="adaln_proj"):
        model.validate_lora_layers(["blocks.0.adaln_proj.linear"])


def _updater_for(model, model_path: str):
    from types import SimpleNamespace

    from sglang.multimodal_gen.runtime.post_training.weights_updater import (
        WeightsUpdater,
    )

    model.register_parameter("probe", torch.nn.Parameter(torch.zeros(2)))
    pipeline = SimpleNamespace(modules={"transformer": model}, model_path=model_path)
    return WeightsUpdater(pipeline), pipeline


def test_weights_updater_rejects_sidecar_update_before_writing_weights(tmp_path):
    model = _cache_mode_model(_sidecar_cache(tmp_path))
    updater, pipeline = _updater_for(model, str(tmp_path))
    new_checkpoint = tmp_path / "new"
    (new_checkpoint / "transformer").mkdir(parents=True)
    save_file(
        {"probe": torch.ones(2)}, new_checkpoint / "transformer" / "model.safetensors"
    )

    ok, message = updater.update_weights_from_disk(str(new_checkpoint))

    assert not ok
    assert "sidecar" in message
    # The rejection has to land before _apply_weights touches anything.
    assert torch.equal(model.probe, torch.zeros(2))
    assert pipeline.model_path == str(tmp_path)


def test_weights_updater_rejects_tensor_update_in_online_cache_mode(tmp_path):
    model = _cache_mode_model(_online_cache(tmp_path))
    updater, _ = _updater_for(model, str(tmp_path))

    ok, message = updater.update_weights_from_tensor([("probe", torch.ones(2))])

    assert not ok
    assert "update_weights_from_disk" in message
    assert torch.equal(model.probe, torch.zeros(2))
