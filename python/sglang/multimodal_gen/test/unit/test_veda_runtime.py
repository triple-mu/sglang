# SPDX-License-Identifier: Apache-2.0
"""Tests of the Miowtion glue behind the Veda attention backend.

The first half needs no Miowtion (pure slicing and plan lookup); the second
half runs against the real package when it is installed and checks that a
bundle sliced to some heads reproduces the full-head student bit for bit.
"""

import importlib.util
from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.runtime.layers.attention.backends import veda_runtime
from sglang.multimodal_gen.runtime.layers.attention.backends.veda_runtime import (
    HeadRange,
    find_plan,
    head_range,
    slice_head_shape,
    slice_state_dict,
)

# The 12 (geometry, grid) pairs of the released MiniMax-H3 T2VA bundle.
_RELEASED_GRIDS = [
    (f"{aspect}_t{t}", (t, h, w))
    for aspect, (h, w) in (
        ("16x9", (24, 42)),
        ("9x16", (42, 24)),
        ("4x3", (24, 32)),
        ("1x1", (24, 24)),
    )
    for t in (37, 72, 102)
]


def _table(entries, mirrored=()):
    plans = {
        name: SimpleNamespace(
            geometry=name,
            grid=grid,
            provenance={"mirrored_from": "x"} if name in mirrored else {},
        )
        for name, grid in entries
    }
    return SimpleNamespace(plans=plans)


def test_head_range_tp_block_then_ulysses_slice():
    assert head_range(28, 7, 1, 2) == HeadRange(42, 49)
    assert head_range(56, 56, 0, 0) == HeadRange(0, 56)
    assert len(head_range(28, 7, 0, 3)) == 7
    with pytest.raises(ValueError):
        head_range(28, 5, 0, 0)


def test_slicing_helpers():
    assert slice_head_shape([[0, 1, 1, 0, 1, 0, 0, 1]], HeadRange(4, 8)) == [
        [1, 0, 0, 1]
    ]
    state = {"layers.0.proj_q": torch.arange(8).view(8, 1, 1)}
    sliced = slice_state_dict(state, HeadRange(4, 8))
    assert sliced["layers.0.proj_q"].flatten().tolist() == [4, 5, 6, 7]
    with pytest.raises(ValueError):
        slice_state_dict(state, HeadRange(4, 9))


def test_find_plan_exact_only_by_default():
    table = _table(_RELEASED_GRIDS, mirrored={"9x16_t37", "9x16_t72", "9x16_t102"})
    plan, reason = find_plan(table, (37, 24, 42), "none")
    assert plan.geometry == "16x9_t37" and reason == "exact"
    plan, _ = find_plan(table, (72, 42, 24), "none")
    assert plan.geometry == "9x16_t72"
    with pytest.raises(ValueError, match=r"\(52, 24, 42\)"):
        find_plan(table, (52, 24, 42), "none")


def test_find_plan_prefers_a_searched_plan_over_a_mirrored_twin():
    table = _table(
        [("a_t37", (37, 24, 24)), ("b_t37", (37, 24, 24))], mirrored={"a_t37"}
    )
    plan, _ = find_plan(table, (37, 24, 24), "none")
    assert plan.geometry == "b_t37"


# --- real Miowtion below -----------------------------------------------------

needs_miowtion = pytest.mark.skipif(
    importlib.util.find_spec("miowtion") is None, reason="miowtion is not installed"
)


@pytest.fixture
def small_case():
    from miowtion.h3 import geometry as h3_geometry
    from miowtion.h3 import layout as h3_layout
    from miowtion.veda import plan as veda_plan
    from miowtion.veda import predictor as veda_predictor
    from miowtion.veda import tiling

    geometry = h3_geometry.Geometry("16:9", 512, 256, 39, 12, 16, 32, 20)
    layout = h3_layout.pack(torch.ones(64, dtype=torch.long), geometry)
    plan = veda_plan.TilePlan(
        geometry.name,
        geometry.video_grid,
        [tiling.TileShape(4, 4, 8), tiling.TileShape(2, 8, 8)],
        [[0, 1, 1, 0, 1, 0, 0, 1], [1, 1, 0, 0, 0, 0, 1, 1]],
    )
    torch.manual_seed(0)
    predictor = veda_predictor.TileScorePredictor(2, 8, 32)
    # A predictor at its N(0, 1e-4) init scores every tile alike; give the
    # top-k something to rank.
    with torch.no_grad():
        for layer in predictor.layers:
            layer.proj_q.normal_(std=0.1)
            layer.proj_k.normal_(std=0.1)
    gen = torch.Generator().manual_seed(1)
    q, k, v = (torch.randn(layout.seq_len, 8, 32, generator=gen) for _ in range(3))
    return SimpleNamespace(
        geometry=geometry, layout=layout, plan=plan, predictor=predictor, q=q, k=k, v=v
    )


def _full_student(case, keep_ratio=0.2):
    from miowtion.veda import attention as veda_attention
    from miowtion.veda import mask as veda_mask

    config = veda_attention.VedaConfig(target_budget=veda_mask.Budget(ratio=keep_ratio))
    clip = veda_attention.ClipTiling(case.layout, config, torch.device("cpu"))
    student = veda_attention.SparseStudent(
        clip, case.plan, case.predictor, allow_reference_kernel=True
    )
    student.use_fa4 = False  # CPU tensors: the token-level reference kernel
    return student


def _local_student(case, rng, keep_ratio=0.2):
    from miowtion.veda import bundle as veda_bundle
    from miowtion.veda import plan as veda_plan

    full = veda_bundle.Bundle(
        predictor=case.predictor,
        plans=veda_plan.PlanTable([case.plan]),
        keep_ratio=keep_ratio,
        metadata={},
    )
    local = veda_runtime.slice_bundle(full, rng, torch.device("cpu"))
    plan, reason = find_plan(local.plans, case.layout.target.grid, "none")
    assert reason == "exact" and plan.provenance["sliced_heads"] == [
        rng.start,
        rng.stop,
    ]
    student = veda_runtime.make_student(
        local,
        plan,
        video_start=case.layout.target.start,
        grid=case.layout.target.grid,
        used=case.layout.used,
        seq_len=case.layout.seq_len,
        keep_ratio=keep_ratio,
        collect_mib=256,
        device=torch.device("cpu"),
        allow_reference_kernel=True,
    )
    student.use_fa4 = False
    return student, local


@needs_miowtion
@pytest.mark.parametrize("rng", [HeadRange(4, 8), HeadRange(0, 4), HeadRange(2, 5)])
@pytest.mark.parametrize("layer", [0, 1])
def test_sliced_bundle_matches_the_full_student_bitwise(small_case, rng, layer):
    case = small_case
    full = _full_student(case)(case.q, case.k, case.v, layer)
    student, local = _local_student(case, rng)
    assert local.num_heads == 8 and local.num_layers == 2 and local.head_dim == 32
    assert local.predictor.layers[0].proj_q.shape[0] == len(rng)
    sl = slice(rng.start, rng.stop)
    out = student(case.q[:, sl], case.k[:, sl], case.v[:, sl], layer)
    # The CPU reference kernel batches its matmuls over the heads of a group;
    # a group of one head takes a different BLAS path and can differ by an
    # fp32 ulp. Every group of two or more heads is bit-identical, which is
    # what the per-head GPU kernel gives in all cases (see the manual smoke).
    plan = student.plan
    singleton = any(
        g.heads.numel() == 1 for g in plan.head_groups(layer, torch.device("cpu"))
    )
    if singleton:
        torch.testing.assert_close(out, full[:, sl], rtol=0.0, atol=1e-6)
    else:
        assert torch.equal(out, full[:, sl])
    assert torch.equal(
        out[case.layout.used :], torch.zeros_like(out[case.layout.used :])
    )


@needs_miowtion
def test_whole_range_keeps_the_bundle_as_is(small_case):
    from miowtion.veda import bundle as veda_bundle
    from miowtion.veda import plan as veda_plan

    case = small_case
    full = veda_bundle.Bundle(case.predictor, veda_plan.PlanTable([case.plan]), 0.1, {})
    local = veda_runtime.slice_bundle(full, HeadRange(0, 8), torch.device("cpu"))
    assert local.predictor is case.predictor and local.plans is full.plans
    with pytest.raises(ValueError):
        veda_runtime.slice_bundle(full, HeadRange(4, 9), torch.device("cpu"))


@needs_miowtion
def test_packed_layout_tiles_like_miowtion_pack(small_case):
    from miowtion.veda import attention as veda_attention
    from miowtion.veda import mask as veda_mask

    case = small_case
    config = veda_attention.VedaConfig(target_budget=veda_mask.Budget(ratio=0.2))
    ours = veda_runtime.packed_layout(
        case.layout.target.start,
        case.layout.target.grid,
        case.layout.used,
        case.layout.seq_len,
    )
    cpu = torch.device("cpu")
    for shape in case.plan.shapes:
        mine = veda_attention.ClipTiling(ours, config, cpu).get(shape)
        theirs = veda_attention.ClipTiling(case.layout, config, cpu).get(shape)
        assert torch.equal(mine.perm, theirs.perm)
        assert torch.equal(mine.valid_count, theirs.valid_count)
        assert (mine.n_video_tiles, mine.n_global_tiles) == (
            theirs.n_video_tiles,
            theirs.n_global_tiles,
        )
    with pytest.raises(ValueError):
        veda_runtime.packed_layout(
            case.layout.used - 10,
            case.layout.target.grid,
            case.layout.used,
            case.layout.seq_len,
        )


@needs_miowtion
def test_fp8_bundle_round_trip_slices_the_dequantised_weights(small_case, tmp_path):
    from miowtion.veda import bundle as veda_bundle
    from miowtion.veda import plan as veda_plan
    from miowtion.veda import predictor as veda_predictor

    case = small_case
    torch.manual_seed(2)
    predictor = veda_predictor.TileScorePredictor(2, 8, 128)
    path = str(tmp_path / "bundle.safetensors")
    veda_bundle.save(
        path,
        predictor.state_dict(),
        veda_plan.PlanTable([case.plan]),
        num_layers=2,
        num_heads=8,
        head_dim=128,
        keep_ratio=0.1,
        source="test",
        source_weights="live",
        step=0,
        dtype=torch.float8_e4m3fn,
    )
    local = veda_runtime.load_local_bundle(path, torch.device("cpu"), HeadRange(4, 8))
    reference = veda_bundle.load(path).predictor
    assert local.keep_ratio == 0.1 and local.metadata["dtype"] == "float8_e4m3fn"
    for mine, theirs in zip(local.predictor.parameters(), reference.parameters()):
        assert mine.dtype == torch.bfloat16
        assert torch.equal(mine, theirs[4:8])
    assert all(
        len(row) == 4 for row in local.plans.plans[case.geometry.name].head_shape
    )


@needs_miowtion
def test_find_plan_select_follows_miowtion_nearest_latent_t():
    from miowtion.h3 import geometry as h3_geometry
    from miowtion.veda import plan as veda_plan
    from miowtion.veda import tiling

    plans = [
        veda_plan.TilePlan.uniform(
            h3_geometry.geometry_from_latent_t("16:9", t),
            tiling.TileShape(4, 4, 8),
            2,
            8,
        )
        for t in (37, 72)
    ]
    table = veda_plan.PlanTable(plans + [plans[0].mirrored()])
    assert find_plan(table, (37, 24, 42), "none")[0].geometry == "16x9_t37"
    assert find_plan(table, (37, 42, 24), "none")[0].geometry == "9x16_t37"
    with pytest.raises(ValueError):
        find_plan(table, (52, 24, 42), "none")
    plan, reason = find_plan(table, (52, 24, 42), "select")
    assert plan.geometry == "16x9_t37" and reason == "select(16x9_t52)"
    with pytest.raises(ValueError):
        find_plan(table, (52, 30, 30), "select")


@needs_miowtion
def test_check_contract_passes_on_the_pinned_api():
    veda_runtime.check_contract()
