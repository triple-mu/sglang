# SPDX-License-Identifier: Apache-2.0
"""Miowtion glue for the Veda attention backend; the only module importing it.

Upstream Miowtion knows no tensor or sequence parallelism: ``SparseStudent``
scores and attends every model head of a layer, and the same head index
selects the q / k / v head axis and the predictor projection. Every stage of
that path is per head (tile gather, pooled features, projection, top-k,
block-sparse kernel, scatter), so a rank owning the model heads
``[start, stop)`` can run the unchanged student on a bundle sliced to those
heads: predictor parameters along dim 0 and the ``head_shape`` row of every
plan. The result is bit-identical to running all heads at once (Miowtion's
own ``test_sparse_student_head_chunks_are_exact`` asserts the same
property).

Pinned to one Miowtion commit; ``check_contract`` refuses to run against an
API that drifted from it.
"""

from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Mapping
from typing import Any

import msgspec
import torch

MIOWTION_COMMIT = "92c9748478fcba3884d56a56c66abf1bc1221909"
MIOWTION_INSTALL = (
    "pip install --no-deps "
    f"'miowtion @ git+https://github.com/veda-sparse/Miowtion@{MIOWTION_COMMIT}'"
)
# Miowtion's SM8x / SM120 block-sparse patch hash-checks the installed FA4.
FA4_PIN = "4.0.0b32"
FA4_INSTALL = f"pip install --no-deps 'flash-attn-4=={FA4_PIN}'"
# VedaConfig.collect_bytes default (256 MiB); the bound on one tile-ordered
# q / k / v / out copy, which decides how many heads one launch processes.
DEFAULT_COLLECT_MIB = 256
PLAN_FALLBACKS = ("none", "select")


class HeadRange(msgspec.Struct, frozen=True):
    """Model heads ``[start, stop)`` owned by this process."""

    start: int
    stop: int

    def __len__(self) -> int:
        return self.stop - self.start


def head_range(
    heads_per_tp: int, local_heads: int, tp_rank: int, ulysses_rank: int
) -> HeadRange:
    """Model heads of one rank: TP splits heads into contiguous blocks,
    the Ulysses all-to-all splits each block again (``usp.py``)."""
    if local_heads <= 0 or heads_per_tp % local_heads:
        raise ValueError(
            f"{local_heads} local heads do not divide the {heads_per_tp} heads "
            "of this tensor-parallel rank"
        )
    start = tp_rank * heads_per_tp + ulysses_rank * local_heads
    return HeadRange(start, start + local_heads)


def slice_head_shape(head_shape: list[list[int]], rng: HeadRange) -> list[list[int]]:
    return [list(row[rng.start : rng.stop]) for row in head_shape]


def slice_state_dict(
    state: Mapping[str, torch.Tensor], rng: HeadRange
) -> dict[str, torch.Tensor]:
    """Predictor tensors (``layers.{i}.proj_q|proj_k``) cut along the head axis."""
    out = {}
    for key, value in state.items():
        if value.shape[0] < rng.stop:
            raise ValueError(
                f"{key} has {value.shape[0]} heads; this rank needs [{rng.start}, {rng.stop})"
            )
        out[key] = value[rng.start : rng.stop]
    return out


def find_plan(table: Any, grid: tuple[int, int, int], fallback: str) -> tuple[Any, str]:
    """The tile plan of a token grid and how it was picked.

    Exact grid matches only, unless ``fallback == "select"``, which applies
    Miowtion's own ``PlanTable.select`` rule (same aspect ratio with the
    nearest latent_t, else the transposed plan of the mirrored aspect).
    """
    grid = tuple(int(v) for v in grid)
    exact = [p for p in table.plans.values() if tuple(p.grid) == grid]
    if exact:
        # Prefer a searched plan over a mirrored copy of another aspect.
        exact.sort(key=lambda p: ("mirrored_from" in p.provenance, p.geometry))
        return exact[0], "exact"
    if fallback == "select":
        from miowtion.h3 import geometry as h3_geometry

        for key in sorted({p.geometry.split("_t")[0] for p in table.plans.values()}):
            try:
                geometry = h3_geometry.geometry_from_latent_t(
                    key.replace("x", ":"), grid[0]
                )
            except ValueError:
                continue
            if tuple(geometry.video_grid)[1:] == grid[1:]:
                return table.select(geometry), f"select({geometry.name})"
    known = sorted((p.geometry, tuple(p.grid)) for p in table.plans.values())
    raise ValueError(
        f"no Veda tile plan for token grid (T, H, W) = {grid}; the bundle has {known}"
        + (
            ""
            if fallback == "select"
            else "; veda_plan_fallback=select allows the nearest plan"
        )
    )


class LocalBundle(msgspec.Struct):
    """A bundle reduced to this rank's heads."""

    predictor: Any
    plans: Any
    keep_ratio: float
    metadata: dict[str, str]
    heads: HeadRange
    num_heads: int
    num_layers: int
    head_dim: int


def slice_plan(plan: Any, rng: HeadRange) -> Any:
    return dataclasses.replace(
        plan,
        head_shape=slice_head_shape(head_shape=plan.head_shape, rng=rng),
        provenance={**plan.provenance, "sliced_heads": [rng.start, rng.stop]},
    )


def slice_bundle(full: Any, rng: HeadRange, device: torch.device) -> LocalBundle:
    """``full`` (``miowtion.veda.bundle.Bundle``) cut to ``rng``, on ``device``."""
    from miowtion.veda import plan as veda_plan
    from miowtion.veda import predictor as veda_predictor

    first = full.predictor.layers[0]
    num_heads, head_dim = int(first.proj_q.shape[0]), int(first.head_dim)
    num_layers = len(full.predictor.layers)
    if not 0 <= rng.start < rng.stop <= num_heads:
        raise ValueError(f"bundle has {num_heads} heads; this rank needs {rng}")
    for plan in full.plans.plans.values():
        if plan.num_layers != num_layers or any(
            len(row) != num_heads for row in plan.head_shape
        ):
            raise ValueError(
                f"plan {plan.geometry} does not describe {num_layers} layers x {num_heads} heads"
            )
    if (rng.start, rng.stop) == (0, num_heads):
        predictor, plans = full.predictor, full.plans
    else:
        predictor = veda_predictor.TileScorePredictor(num_layers, len(rng), head_dim)
        predictor = predictor.to(first.proj_q.dtype)
        predictor.load_state_dict(
            slice_state_dict(state=full.predictor.state_dict(), rng=rng), strict=True
        )
        plans = veda_plan.PlanTable(
            [slice_plan(plan=p, rng=rng) for p in full.plans.plans.values()]
        )
    predictor = predictor.to(device).eval().requires_grad_(False)
    return LocalBundle(
        predictor=predictor,
        plans=plans,
        keep_ratio=float(full.keep_ratio),
        metadata=dict(full.metadata),
        heads=rng,
        num_heads=num_heads,
        num_layers=num_layers,
        head_dim=head_dim,
    )


def load_local_bundle(path: str, device: torch.device, rng: HeadRange) -> LocalBundle:
    """Loads a bundle (bf16 or fp8 storage) and keeps this rank's heads only."""
    from miowtion.veda import bundle as veda_bundle

    return slice_bundle(full=veda_bundle.load(str(path), "cpu"), rng=rng, device=device)


def packed_layout(
    video_start: int, grid: tuple[int, int, int], used: int, seq_len: int
):
    """Miowtion's ``PackedLayout`` for sglang's packed sequence.

    ``ClipTiling`` reads the target span, ``used`` and ``seq_len`` only; all
    other real rows (text, audio, keyframes) stay global, as in the layout
    the predictor was trained on. Built by keyword so a field added upstream
    fails here instead of inside the tiling.
    """
    from miowtion.h3 import layout as h3_layout

    target = h3_layout.VideoSpan(
        int(video_start), tuple(int(v) for v in grid), "target"
    )
    if not 0 <= target.start <= target.stop <= used <= seq_len:
        raise ValueError(
            f"video rows [{target.start}, {target.stop}) do not fit "
            f"used={used} <= seq_len={seq_len}"
        )
    rows = torch.arange(target.start, target.stop)
    return h3_layout.PackedLayout(
        seq_len=int(seq_len),
        used=int(used),
        text_len=0,
        img_pos=rows,
        audio_pos=torch.empty(0, dtype=torch.long),
        update_mask=torch.ones(rows.numel(), dtype=torch.bool),
        audio_update_mask=torch.empty(0, dtype=torch.bool),
        position_ids=torch.empty(0, 3, dtype=torch.float64),
        token_tags=torch.empty(0, dtype=torch.long),
        cu_seqlens=torch.tensor([0, used, seq_len], dtype=torch.int32),
        spans=(target,),
        visual_cond_shapes=(),
        audio_cond_lengths=(),
    )


def make_student(
    local: LocalBundle,
    plan: Any,
    *,
    video_start: int,
    grid: tuple[int, int, int],
    used: int,
    seq_len: int,
    keep_ratio: float,
    collect_mib: int,
    device: torch.device,
    allow_reference_kernel: bool = False,
):
    """Upstream ``SparseStudent`` of one packed layout on this rank's heads.

    ``dense_layers`` stays empty on purpose: dense steps and layers are
    routed by the backend to sglang's own SDPA path, not to Miowtion's.
    """
    from miowtion.veda import attention as veda_attention
    from miowtion.veda import mask as veda_mask

    config = veda_attention.VedaConfig(
        target_budget=veda_mask.Budget(ratio=float(keep_ratio)),
        collect_bytes=int(collect_mib) * 2**20,
    )
    clip = veda_attention.ClipTiling(
        packed_layout(video_start=video_start, grid=grid, used=used, seq_len=seq_len),
        config,
        device,
    )
    return veda_attention.SparseStudent(
        clip, plan, local.predictor, allow_reference_kernel
    )


def check_contract() -> None:
    """Raises ImportError unless Miowtion exposes the API this module was written against."""
    try:
        from miowtion.h3 import layout as h3_layout
        from miowtion.kernels import fa4
        from miowtion.veda import attention, bundle, plan, predictor
    except ImportError as e:
        raise ImportError(
            f"Veda attention needs Miowtion @ {MIOWTION_COMMIT[:10]} "
            f"({MIOWTION_INSTALL}) and flash-attn-4=={FA4_PIN} ({FA4_INSTALL}): {e}"
        ) from e
    signatures = {
        "SparseStudent.__init__": (
            attention.SparseStudent.__init__,
            ("self", "clip", "plan", "predictor", "allow_reference_kernel"),
        ),
        "SparseStudent.__call__": (
            attention.SparseStudent.__call__,
            ("self", "q", "k", "v", "layer_index"),
        ),
        "ClipTiling.__init__": (
            attention.ClipTiling.__init__,
            ("self", "layout", "config", "device", "condition_spans"),
        ),
        "PlanTable.select": (plan.PlanTable.select, ("self", "geometry")),
        "TilePlan.head_groups": (
            plan.TilePlan.head_groups,
            ("self", "layer", "device"),
        ),
        "LayerPredictor.forward": (
            predictor.LayerPredictor.forward,
            ("self", "feats_q", "feats_k", "heads"),
        ),
        "bundle.load": (bundle.load, ("path", "device")),
        "fa4.available": (fa4.available, ("device",)),
    }
    for name, (fn, params) in signatures.items():
        found = tuple(inspect.signature(fn).parameters)
        if found != params:
            raise ImportError(
                f"Miowtion drifted from {MIOWTION_COMMIT[:10]}: {name}{found} != {params}"
            )
    fields = {
        "VedaConfig": (
            attention.VedaConfig,
            {"target_budget", "collect_bytes", "dense_layers", "tile_conditions"},
        ),
        "TilePlan": (
            plan.TilePlan,
            {"geometry", "grid", "shapes", "head_shape", "provenance"},
        ),
        "PackedLayout": (
            h3_layout.PackedLayout,
            {
                "seq_len",
                "used",
                "text_len",
                "img_pos",
                "audio_pos",
                "update_mask",
                "audio_update_mask",
                "position_ids",
                "token_tags",
                "cu_seqlens",
                "spans",
                "visual_cond_shapes",
                "audio_cond_lengths",
            },
        ),
    }
    for name, (cls, expected) in fields.items():
        found = {f.name for f in dataclasses.fields(cls)}
        if not expected <= found:
            raise ImportError(
                f"Miowtion drifted from {MIOWTION_COMMIT[:10]}: {name} lacks {expected - found}"
            )
    keys = set(predictor.TileScorePredictor(1, 1, 8).state_dict())
    if keys != {"layers.0.proj_q", "layers.0.proj_k"}:
        raise ImportError(
            f"Miowtion drifted from {MIOWTION_COMMIT[:10]}: predictor state dict {sorted(keys)}"
        )


def fa4_status(device: torch.device) -> tuple[bool, str | None]:
    """(block sparsity available on ``device``, Miowtion's patch error if any).

    The first call imports FA4 and, on SM8x / SM120, installs Miowtion's
    vendored patch, which must happen before anything imports ``flash_attn.cute``.
    """
    from miowtion.kernels import fa4

    return bool(fa4.available(device)), fa4._patch_error
