#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build the serving bundle for Video Rebirth's HyperFlow 8-step LoRA on MiniMax-H3.

    python3 -m sglang.multimodal_gen.tools.build_minimax_h3_hyperflow_weights \
        <base transformer dir> <minimax_h3_hyperflow_8step_v1.0.safetensors> <out dir>

HyperFlow (huggingface.co/videorebirth/hyperflow) is a PEFT LoRA on the DiT blocks,
the token refiner and the time embedder, plus a second "endpoint" time embedder (the
base one with its own LoRA) and a fixed 9-point sigma grid, all in one safetensors
file with a self-describing header. This tool writes

    <out dir>/transformer/                              base shards (symlinks); only the
                                                        shard holding the fp32 time embedder
                                                        is rewritten with its LoRA merged
    <out dir>/hyperflow_lora.safetensors                the block + token-refiner LoRA, to
                                                        serve unmerged
    <out dir>/hyperflow_endpoint_embedder.safetensors   endpoint time embedder: base + its
                                                        delta, fp32
    <out dir>/hyperflow_config.json                     sigma grid, shifts, gate and versions

The block deltas are not merged on purpose: they are 1e-4 to 1e-3 of the bf16 base
weights, so 86-98% of their elements fall below half a bf16 ulp and a merged copy
keeps about half of the adapter (measured 2026-10-07, step-0 LoRA effect 0.14 vs
0.29 unmerged). The fp32 time embedders merge exactly.

Serve with
    --transformer-weights-path <out dir>/transformer
    --lora-path <out dir>/hyperflow_lora.safetensors --lora-merge-mode dynamic --lora-alpha <alpha>
    SGLANG_DIFFUSION_MINIMAX_H3_HYPERFLOW=<out dir>   --num-inference-steps 9
(H3 counts sigma grid points). The same LoRA file fits `transformer_ref`, so ref2va
needs one more run of this tool against that partition.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from sglang.multimodal_gen.runtime.models.dits.minimax_h3_hyperflow import (
    CONFIG_NAME,
    ENDPOINT_EMBEDDER_NAME,
    LORA_NAME,
)

_TIME_EMBEDDER = {
    "transformer.time_embedder.linear_1": "time_embedder.proj_in.weight",
    "transformer.time_embedder.linear_2": "time_embedder.proj_out.weight",
}
_ENDPOINT_EMBEDDER = {
    "transformer.endpoint_time_embedder.linear_1": "proj_in.weight",
    "transformer.endpoint_time_embedder.linear_2": "proj_out.weight",
}
_EMBEDDER_TENSORS = ("proj_in.weight", "proj_in.bias", "proj_out.weight", "proj_out.bias")
_SUFFIXES = (".lora_A.weight", ".lora_B.weight")


def _lora_module(key: str) -> tuple[str, str]:
    for suffix in _SUFFIXES:
        if key.endswith(suffix):
            return key[: -len(suffix)], suffix[6]  # "A" or "B"
    raise SystemExit(f"unexpected key {key!r} in a HyperFlow weights file")


def _embedder_deltas(
    lora: dict[str, torch.Tensor], mapping: dict[str, str], scale: float
) -> dict[str, torch.Tensor]:
    deltas = {}
    for module, target in mapping.items():
        a = lora.get(f"{module}.lora_A.weight")
        b = lora.get(f"{module}.lora_B.weight")
        if a is None or b is None:
            raise SystemExit(f"{module} LoRA pair is incomplete")
        deltas[target] = (b.float() @ a.float()) * scale
    return deltas


def _shard_of(base_dir: Path, key: str) -> Path:
    for index in base_dir.glob("*.index.json"):
        weight_map = json.loads(index.read_text())["weight_map"]
        if key in weight_map:
            return base_dir / weight_map[key]
    for shard in sorted(base_dir.glob("*.safetensors")):
        with safe_open(str(shard), "pt") as f:
            if key in f.keys():
                return shard
    raise SystemExit(f"{key} not found in {base_dir}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("base_dir", type=Path)
    ap.add_argument("lora_path", type=Path)
    ap.add_argument("out_dir", type=Path)
    args = ap.parse_args(argv)

    with safe_open(str(args.lora_path), "pt") as f:
        meta = f.metadata() or {}
        lora = {k: f.get_tensor(k) for k in f.keys()}
    if meta.get("hyperflow", "").lower() != "true":
        raise SystemExit("not a HyperFlow weights file: the header lacks hyperflow=true")
    rank = int(meta["lora_rank"])
    alpha = float(meta["lora_alpha"])
    scale = alpha / rank
    config = {
        "hyperflow_version": meta["hyperflow_version"],
        "sigmas": json.loads(meta["hyperflow_sigmas"]),
        "video_shift": float(meta["hyperflow_video_shift"]),
        "audio_shift": float(meta["hyperflow_audio_shift"]),
        "gate": float(meta["hyperflow_gate"]),
        "lora_rank": rank,
        "lora_alpha": alpha,
        "base_model": meta.get("base_model"),
        "base_model_revision": meta.get("base_model_revision"),
        "tasks": json.loads(meta.get("tasks", "[]")),
        "source_file": args.lora_path.name,
    }
    print(
        f"HyperFlow {config['hyperflow_version']}: rank={rank} alpha={alpha} "
        f"scale={scale} gate={config['gate']} steps={len(config['sigmas']) - 1}"
    )
    for key in lora:
        _lora_module(key)  # every key must be a PEFT A/B matrix

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    # 1. The block + refiner LoRA, served unmerged; both time embedders leave.
    embedders = set(_TIME_EMBEDDER) | set(_ENDPOINT_EMBEDDER)
    blocks = {k: v for k, v in lora.items() if _lora_module(k)[0] not in embedders}
    save_file(blocks, str(out / LORA_NAME), metadata=meta)
    print(f"{len(blocks)} block/refiner LoRA tensors -> {out / LORA_NAME}")

    # 2. The transformer: symlinks, plus one rewritten shard with the fp32 time
    #    embedder merged (its deltas are exact in fp32).
    time_deltas = _embedder_deltas(lora, _TIME_EMBEDDER, scale)
    shard = _shard_of(args.base_dir, "time_embedder.proj_in.weight")
    transformer_out = out / "transformer"
    transformer_out.mkdir(exist_ok=True)
    for entry in sorted(args.base_dir.iterdir()):
        target = transformer_out / entry.name
        if target.is_symlink() or target.exists():
            target.unlink()
        if entry == shard or not entry.name.endswith(".safetensors"):
            continue
        os.symlink(entry.resolve(), target)
    for extra in args.base_dir.iterdir():
        if not extra.name.endswith(".safetensors") and extra.is_file():
            shutil.copy(extra, transformer_out / extra.name)
    tensors = load_file(str(shard))
    base_embedder = {
        key[len("time_embedder.") :]: value.clone()
        for key, value in tensors.items()
        if key.startswith("time_embedder.")
    }
    if set(base_embedder) != set(_EMBEDDER_TENSORS):
        raise SystemExit(
            f"time embedder tensors {sorted(base_embedder)} != {sorted(_EMBEDDER_TENSORS)}"
        )
    for key, delta in time_deltas.items():
        base = tensors[key]
        if base.dtype != torch.float32:
            raise SystemExit(f"{key} is {base.dtype}; the time embedder must be fp32 to merge")
        tensors[key] = base + delta
    save_file(tensors, str(transformer_out / shard.name), metadata={"format": "pt"})
    print(f"time embedder merged into {transformer_out / shard.name}; other shards symlinked")

    # 3. The endpoint embedder: base time embedder + its own deltas (no bias deltas).
    endpoint_deltas = _embedder_deltas(lora, _ENDPOINT_EMBEDDER, scale)
    endpoint = {
        name: tensor + endpoint_deltas[name] if name in endpoint_deltas else tensor
        for name, tensor in base_embedder.items()
    }
    save_file(endpoint, str(out / ENDPOINT_EMBEDDER_NAME), metadata={"format": "pt"})
    (out / CONFIG_NAME).write_text(json.dumps(config, indent=2) + "\n")
    print(f"endpoint time embedder -> {out / ENDPOINT_EMBEDDER_NAME}")
    print(f"sampling contract      -> {out / CONFIG_NAME}")
    print(
        "serve: --transformer-weights-path "
        f"{transformer_out} --lora-path {out / LORA_NAME} --lora-merge-mode dynamic "
        f"--lora-alpha {int(alpha)} SGLANG_DIFFUSION_MINIMAX_H3_HYPERFLOW={out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
