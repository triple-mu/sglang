# SPDX-License-Identifier: Apache-2.0
"""HyperFlow (Video Rebirth) sampling contract for MiniMax-H3.

HyperFlow is an 8-step LoRA. `tools/build_minimax_h3_hyperflow_weights.py` turns the
weights file into a bundle: the block LoRA stays a LoRA and is served unmerged
(``--lora-path <bundle>/hyperflow_lora.safetensors --lora-merge-mode dynamic``; its
deltas are below bf16 resolution, so a merged copy keeps about half of them), the fp32
time embedder is merged exactly, and two things that are not weights live here:

* the fixed, non-uniform 9-point sigma grid (shifted per modality like the official
  linspace grid), applied by the timestep preparation stage;
* the two-time conditioning: every row carries the step's endpoint ``r`` next to its
  timestep ``t`` (generated rows ``r = 1 - sigma[i + 1]``, pinned condition rows
  ``r = t``), and the time embedding becomes ``emb_t + gate * (emb_r - emb_t)`` with
  ``emb_r`` from a second time embedder (the base one plus its own LoRA).

Enabled by ``SGLANG_DIFFUSION_MINIMAX_H3_HYPERFLOW=<bundle dir>`` holding
``hyperflow_config.json`` and ``hyperflow_endpoint_embedder.safetensors``.
"""

from __future__ import annotations

import json
from pathlib import Path

import msgspec
import torch

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

CONFIG_NAME = "hyperflow_config.json"
ENDPOINT_EMBEDDER_NAME = "hyperflow_endpoint_embedder.safetensors"
LORA_NAME = "hyperflow_lora.safetensors"
_EMBEDDER_TENSORS = ("proj_in.weight", "proj_in.bias", "proj_out.weight", "proj_out.bias")


class MiniMaxH3HyperFlowConfig(msgspec.Struct, frozen=True):
    sigmas: tuple[float, ...]
    video_shift: float
    audio_shift: float
    gate: float
    version: str
    path: str

    @property
    def num_steps(self) -> int:
        return len(self.sigmas) - 1


def hyperflow_dir(path: str | Path) -> Path:
    path = Path(path)
    return path.parent if path.is_file() else path


def load_minimax_h3_hyperflow_config(path: str | Path) -> MiniMaxH3HyperFlowConfig:
    """Read ``hyperflow_config.json`` from a directory (or the file itself)."""
    directory = hyperflow_dir(path)
    raw = json.loads((directory / CONFIG_NAME).read_text())
    sigmas = tuple(float(s) for s in raw["sigmas"])
    if len(sigmas) < 2 or any(b >= a for a, b in zip(sigmas, sigmas[1:])):
        raise ValueError(f"HyperFlow sigma grid must be strictly decreasing, got {sigmas}")
    if sigmas[0] > 1.0 or sigmas[-1] != 0.0:
        raise ValueError(f"HyperFlow sigma grid must start at or below 1.0 and end at 0.0, got {sigmas}")
    return MiniMaxH3HyperFlowConfig(
        sigmas=sigmas,
        video_shift=float(raw["video_shift"]),
        audio_shift=float(raw["audio_shift"]),
        gate=float(raw["gate"]),
        version=str(raw.get("hyperflow_version", "?")),
        path=str(directory),
    )


def hyperflow_shift_sigmas(sigmas: tuple[float, ...], shift: float) -> list[float]:
    """The scheduler's exponential shift ``s * sigma / (1 + (s - 1) * sigma)`` on a raw grid."""
    if shift <= 0:
        raise ValueError(f"shift must be positive, got {shift}")
    base = torch.tensor(sigmas, dtype=torch.float32)
    shifted = shift * base / (1.0 + (shift - 1.0) * base)
    return [float(v) for v in shifted.tolist()]


def dedup_timestep_pairs(
    timesteps: list[float], endpoints: list[float]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unique ``(t, r)`` pairs of one step as two fp32 vectors plus slot -> unique index.

    Rows that share a timestep but not an endpoint (a pinned condition row at
    ``t == t_video`` next to generated rows heading for ``r_video``) stay distinct,
    which is what the single-timestep dedup would collapse.
    """
    pairs = torch.tensor(list(zip(timesteps, endpoints)), dtype=torch.float32)
    unique, slot_to_unique = torch.unique(pairs, dim=0, sorted=True, return_inverse=True)
    return unique[:, 0].contiguous(), unique[:, 1].contiguous(), slot_to_unique


def install_minimax_h3_hyperflow(model) -> MiniMaxH3HyperFlowConfig | None:
    """Attach the endpoint time embedder and the sampling contract to a loaded DiT."""
    path = envs.SGLANG_DIFFUSION_MINIMAX_H3_HYPERFLOW
    if not path:
        return None
    if envs.SGLANG_DIFFUSION_MINIMAX_H3_PDD_HEADS:
        raise ValueError(
            "SGLANG_DIFFUSION_MINIMAX_H3_HYPERFLOW and SGLANG_DIFFUSION_MINIMAX_H3_PDD_HEADS "
            "select two different distilled samplers; set only one"
        )
    if getattr(model, "time_embedder", None) is None:
        raise ValueError("MiniMax-H3 HyperFlow needs the standard time embedder (curve-AdaLN checkpoints are unsupported)")
    config = load_minimax_h3_hyperflow_config(path)

    from safetensors.torch import load_file

    from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import MiniMaxH3TimeEmbedder

    weights = load_file(str(hyperflow_dir(path) / ENDPOINT_EMBEDDER_NAME))
    if set(weights) != set(_EMBEDDER_TENSORS):
        raise ValueError(f"{ENDPOINT_EMBEDDER_NAME} must hold {_EMBEDDER_TENSORS}, got {sorted(weights)}")
    embedder = MiniMaxH3TimeEmbedder(model.arch, prefix="hyperflow_endpoint_embedder")
    params = dict(embedder.named_parameters())
    for name in _EMBEDDER_TENSORS:
        param = params[name]
        loader = getattr(param, "weight_loader", None)
        if loader is not None:
            loader(param, weights[name])  # TP-aware, like the checkpoint path
        else:
            param.data.copy_(weights[name])
    embedder.to(model.time_embedder.proj_in.weight.device)
    model.hyperflow_endpoint_embedder = embedder
    model.hyperflow = config
    logger.info(
        "MiniMax-H3 HyperFlow %s: %d-step grid %s, shifts video %.3g / audio %.3g, gate %.3g, "
        "endpoint time embedder loaded from %s; the block LoRA must be served with "
        "--lora-path %s/%s --lora-merge-mode dynamic",
        config.version,
        config.num_steps,
        [round(s, 4) for s in config.sigmas],
        config.video_shift,
        config.audio_shift,
        config.gate,
        config.path,
        config.path,
        LORA_NAME,
    )
    return config


def warn_unless_dynamic_lora(model: torch.nn.Module) -> None:
    """Say once when the HyperFlow block LoRA is missing or was merged into bf16 weights."""
    from sglang.multimodal_gen.runtime.layers.lora.linear import BaseLayerWithLoRA

    active = [
        layer
        for layer in model.modules()
        if isinstance(layer, BaseLayerWithLoRA) and not layer.disable_lora
    ]
    if not active:
        logger.warning_once(
            "MiniMax-H3 HyperFlow: no LoRA layer is active on the DiT, so the 8-step grid "
            "runs on base block weights; serve --lora-path <bundle>/%s --lora-merge-mode dynamic",
            LORA_NAME,
        )
    elif any(layer.merged for layer in active):
        logger.warning_once(
            "MiniMax-H3 HyperFlow: the block LoRA was merged into bf16 weights, which keeps "
            "about half of its deltas; use --lora-merge-mode dynamic"
        )
