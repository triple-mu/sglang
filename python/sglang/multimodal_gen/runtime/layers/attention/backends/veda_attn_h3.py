# SPDX-License-Identifier: Apache-2.0
"""Veda block-sparse attention for the MiniMax-H3 DiT.

Veda (Miowtion, https://github.com/veda-sparse/Miowtion) permutes the video
tokens of the packed sequence into 3D tiles of 128 tokens, one tile shape per
head chosen by a searched plan, scores every (query tile, key tile) pair with
a trained per-head predictor, keeps the top blocks of each query tile and runs
a FlashAttention-4 CuTe block-sparse kernel on them. Text, audio and condition
rows stay global (dense both ways). The predictor weights and the tile plans
travel together in one safetensors bundle so they cannot be mispaired.

Contract with the H3 attention core: ``forward_varlen`` receives the whole
packed sequence ``[seq_len, H_local, 128]`` after the Ulysses all-to-all,
bf16, post QK-norm and RoPE, with ``cu_seqlens = (0, used, seq_len)`` and
``max_seqlen = used``. Rows ``[used, seq_len)`` are trailing padding; their
output is zeroed. Whatever the bundle does not serve (the token refiner, dense
warm-up steps, dense layers, calls outside the denoising loop) goes to the
Torch SDPA backend, so the fallback is numerically the baseline path.

Head parallelism: each rank holds ``H_local`` consecutive model heads,
``tp_rank * heads_per_tp_rank + ulysses_rank * H_local + j``; the plan and the
predictor are indexed by model head.

``--attention-backend-config`` keys (``k=v`` or JSON):

* ``veda_bundle=<path>``: predictor bundle written by Miowtion's
  ``scripts/export_predictor.py`` (required).
* ``veda_keep_ratio=<float>``: block budget; default is the bundle's own.
* ``veda_dense_first_n_steps=<n>``: denoising steps that stay dense
  (0-based loop steps; default 0).
* ``veda_dense_layers=a,b,...``: DiT blocks that stay dense (default none).
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Any

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import SDPAImpl
from sglang.multimodal_gen.runtime.managers.forward_context import (
    get_forward_context_or_none,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# DiT blocks are ``blocks.{i}.attn``; the token refiner (``token_refiner.blocks.*``)
# and anything else stay dense. Anchored so the refiner does not match.
_LAYER_RE = re.compile(r"^blocks\.(\d+)\.")
_HEAD_DIM = 128
# Students are cached per packed layout; a serving process sees a handful of
# geometries at most, so a small bound keeps stale tile layouts from piling up.
_MAX_CACHED_LAYOUTS = 8


class VedaAttentionBackend(AttentionBackend):
    accept_output_buffer = False

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [_HEAD_DIM]

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.VEDA_ATTN

    @staticmethod
    def get_impl_cls() -> type[VedaAttentionImpl]:
        return VedaAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type[VedaAttentionMetadata]:
        return VedaAttentionMetadata

    @staticmethod
    def get_builder_cls() -> type[VedaAttentionMetadataBuilder]:
        return VedaAttentionMetadataBuilder


@dataclass
class VedaAttentionMetadata(AttentionMetadata):
    """Request-static description of the packed sequence Veda tiles.

    Built once per request by the H3 denoising stage and published through
    the forward context; the loop only advances ``current_timestep``.
    """

    video_start: int = 0
    grid: tuple[int, int, int] = (0, 0, 0)
    used: int = 0
    seq_len: int = 0
    num_steps: int = 0


class VedaAttentionMetadataBuilder(AttentionMetadataBuilder):
    def __init__(self) -> None:
        pass

    def prepare(self) -> None:
        pass

    def build(self, **kwargs: dict[str, Any]) -> VedaAttentionMetadata:  # type: ignore[override]
        return VedaAttentionMetadata(**kwargs)


def _int_list(value: Any) -> list[int]:
    """``veda_dense_layers`` from ``k=v`` ("0,1"), JSON lists or a single int."""
    if value is None or value == "":
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, str):
        return [int(v) for v in re.split(r"[,\s]+", value.strip()) if v]
    return [int(v) for v in value]


class _VedaRuntime:
    """Bundle, config and per-layout students of one worker process.

    One process drives one device, so the singleton is keyed by device.
    """

    _instances: dict[torch.device, _VedaRuntime] = {}
    _lock = threading.Lock()

    @classmethod
    def get(cls, device: torch.device) -> _VedaRuntime:
        with cls._lock:
            runtime = cls._instances.get(device)
            if runtime is None:
                runtime = cls(device)
                cls._instances[device] = runtime
            return runtime

    def __init__(self, device: torch.device) -> None:
        from miowtion.veda import bundle as veda_bundle

        from sglang.multimodal_gen.runtime.server_args import get_global_server_args

        config = get_global_server_args().attention_backend_config or {}
        path = config.get("veda_bundle")
        if not path:
            raise ValueError(
                "Veda attention needs --attention-backend-config "
                "veda_bundle=<predictor bundle .safetensors>"
            )
        self.device = device
        self.loaded = veda_bundle.load(str(path), device)
        keep_ratio = config.get("veda_keep_ratio")
        self.keep_ratio = (
            float(keep_ratio)
            if keep_ratio is not None
            else float(self.loaded.keep_ratio)
        )
        if not 0.0 < self.keep_ratio <= 1.0:
            raise ValueError(
                f"veda_keep_ratio must be in (0, 1], got {self.keep_ratio}"
            )
        self.dense_first_n_steps = int(config.get("veda_dense_first_n_steps", 0))
        self.dense_layers = frozenset(_int_list(config.get("veda_dense_layers")))
        self._students: dict[tuple, Any] = {}
        self._announced_sparse = False
        logger.info(
            "Veda attention: bundle %s (source %s, step %s), keep ratio %.3f, "
            "dense first %d steps, dense layers %s, plans %s",
            path,
            self.loaded.metadata.get("source", "?"),
            self.loaded.metadata.get("step", "?"),
            self.keep_ratio,
            self.dense_first_n_steps,
            sorted(self.dense_layers) or "none",
            sorted(self.loaded.plans.plans),
        )

    def student(self, metadata: VedaAttentionMetadata):
        """The SparseStudent of one packed layout (tile layouts are cached)."""
        from miowtion.veda import attention as veda_attention
        from miowtion.veda import mask as veda_mask

        grid = tuple(int(v) for v in metadata.grid)
        key = (
            int(metadata.video_start),
            grid,
            int(metadata.used),
            int(metadata.seq_len),
        )
        student = self._students.get(key)
        if student is None:
            if len(self._students) >= _MAX_CACHED_LAYOUTS:
                self._students.clear()
            plan = self.loaded.plans.select_grid(grid)
            config = veda_attention.VedaConfig(
                target_budget=veda_mask.Budget(ratio=self.keep_ratio)
            )
            clip = veda_attention.ClipTiling.for_target(
                key[0], grid, key[2], key[3], config, self.device
            )
            student = veda_attention.SparseStudent(clip, plan, self.loaded.predictor)
            self._students[key] = student
            logger.info(
                "Veda attention: video rows [%d, %d) on grid %s of %d real / %d "
                "packed rows -> plan %s (%d shapes)",
                key[0],
                key[0] + grid[0] * grid[1] * grid[2],
                grid,
                key[2],
                key[3],
                plan.geometry,
                len(plan.shapes),
            )
        return student


class VedaAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        # ``num_heads`` is this tensor-parallel rank's head count; the Ulysses
        # all-to-all shrinks it further, so the runtime head count is q.shape[1].
        self.num_heads = num_heads
        self.head_size = head_size
        self.softmax_scale = softmax_scale
        self.prefix = prefix
        match = _LAYER_RE.match(prefix)
        self.layer_index = int(match.group(1)) if match else None
        # Only non-causal DiT blocks with 128-wide heads at the kernel's
        # 1/sqrt(D) scale are sparse. The token refiner, and the VAE / text
        # encoder attention when the backend is set globally, stay dense.
        self._sparse_capable = (
            self.layer_index is not None
            and head_size == _HEAD_DIM
            and not causal
            and abs(softmax_scale - head_size**-0.5) <= 1e-6
        )
        if self.layer_index is not None and not self._sparse_capable:
            raise ValueError(
                f"Veda attention serves non-causal DiT blocks with head_dim {_HEAD_DIM} at "
                f"the 1/sqrt(head_dim) scale; {prefix} has head_dim {head_size}, "
                f"causal={causal}, softmax_scale={softmax_scale}"
            )
        self._dense = SDPAImpl(
            num_heads=num_heads,
            head_size=head_size,
            causal=causal,
            softmax_scale=softmax_scale,
            num_kv_heads=num_kv_heads,
            prefix=prefix,
            **extra_impl_args,
        )
        self._model_heads: torch.Tensor | None = None
        self.calls = {"sparse": 0, "dense": 0}

    def _model_heads_for(self, local_heads: int, device: torch.device) -> torch.Tensor:
        """Model head index of every local head (see module docstring)."""
        if self._model_heads is None or self._model_heads.numel() != local_heads:
            from sglang.multimodal_gen.runtime.distributed.parallel_state import (
                get_tp_rank,
                get_ulysses_parallel_rank,
            )

            start = (
                get_tp_rank() * self.num_heads
                + get_ulysses_parallel_rank() * local_heads
            )
            self._model_heads = torch.arange(
                start, start + local_heads, device=device, dtype=torch.long
            )
        return self._model_heads

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        # Only the packed varlen path is sparse; plain forward is dense.
        self.calls["dense"] += 1
        return self._dense.forward(query, key, value, attn_metadata)

    def forward_varlen(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        cu_seqlens_host: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        context = get_forward_context_or_none()
        metadata = context.attn_metadata if context is not None else None
        if not self._sparse_capable or not isinstance(metadata, VedaAttentionMetadata):
            return self._dense_varlen(
                query, key, value, cu_seqlens, max_seqlen, cu_seqlens_host
            )
        runtime = _VedaRuntime.get(query.device)
        if (
            context.current_timestep < runtime.dense_first_n_steps
            or self.layer_index in runtime.dense_layers
        ):
            return self._dense_varlen(
                query, key, value, cu_seqlens, max_seqlen, cu_seqlens_host
            )

        bounds = (
            tuple(int(v) for v in cu_seqlens_host)
            if cu_seqlens_host is not None
            else tuple(int(v) for v in cu_seqlens.tolist())
        )
        expected = (0, int(metadata.used), int(metadata.seq_len))
        if (
            bounds != expected
            or int(max_seqlen) != expected[1]
            or query.shape[0] != expected[2]
        ):
            raise ValueError(
                "Veda attention expects the packed H3 layout (0, used, seq_len) = "
                f"{expected} with max_seqlen = used; got cu_seqlens {bounds}, "
                f"max_seqlen {max_seqlen}, {query.shape[0]} rows"
            )
        student = runtime.student(metadata)
        model_heads = self._model_heads_for(query.shape[1], query.device)
        if not runtime._announced_sparse:
            # Evidence that the sparse path ran (a silent dense fallback
            # would leave this line out of the log).
            runtime._announced_sparse = True
            logger.info(
                "Veda attention active: %s at step %d, %d local heads (model heads %d..%d)",
                self.prefix,
                context.current_timestep,
                query.shape[1],
                int(model_heads[0]),
                int(model_heads[-1]),
            )
        # The all-to-all leaves q / k / v as strided views on a staging buffer;
        # the tile gather kernels want plain [S, H, D] tensors.
        q, k, v = (t.contiguous() for t in (query, key, value))
        self.calls["sparse"] += 1
        with torch.no_grad():
            return student.forward_heads(q, k, v, self.layer_index, model_heads)

    def _dense_varlen(self, query, key, value, cu_seqlens, max_seqlen, cu_seqlens_host):
        self.calls["dense"] += 1
        return self._dense.forward_varlen(
            query,
            key,
            value,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            cu_seqlens_host=cu_seqlens_host,
        )
