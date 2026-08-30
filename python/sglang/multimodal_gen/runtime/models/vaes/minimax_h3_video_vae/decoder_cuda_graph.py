# SPDX-License-Identifier: Apache-2.0
"""Per-tile CUDA graph replay for the MiniMax H3 VAE ViT decoder (V2 fusion).

Tiled decode runs the same fixed-shape ViT forward ~100 times per request
(~700 kernel launches each) with no collectives inside a tile. Dispatching one
tile costs ~12 ms CPU (H200, torch 2.13) vs ~260 us for a graph replay, so
this runner captures one CUDA graph per observed input signature and replays
it through static buffers, freeing the CPU the eager loop burns on launches.

Correctness-first protocol, per signature:

1. First sight runs eager and retains clones of the tile's input and output.
2. Second sight warms up on a side stream, captures the graph, then replays
   the retained first tile and requires ``torch.equal`` against its eager
   output before any graphed result is returned (the runtime analogue of
   ``BitExactFusionGate`` first-sight verification).
3. Any capture failure or mismatch permanently disables the runner for this
   process; decode continues eager.
"""

from __future__ import annotations

import torch
from diffusers.utils import logging

from sglang.multimodal_gen.runtime.platforms import current_platform

from .vit_utils import _env_flag

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

_CUDA_GRAPH_ENV = "MINIMAX_H3_VAE_DECODER_CUDA_GRAPH"
# Interior/right/bottom/corner tile shapes, doubled for head/tail temporal
# clip variants; a canvas needing more signatures stays eager past the cap.
_MAX_SHAPE_ENTRIES = 16
# These env toggles wrap parts of the tile forward in torch.compile
# (reduce-overhead uses its own cudagraphs); capturing over them is unsafe.
_TORCH_COMPILE_ENVS = (
    "MINIMAX_H3_VAE_DECODER_VIT_FF_TORCH_COMPILE",
    "MINIMAX_H3_VAE_DECODER_VIT_ROPE_TORCH_COMPILE",
)


def _layerwise_offload_active(module) -> bool:
    """True when layerwise offload hooks manage this module's weights.

    Offload pages block weights in and out around each block forward; the
    hooks host-sync and the captured weight pointers would go stale.
    """
    if module is None:
        return False
    from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
        LayerwiseOffloadableModuleMixin,
    )

    return isinstance(module, LayerwiseOffloadableModuleMixin) and bool(
        module.layerwise_offload_managers
    )


class _ShapeEntry:
    """Per-signature state machine: eager-once -> captured -> replay."""

    __slots__ = (
        "saved_input",
        "saved_output",
        "graph",
        "static_input",
        "static_output",
        "pinned_rotary_record",
    )

    def __init__(self, saved_input: torch.Tensor, saved_output: torch.Tensor) -> None:
        self.saved_input = saved_input
        self.saved_output = saved_output
        self.graph = None
        self.static_input = None
        self.static_output = None
        self.pinned_rotary_record = None


class DecoderTileCudaGraphRunner:
    """Shape-keyed CUDA graph runner around ``ViT3DDecoder.forward``.

    ``offload_owner`` is the module whose layerwise-offload state gates
    capture (the composed VAE); pass ``None`` when no offload can apply.
    """

    def __init__(self, *, decoder, offload_owner=None) -> None:
        self._decoder = decoder
        self._offload_owner = offload_owner
        self._platform_ok = current_platform.is_cuda()
        self._entries: dict = {}
        self._entries_epoch = decoder._graph_epoch
        self._pool = None
        self._disabled_reason: str | None = None
        self._entry_cap_warned = False

    def _disable(self, reason: str) -> None:
        self._disabled_reason = reason
        self._entries.clear()
        logger.warning(
            f"MiniMax H3 VAE decoder CUDA graph disabled; decode stays eager "
            f"(correctness preserved): {reason}"
        )

    def _eligible(self, z: torch.Tensor) -> bool:
        if self._disabled_reason is not None or not self._platform_ok:
            return False
        if not _env_flag(_CUDA_GRAPH_ENV, "1"):
            return False
        if not z.is_cuda:
            return False
        if 0 in z.stride():
            # A broadcast view cannot round-trip through the strided static
            # buffer copy; decode such inputs eager.
            return False
        if self._decoder.training or torch.is_grad_enabled():
            return False
        if torch.compiler.is_compiling():
            return False
        if torch.cuda.is_current_stream_capturing():
            return False
        if any(_env_flag(name, "0") for name in _TORCH_COMPILE_ENVS):
            return False
        return not _layerwise_offload_active(self._offload_owner)

    def _entry_key(self, z: torch.Tensor):
        # Strides are part of the signature: the tile forward's patchify
        # aliases a contiguous input but copies a strided tile view, which
        # changes downstream GEMM layouts and therefore reduction order.
        autocast_enabled = z.is_cuda and torch.is_autocast_enabled("cuda")
        return (
            tuple(z.shape),
            tuple(z.stride()),
            z.dtype,
            z.device.index,
            autocast_enabled,
            torch.get_autocast_dtype("cuda") if autocast_enabled else None,
        )

    def run(self, z: torch.Tensor) -> torch.Tensor:
        decoder = self._decoder
        if not self._eligible(z):
            return decoder(z)

        if self._entries_epoch != decoder._graph_epoch:
            # Weights were re-materialized (.to() / autocast weight prep); the
            # captured kernels hold raw pointers into the old storages.
            self._entries.clear()
            self._entries_epoch = decoder._graph_epoch

        key = self._entry_key(z)
        entry = self._entries.get(key)
        if entry is None:
            if len(self._entries) >= _MAX_SHAPE_ENTRIES:
                if not self._entry_cap_warned:
                    self._entry_cap_warned = True
                    logger.warning(
                        f"MiniMax H3 VAE decoder CUDA graph signature cap "
                        f"({_MAX_SHAPE_ENTRIES}) reached; new tile shapes "
                        f"decode eager"
                    )
                return decoder(z)
            out = decoder(z)
            self._entries[key] = _ShapeEntry(
                saved_input=z.detach().clone(), saved_output=out.detach().clone()
            )
            return out

        if entry.graph is None:
            return self._capture_verify_replay(entry, z)

        entry.static_input.copy_(z)
        entry.graph.replay()
        return entry.static_output.clone()

    def _capture_verify_replay(
        self, entry: _ShapeEntry, z: torch.Tensor
    ) -> torch.Tensor:
        decoder = self._decoder
        device = z.device
        rotary_record_before = decoder._rotary_pos_emb_cache
        try:
            # Reproduce the tile's exact strides (see _entry_key): a contiguous
            # buffer would flip the patchify alias-vs-copy decision and replay
            # different GEMM bits than the eager first tile.
            static_input = torch.empty_strided(
                z.size(), z.stride(), dtype=z.dtype, device=device
            )
            static_input.copy_(z)

            # Side-stream warmup (torch's capture recipe): settles cuBLASLt
            # algorithm selection and lazy JIT state, and primes the rotary
            # cache for this signature so no cache build lands in the graph.
            side_stream = torch.cuda.Stream(device=device)
            side_stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(side_stream):
                decoder(static_input)
            torch.cuda.current_stream(device).wait_stream(side_stream)

            # Captured kernels read the cached rotary tensors by raw pointer;
            # pin the record so a later signature cannot evict and free it.
            pinned_rotary_record = decoder._rotary_pos_emb_cache

            graph = torch.cuda.CUDAGraph()
            if self._pool is None:
                self._pool = torch.cuda.graph_pool_handle()
            with torch.cuda.graph(graph, pool=self._pool):
                static_output = decoder(static_input)
        except Exception as exc:
            decoder._rotary_pos_emb_cache = rotary_record_before
            self._disable(f"capture failed: {type(exc).__name__}: {exc}")
            return decoder(z)
        # A cache build captured on a rotary-cache miss would leave the record
        # pointing at graph-pool memory that capture never executed into.
        decoder._rotary_pos_emb_cache = pinned_rotary_record
        # Any parameter cast the capture put into the global autocast cache
        # lives in graph-pool memory; eager callers must never reuse it.
        torch.clear_autocast_cache()

        # First replay re-runs the retained first tile: replay must reproduce
        # the eager output bit-for-bit before any graphed result is trusted.
        static_input.copy_(entry.saved_input)
        graph.replay()
        if not torch.equal(static_output, entry.saved_output):
            self._disable("replay output is not bit-exact vs the retained eager tile")
            return decoder(z)

        entry.graph = graph
        entry.static_input = static_input
        entry.static_output = static_output
        entry.pinned_rotary_record = pinned_rotary_record
        entry.saved_input = None
        entry.saved_output = None

        static_input.copy_(z)
        graph.replay()
        return static_output.clone()
