# SPDX-License-Identifier: Apache-2.0
"""FlashInfer PCIe transport for the MiniMax-H3 Ulysses exchange.

On a single node without NVLink the exchange is link-bound. FlashInfer's PCIe
Ulysses backend moves the payload with copy engines rather than SM stores, and
at world size 8 it routes the cross-NUMA half over each rank's local mlx5 NIC
while the same-NUMA half stays on CUDA P2P.

The adapter exposes ``qkv_buffer`` / ``exchange_input`` / ``exchange_output``
to the attention core and pre-registers each scatter/gather pair before the
projection for that sequence geometry begins.

Q/K/V travel as one operand. The projection already emits
``[tokens, heads, 3, head_dim]``, so the head axis carries ``3 * H`` entries and
splitting it across ranks hands each one ``H/ws`` heads with their own Q, K and
V beside them. Nothing is packed going in and nothing is copied coming out, and
a layer costs two exchanges rather than four.

``qkv_buffer`` hands the projection the buffer the NIC reads, so the hybrid
route stages nothing on the way in. The way back keeps its staging copy: the
attention backend allocates its own output and will not be told to write
elsewhere, so staging it here and staging it inside FlashInfer cost the same
single pass.

That last part is the point, and it is not what an isolated benchmark would
tell you: per byte, this backend is no faster fused than unfused. But the hybrid
route blocks the calling thread inside every exchange, so the number of
exchanges is a cost of its own, separate from what each one moves.

Registration is collective and permanent -- constructing the communicator and
allocating an output both need every rank -- so every gate below is a function
of values identical on every rank.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

_SUPPORTED_WORLD_SIZES = (2, 4, 8)
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)

_TRANSPORTS: dict[str, "FlashInferUlyssesA2A | None"] = {}


def _unusable(world_size: int, dtype: torch.dtype) -> str | None:
    """Why this exchange cannot use the PCIe backend, or None if it can.

    Only conditions that exist before a group does, and only ones every rank
    evaluates identically.
    """
    try:
        import flashinfer.comm  # noqa: F401
    except ImportError as error:
        return f"flashinfer is not installed ({error})"
    if world_size not in _SUPPORTED_WORLD_SIZES:
        return f"world size {world_size} is not one of {_SUPPORTED_WORLD_SIZES}"
    if dtype not in _SUPPORTED_DTYPES:
        return f"dtype {dtype} is unsupported"
    if torch.is_grad_enabled():
        return "autograd is enabled"
    if torch.compiler.is_compiling():
        return "running under torch.compile"
    if torch.cuda.is_current_stream_capturing():
        return "running under CUDA graph capture"
    return None


class FlashInferUlyssesA2A:
    """One communicator, plus per-shape send buffers and registered outputs."""

    def __init__(
        self,
        comm,
        world_size: int,
        max_shapes: int,
        max_elems: int,
        *,
        strict: bool,
    ) -> None:
        self._comm = comm
        self.world_size = world_size
        self.device = comm.device
        self._max_shapes = max_shapes
        self.max_elems = max_elems
        self.strict = strict
        # Constructing a transport says nothing about whether anything routes
        # through it. An integration that silently keeps calling NCCL looks
        # exactly like one that works, so count what is actually carried and
        # say so on the first one and again at teardown. A benchmark run whose
        # log has no such line measured NCCL, whatever it was labelled.
        self._exchanges = 0
        # (role, source shape, dtype) -> registered exchange output
        self._outputs: dict[tuple, torch.Tensor] = {}
        # Same key -> the operand buffer that output reads with no staging copy.
        self._sends: dict[tuple, torch.Tensor] = {}
        self._declined: set[tuple] = set()
        self._geometries: set[tuple] = set()

    @property
    def backend(self) -> str:
        return self._comm.backend

    @property
    def transport(self) -> str | None:
        return self._comm.transport

    @property
    def pcie_engine(self) -> str:
        return self._comm.pcie_engine

    def _output(self, role: str, sample: torch.Tensor, op: str):
        """The registered exchange output for one role and operand shape.

        Allocated on first sight. ``allocate_output`` is collective and the key
        comes from rank-identical values, so every rank allocates the same
        outputs in the same order or none of them. The operand itself stays the
        caller's -- the transport no longer registers or tracks it, so there is
        nothing to project or stage into.
        """
        source_shape, dtype = tuple(sample.shape), sample.dtype
        key = (role, source_shape, dtype)
        cached = self._outputs.get(key)
        if cached is not None:
            return cached
        if key in self._declined:
            return None
        # Fused QKV uses one scatter and attention output one gather
        # registration per geometry. Count geometry pairs, not individual
        # registrations, so a budget of four really admits four request shapes.
        if len(self._outputs) >= 2 * self._max_shapes:
            self._declined.add(key)
            reason = (
                f"{self._max_shapes} sequence geometries are already registered; "
                f"cannot register {key}"
            )
            if self.strict:
                raise RuntimeError(f"flashinfer ulysses strict mode: {reason}")
            logger.info("flashinfer ulysses: %s; staying on NCCL", reason)
            return None
        if torch.Size(source_shape).numel() > self.max_elems:
            self._declined.add(key)
            reason = (
                f"{key} exceeds the declared capacity of {self.max_elems} elements; "
                "raise --minimax-h3-ulysses-max-seq-len"
            )
            if self.strict:
                raise RuntimeError(f"flashinfer ulysses strict mode: {reason}")
            logger.warning("flashinfer ulysses: %s; staying on NCCL", reason)
            return None
        try:
            out = self._comm.allocate_output(sample, op)
        except Exception as error:  # noqa: BLE001
            self._declined.add(key)
            if self.strict:
                raise RuntimeError(
                    f"flashinfer ulysses strict mode: could not register {key}"
                ) from error
            logger.warning(
                "flashinfer ulysses: could not register %s (%s); staying on NCCL "
                "for this shape",
                key,
                error,
            )
            return None
        self._outputs[key] = out
        return out

    def prepare_geometry(
        self,
        tokens: int,
        heads: int,
        head_dim: int,
        dtype: torch.dtype,
        *,
        direct_input: bool,
    ) -> None:
        """Collectively pre-register the paired scatter/gather workspaces."""
        geometry = (tokens, heads, head_dim, dtype)
        if geometry in self._geometries:
            return
        if len(self._geometries) >= self._max_shapes:
            reason = f"geometry budget {self._max_shapes} exhausted before {geometry}"
            if self.strict:
                raise RuntimeError(f"flashinfer ulysses strict mode: {reason}")
            logger.warning("flashinfer ulysses: %s; staying on NCCL", reason)
            return

        qkv = torch.empty(
            (1, tokens, heads * 3, head_dim), dtype=dtype, device=self.device
        )
        scatter = self._output("qkv", qkv, "scatter_heads")
        if scatter is None:
            return
        if direct_input:
            try:
                send = self._comm.allocate_input(scatter, "scatter_heads")
            except Exception as error:  # noqa: BLE001
                if self.strict:
                    raise RuntimeError(
                        "flashinfer ulysses strict mode: direct input buffer "
                        f"allocation failed for {geometry}"
                    ) from error
                logger.warning(
                    "flashinfer ulysses: direct input unavailable for %s (%s); "
                    "using the staged path",
                    geometry,
                    error,
                )
            else:
                self._sends[("qkv", tuple(qkv.shape), dtype)] = send

        gathered_source = torch.empty(
            (1, tokens * self.world_size, heads // self.world_size, head_dim),
            dtype=dtype,
            device=self.device,
        )
        if self._output("attn_out", gathered_source, "gather_heads") is None:
            return
        self._geometries.add(geometry)

    def stats(self, reset: bool = False) -> dict[str, object]:
        stats = dict(self._comm.stats(reset=reset))
        stats.update(
            {
                "sglang_exchanges": self._exchanges,
                "registered_geometries": len(self._geometries),
                "geometry_budget": self._max_shapes,
                "declined_geometries": len(self._declined),
                "strict": self.strict,
            }
        )
        if reset:
            self._exchanges = 0
        return stats

    # -------------------------------------------------------------- input --
    def qkv_buffer(self, tokens: int, heads: int, head_dim: int, dtype):
        """The projection's destination, which is also the exchange's source.

        On the hybrid route the NIC reads out of a buffer the transport owns,
        so an operand that arrives anywhere else is copied there first --
        203.7 MB per layer at this model's sizes. Projecting straight in here
        removes that copy. Returns None when the shape is not registrable, in
        which case the caller projects wherever it likes and the exchange
        stages as usual.

        The other routes have no such buffer and hand back an ordinary tensor,
        so the caller has one code path either way.
        """
        shape = (1, tokens, heads * 3, head_dim)
        key = ("qkv", shape, dtype)
        send = self._sends.get(key)
        if send is None:
            # allocate_output wants an operand to take its geometry from, and
            # this is the cold path, so a throwaway is cheaper than plumbing a
            # shape-only variant through the whole stack.
            sample = torch.empty(shape, dtype=dtype, device=self.device)
            out = self._output("qkv", sample, "scatter_heads")
            del sample
            if out is None:
                return None
            try:
                send = self._comm.allocate_input(out, "scatter_heads")
            except Exception as error:  # noqa: BLE001
                if self.strict:
                    raise RuntimeError(
                        f"flashinfer ulysses strict mode: no operand buffer for {key}"
                    ) from error
                logger.warning(
                    "flashinfer ulysses: no operand buffer for %s (%s); this "
                    "shape keeps its staging copy",
                    key,
                    error,
                )
                return None
            self._sends[key] = send
        # The projection wants [tokens, heads, 3, head_dim] and the exchange
        # wants [1, tokens, 3 * heads, head_dim]: same memory, same order.
        return send.view(tokens, heads, 3, head_dim)

    def exchange_input(self, qkv: torch.Tensor):
        """``[T_local, H, 3, D]`` -> three ``[T_global, H/ws, D]`` views."""
        tokens, heads, three, head_dim = qkv.shape
        assert three == 3, f"expected [T, H, 3, D], got {tuple(qkv.shape)}"
        source = qkv.view(1, tokens, heads * 3, head_dim)
        out = self._output("qkv", source, "scatter_heads")
        if out is None:
            return None
        exchanged = self._comm.scatter_heads(source, out=out)
        self._count(source)
        merged = exchanged.view(
            tokens * self.world_size, heads // self.world_size, 3, head_dim
        )
        return merged[:, :, 0], merged[:, :, 1], merged[:, :, 2]

    # ------------------------------------------------------------- output --
    def exchange_output(self, attn_out: torch.Tensor):
        """``[T_global, H/ws, D]`` -> ``[T_local, H, D]``, head axis merged."""
        tokens, heads, head_dim = attn_out.shape
        source = attn_out.view(1, tokens, heads, head_dim)
        out = self._output("attn_out", source, "gather_heads")
        if out is None:
            return None
        exchanged = self._comm.gather_heads(source, out=out)
        self._count(source)
        return exchanged.view(
            tokens // self.world_size, heads * self.world_size, head_dim
        )

    # --------------------------------------------------------------- life --
    def _count(self, source: torch.Tensor) -> None:
        self._exchanges += 1
        if self._exchanges == 1:
            logger.info(
                "flashinfer ulysses carried its first exchange "
                "(backend=%s, transport=%s, pcie_engine=%s, operand=%s)",
                self.backend,
                self.transport,
                self.pcie_engine,
                tuple(source.shape),
            )

    def shutdown(self) -> None:
        """Collective: every rank must call this before any rank exits."""
        logger.info(
            "flashinfer ulysses carried %d exchanges over its lifetime",
            self._exchanges,
        )
        self._sends.clear()
        self._outputs.clear()
        self._geometries.clear()
        self._comm.close()


def get_flashinfer_ulysses_a2a(
    group,
    device: torch.device,
    dtype: torch.dtype,
    capacity: int,
    *,
    strict: bool,
    max_shapes: int,
    pcie_engine: str,
) -> FlashInferUlyssesA2A | None:
    """The transport for this process group, constructing it once.

    ``capacity`` is the declared maximum and sizes every registration. It cannot
    be raised in place, so an operand beyond it stays on NCCL rather than
    forcing a rebuild -- which is what every other communication workspace in
    both trees does, and measurably cheaper than the alternative.
    """
    world_size = dist.get_world_size(group)
    reason = _unusable(world_size, dtype)
    name = getattr(group, "group_name", None) or str(id(group))
    if reason is not None:
        if strict:
            raise RuntimeError(f"flashinfer ulysses strict mode: {reason}")
        if name not in _TRANSPORTS:
            _TRANSPORTS[name] = None
            logger.info("flashinfer ulysses is unusable (%s); staying on NCCL", reason)
        return None

    if name in _TRANSPORTS:
        return _TRANSPORTS[name]

    from flashinfer.comm import UlyssesCommunicator

    # The constructor raises rather than returning None, and FlashInfer makes
    # topology failures collective. Vote anyway: a rank-local failure it did not
    # classify as collective would otherwise split the group across two paths,
    # which is a hang rather than a slowdown.
    comm = None
    error: BaseException | None = None
    try:
        comm = UlyssesCommunicator(
            group,
            max_elems=capacity,
            dtype=dtype,
            backend="pcie",
            device=device,
            pcie_engine=pcie_engine,
        )
    except BaseException as exc:  # noqa: BLE001
        error = exc

    vote = torch.tensor(
        [1 if comm is not None else 0], dtype=torch.int32, device=device
    )
    dist.all_reduce(vote, op=dist.ReduceOp.MIN, group=group)
    if vote.item() == 0:
        if comm is not None:
            try:
                comm.close()
            except Exception:  # noqa: BLE001
                logger.exception("flashinfer ulysses: close after a failed start")
        _TRANSPORTS[name] = None
        if strict:
            raise RuntimeError(
                "flashinfer ulysses strict mode: communicator construction failed"
            ) from error
        logger.warning(
            "flashinfer ulysses could not start on some rank (%s); staying on NCCL",
            error,
        )
        return None

    try:
        comm.preflight(strict=strict)
    except Exception:
        # The strict world-size-8 gate is rank-invariant. Close collectively
        # while the process group is still alive, then surface the route error.
        comm.close()
        raise

    transport = FlashInferUlyssesA2A(
        comm,
        world_size,
        max_shapes,
        capacity,
        strict=strict,
    )
    _TRANSPORTS[name] = transport
    logger.info(
        "flashinfer ulysses started for this group "
        "(backend=%s, transport=%s, pcie_engine=%s, world_size=%d, capacity=%d)",
        transport.backend,
        transport.transport,
        transport.pcie_engine,
        world_size,
        capacity,
    )
    return transport


def shutdown_flashinfer_ulysses_a2a() -> None:
    """Collective: every rank must call this before any rank exits."""
    for name in list(_TRANSPORTS):
        transport = _TRANSPORTS.pop(name)
        if transport is None:
            continue
        try:
            transport.shutdown()
        except Exception:  # noqa: BLE001
            logger.exception("flashinfer ulysses: teardown failed for %s", name)


def reset_flashinfer_ulysses_request_stats() -> None:
    """Start a request-local interval without disturbing registrations."""
    for transport in _TRANSPORTS.values():
        if transport is not None:
            transport.stats(reset=True)


def validate_flashinfer_ulysses_request(
    expected_exchanges: int, *, strict: bool
) -> list[dict[str, object]]:
    """Validate that every live transport carried the whole request."""
    reports = [
        transport.stats() for transport in _TRANSPORTS.values() if transport is not None
    ]
    problems = []
    if not reports:
        problems.append("no active FlashInfer Ulysses communicator")
    for report in reports:
        actual = int(report["sglang_exchanges"])
        if actual != expected_exchanges:
            problems.append(
                f"rank-local exchange delta is {actual}, expected {expected_exchanges}"
            )
    if problems:
        message = "MiniMax-H3 FlashInfer Ulysses request validation: " + "; ".join(
            problems
        )
        if strict:
            raise RuntimeError(message)
        logger.warning("%s; request used an NCCL fallback", message)
    else:
        logger.info(
            "MiniMax-H3 FlashInfer Ulysses request completed with %d exchanges",
            expected_exchanges,
        )
    return reports
