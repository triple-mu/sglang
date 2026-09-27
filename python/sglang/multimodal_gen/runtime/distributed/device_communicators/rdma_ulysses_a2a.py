# SPDX-License-Identifier: Apache-2.0
"""RDMA transport for the MiniMax-H3 Ulysses exchange (see csrc/distributed/rdma_ulysses.cuh).

On a node whose GPUs sit on PCIe without NVLink, NCCL's all-to-all is the
slowest part of a sequence-parallel attention layer. This transport gives every
rank its own mlx5 RoCE port and moves the exchange with RDMA writes into
pre-registered buffers. Two operations serve the attention core:

* `exchange_chunks`: the quantised `[W, C]` payload, all_to_all_single semantics;
* `exchange_stats`: the per-layer quantisation statistics, all_gather semantics
  (a chunk exchange whose W chunks are copies of one record; NCCL's ring
  all-gather of this half-megabyte measured 8 ms per layer on the PCIe box);
* `gather_heads`: the attention output `[S_global, H_local, D] -> [S_local, H, D]`.

Construction and slot registration are collective (every rank all-gathers
metadata); a request never triggers a collective it did not already expect,
because both slots are registered at construction from the declared sequence
capacity. Every gate is a function of values identical on every rank, so the
group either all use the transport or all stay on NCCL.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.distributed.device_communicators.rdma_ulysses_topology import (
    RankProbe,
    RoutePlan,
    plan_rdma_route,
    probe_rank,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.nvtx_pytorch_hooks import maybe_nvtx_range

logger = init_logger(__name__)

MODE_GATHER = 1
MODE_CHUNKS = 2
_ALIGN = 128
_TRANSPORTS: dict[str, RdmaUlyssesA2A | None] = {}


class RdmaUlyssesError(RuntimeError):
    """The transport could not be brought up; the reason is group-wide."""


def _round_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


def chunks_capacity_bytes(max_seq_len: int, heads: int, world_size: int) -> int:
    """Payload capacity of the quantised exchange: `[W, chunk_bytes]` at the widest shard."""
    from sglang.kernels.ops.diffusion import ulysses_lowp_payload_spec

    local = _round_up(_round_up(max_seq_len, world_size * _ALIGN) // world_size, _ALIGN)
    spec = ulysses_lowp_payload_spec(
        batch=1, local_sequence=local, num_heads=heads, world_size=world_size
    )
    return _round_up(spec.world_size * spec.chunk_bytes, _ALIGN)


def stats_capacity_bytes(heads: int, head_dim: int, world_size: int) -> int:
    """Capacity of the statistics all-gather: `[W, 2 * heads * head_dim]` fp32."""
    return _round_up(world_size * 2 * heads * head_dim * 4, _ALIGN)


def gather_capacity_bytes(
    max_seq_len: int, heads: int, head_dim: int, element_size: int
) -> int:
    """Capacity of the attention-output gather: the whole `[S_global, H, D]` operand."""
    return _round_up(
        _round_up(max_seq_len, _ALIGN) * heads * head_dim * element_size, _ALIGN
    )


class RdmaUlyssesA2A:
    """One connected transport with its two registered slots."""

    def __init__(
        self,
        module,
        handle: int,
        *,
        group,
        world_size: int,
        rank: int,
        device: torch.device,
        plan: RoutePlan,
    ) -> None:
        self._module = module
        self._handle = handle
        self.group = group
        self.world_size = world_size
        self.rank = rank
        self.device = device
        self.plan = plan
        # slot index -> (mode, output bytes, landing bytes)
        self._slots: dict[str, tuple[int, torch.Tensor, torch.Tensor]] = {}
        self.exchanges = 0
        self._closed = False

    # ---------------------------------------------------------------- slots --
    def _register(self, name: str, mode: int, capacity_bytes: int) -> None:
        """Collective: allocate, register and connect one slot on every rank."""
        index, output, landing, wire = self._module.register_slot(
            self._handle, mode, capacity_bytes
        )
        wires = _gather_object(self.group, list(wire), self.device)
        self._module.connect_slot(self._handle, index, [b for w in wires for b in w])
        self._slots[name] = (index, output, landing)

    def _slot(self, name: str, nbytes: int) -> tuple[int, torch.Tensor, torch.Tensor]:
        index, output, landing = self._slots[name]
        if nbytes > output.numel():
            raise RdmaUlyssesError(
                f"{name} operand of {nbytes} bytes exceeds the registered capacity "
                f"{output.numel()}; raise --minimax-h3-ulysses-max-seq-len"
            )
        return index, output, landing

    def packed_input_buffer(self, shape: tuple[int, int]) -> torch.Tensor:
        """Where to build the `[W, C]` payload so the NIC reads it without a staging copy."""
        world_size, chunk = (int(v) for v in shape)
        _, _, landing = self._slot("chunks", world_size * chunk)
        return landing[: world_size * chunk].view(world_size, chunk)

    def gather_landing(
        self, shape: tuple[int, int, int], dtype: torch.dtype
    ) -> torch.Tensor:
        """Where to write the attention output `[S_global, H_local, D]` before `gather_heads`."""
        numel = 1
        for v in shape:
            numel *= int(v)
        _, _, landing = self._slot("gather", numel * dtype.itemsize)
        return landing[: numel * dtype.itemsize].view(dtype).view(shape)

    # ------------------------------------------------------------- exchange --
    def exchange_chunks(self, payload: torch.Tensor) -> torch.Tensor:
        """`[W, C]` uint8 in, `[W, C]` out; row `i` of the result came from rank `i`."""
        if payload.dtype is not torch.uint8 or payload.ndim != 2:
            raise ValueError("exchange_chunks takes a [world_size, C] uint8 payload")
        index, output, _ = self._slot("chunks", payload.numel())
        out = output[: payload.numel()].view(payload.shape)
        with maybe_nvtx_range("rdma_a2a_exchange_chunks"):
            self._module.exchange(self._handle, index, payload, out)
        self._count()
        return out

    def exchange_stats(self, stats: torch.Tensor) -> torch.Tensor:
        """All-gather of one contiguous record: `[...]` -> `[W, ...]`, row `i` from rank `i`."""
        nbytes = stats.numel() * stats.element_size()
        index, output, landing = self._slot("stats", self.world_size * nbytes)
        send = landing[: self.world_size * nbytes].view(self.world_size, nbytes)
        send.copy_(
            stats.reshape(1, -1).view(torch.uint8).expand(self.world_size, nbytes)
        )
        out = output[: self.world_size * nbytes].view(self.world_size, nbytes)
        with maybe_nvtx_range("rdma_a2a_exchange_stats"):
            self._module.exchange(self._handle, index, send, out)
        self._count()
        return out.view(stats.dtype).view(self.world_size, *stats.shape)

    def gather_heads(self, x: torch.Tensor) -> torch.Tensor:
        """`[S_global, H_local, D]` -> `[S_local, H, D]`, heads ordered by source rank."""
        if x.ndim != 3:
            raise ValueError("gather_heads takes [S_global, H_local, D]")
        s_global, h_local, dim = x.shape
        index, output, landing = self._slot("gather", x.numel() * x.element_size())
        if x.data_ptr() != landing.data_ptr() or not x.is_contiguous():
            staged = landing[: x.numel() * x.element_size()].view(x.dtype).view(x.shape)
            staged.copy_(x)
            x = staged
        out = (
            output[: x.numel() * x.element_size()]
            .view(x.dtype)
            .view(s_global // self.world_size, h_local * self.world_size, dim)
        )
        with maybe_nvtx_range("rdma_a2a_gather_heads"):
            self._module.exchange(self._handle, index, x, out)
        self._count()
        return out

    def _count(self) -> None:
        self.exchanges += 1
        if self.exchanges == 1:
            logger.info(
                "RDMA Ulysses carried its first exchange (rank %d on %s, GID %d)",
                self.rank,
                self.plan.nics[self.rank],
                self.plan.gid_indices[self.rank],
            )

    # ------------------------------------------------------------- lifetime --
    def shutdown(self) -> None:
        """Collective staged teardown; every rank must call this before any rank exits."""
        if self._closed:
            return
        self._closed = True
        logger.info(
            "RDMA Ulysses carried %d exchanges over its lifetime", self.exchanges
        )
        self._slots.clear()
        for name, step in (
            ("teardown safety", lambda: bool(self._module.teardown_safe(self._handle))),
            ("synchronize", lambda: (torch.cuda.synchronize(self.device), True)[1]),
            ("disconnect", lambda: (self._module.disconnect(self._handle), True)[1]),
            ("dispose", lambda: (self._module.dispose(self._handle), True)[1]),
        ):
            try:
                ok = step()
                detail = None if ok else "native GPU work could not be bounded"
            except Exception as error:  # noqa: BLE001 -- must still vote
                ok, detail = False, f"{type(error).__name__}: {error}"
            votes = _gather_object(self.group, (ok, detail), self.device)
            if not all(v[0] for v in votes):
                raise RdmaUlyssesError(
                    f"RDMA Ulysses {name} failed on some rank; terminate the process group: "
                    f"{[v[1] for v in votes if not v[0]]}"
                )


def _gather_object(group, payload, device: torch.device) -> list:
    out = [None] * dist.get_world_size(group)
    with torch.cuda.device(device):
        dist.all_gather_object(out, payload, group=group)
    return out


def _vote(group, device: torch.device, error: str | None) -> str | None:
    """Every rank's local outcome, reduced to the first error or None."""
    votes = _gather_object(group, error, device)
    for rank, vote in enumerate(votes):
        if vote is not None:
            return f"rank {rank}: {vote}"
    return None


def _build(
    group, device: torch.device, *, max_seq_len: int, heads: int, head_dim: int
) -> RdmaUlyssesA2A:
    """Staged collective construction; raises the same RdmaUlyssesError on every rank."""
    from sglang.kernels.ops.communication.rdma_ulysses import load_rdma_ulysses

    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    if not 2 <= world_size <= 8:
        raise RdmaUlyssesError(f"RDMA Ulysses needs 2 to 8 ranks, got {world_size}")

    probe = probe_rank(rank, device.index)
    probes = [
        p if isinstance(p, RankProbe) else RankProbe(rank=i, error="probe payload lost")
        for i, p in enumerate(_gather_object(group, probe, device))
    ]
    plan = plan_rdma_route(probes)  # deterministic in the gathered list

    module, error = None, None
    try:
        module = load_rdma_ulysses()
    except Exception as exc:  # noqa: BLE001 -- becomes a vote
        error = f"{type(exc).__name__}: {exc}"
    if (reason := _vote(group, device, error)) is not None:
        raise RdmaUlyssesError(f"RDMA Ulysses JIT build failed: {reason}")

    handle, wire, error = None, None, None
    try:
        handle, wire = module.init(
            rank,
            world_size,
            device.index,
            plan.nics[rank],
            plan.gid_indices[rank],
            int(envs.SGLANG_DIFFUSION_RDMA_ULYSSES_TIMEOUT_MS),
        )
        wire = list(wire)
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    wires = _gather_object(group, (error, wire), device)
    failures = [f"rank {i}: {e}" for i, (e, _) in enumerate(wires) if e is not None]
    if failures:
        if handle is not None:
            module.dispose(handle)
        raise RdmaUlyssesError(f"RDMA Ulysses init failed: {failures[0]}")

    error = None
    try:
        module.connect(handle, [b for _, w in wires for b in w])
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    if (reason := _vote(group, device, error)) is not None:
        module.dispose(handle)
        raise RdmaUlyssesError(f"RDMA Ulysses connect failed: {reason}")

    transport = RdmaUlyssesA2A(
        module,
        handle,
        group=group,
        world_size=world_size,
        rank=rank,
        device=device,
        plan=plan,
    )
    try:
        transport._register(
            "chunks", MODE_CHUNKS, chunks_capacity_bytes(max_seq_len, heads, world_size)
        )
        transport._register(
            "gather",
            MODE_GATHER,
            gather_capacity_bytes(max_seq_len, heads, head_dim, 2),
        )
        transport._register(
            "stats", MODE_CHUNKS, stats_capacity_bytes(heads, head_dim, world_size)
        )
        torch.cuda.synchronize(device)
        error = None
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    if (reason := _vote(group, device, error)) is not None:
        transport.shutdown()
        raise RdmaUlyssesError(f"RDMA Ulysses slot registration failed: {reason}")
    logger.info(
        "RDMA Ulysses ready: %d ranks, NICs %s, GIDs %s, capacity %.0f MB per slot",
        world_size,
        list(plan.nics),
        list(plan.gid_indices),
        chunks_capacity_bytes(max_seq_len, heads, world_size) / 1e6,
    )
    return transport


def get_rdma_ulysses_a2a(
    group,
    device: torch.device,
    *,
    max_seq_len: int,
    heads: int,
    head_dim: int,
    strict: bool,
) -> RdmaUlyssesA2A | None:
    """The transport for this Ulysses group, built once; None means stay on NCCL.

    Collective on first call. Failure is decided group-wide: either every rank
    gets the transport or every rank gets None (or, with `strict`, the error).
    """
    name = getattr(group, "group_name", None) or str(id(group))
    if name in _TRANSPORTS:
        return _TRANSPORTS[name]
    try:
        transport = _build(
            group, device, max_seq_len=max_seq_len, heads=heads, head_dim=head_dim
        )
    except RdmaUlyssesError as error:
        _TRANSPORTS[name] = None
        if strict:
            raise
        logger.warning("RDMA Ulysses unavailable (%s); staying on NCCL", error)
        return None
    _TRANSPORTS[name] = transport
    return transport


def active_rdma_ulysses_a2a(group) -> RdmaUlyssesA2A | None:
    """The already-built transport for this group, without triggering construction."""
    name = getattr(group, "group_name", None) or str(id(group))
    return _TRANSPORTS.get(name)


def shutdown_rdma_ulysses_a2a() -> None:
    """Collective: every rank must call this before its process groups go away."""
    for name in list(_TRANSPORTS):
        transport = _TRANSPORTS.pop(name)
        if transport is not None:
            transport.shutdown()


__all__ = [
    "RdmaUlyssesA2A",
    "RdmaUlyssesError",
    "active_rdma_ulysses_a2a",
    "chunks_capacity_bytes",
    "gather_capacity_bytes",
    "get_rdma_ulysses_a2a",
    "shutdown_rdma_ulysses_a2a",
    "stats_capacity_bytes",
]
