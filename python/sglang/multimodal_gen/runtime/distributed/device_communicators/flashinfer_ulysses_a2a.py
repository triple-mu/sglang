# SPDX-License-Identifier: Apache-2.0
"""FlashInfer PCIe transport for the MiniMax-H3 Ulysses exchange.

On a single node without NVLink the exchange is link-bound. FlashInfer's PCIe
Ulysses backend moves the payload with copy engines rather than SM stores, and
at world size 8 it routes the cross-NUMA half over each rank's local mlx5 NIC
while the same-NUMA half stays on CUDA P2P.

This exposes the same surface as the fast-ulysses transport beside it --
``qkv_buffer`` / ``exchange_input`` / ``attn_out_buffer`` / ``exchange_output``
-- so the attention core does not care which one it is talking to.

Q/K/V travel as one operand. The projection writes into ``qkv_buffer`` in the
``[tokens, heads, 3, head_dim]`` layout, so the head axis already carries
``3 * H`` entries and splitting it across ranks hands each one ``H/ws`` heads
with their own Q, K and V beside them. Nothing is packed going in and nothing is
copied coming out, and a layer costs two exchanges rather than four.

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

from sglang.multimodal_gen import envs
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
    if not envs.SGLANG_DIFFUSION_MINIMAX_H3_ULYSSES_PCIE:
        return "disabled"
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

    def __init__(self, comm, world_size: int, max_shapes: int, max_elems: int) -> None:
        self._comm = comm
        self.world_size = world_size
        self.device = comm.device
        self._max_shapes = max_shapes
        self.max_elems = max_elems
        # (role, source shape, dtype) -> (send buffer, registered output)
        self._pairs: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
        self._declined: set[tuple] = set()

    @property
    def backend(self) -> str:
        return self._comm.backend

    @property
    def transport(self) -> str | None:
        return self._comm.transport

    def _pair(self, role: str, source_shape: tuple, op: str, dtype):
        """The send buffer and its registered output for one role and shape.

        Both are allocated on first sight. ``allocate_output`` is collective and
        the key comes from rank-identical values, so every rank allocates the
        same pairs in the same order or none of them.
        """
        key = (role, source_shape, dtype)
        cached = self._pairs.get(key)
        if cached is not None:
            return cached
        if key in self._declined:
            return None
        if len(self._pairs) >= self._max_shapes:
            self._declined.add(key)
            logger.info(
                "flashinfer ulysses: %d shapes already registered, staying on NCCL for %s",
                self._max_shapes,
                key,
            )
            return None
        if torch.Size(source_shape).numel() > self.max_elems:
            self._declined.add(key)
            logger.warning(
                "flashinfer ulysses: %s exceeds the declared capacity of %d elements, "
                "so this shape stays on NCCL. Set "
                "SGLANG_DIFFUSION_MINIMAX_H3_ULYSSES_MAX_SEQ_LEN to the largest packed "
                "sequence this deployment serves.",
                key,
                self.max_elems,
            )
            return None
        try:
            send = torch.empty(source_shape, dtype=dtype, device=self.device)
            out = self._comm.allocate_output(send, op)
        except Exception as error:  # noqa: BLE001
            self._declined.add(key)
            logger.warning(
                "flashinfer ulysses: could not register %s (%s); staying on NCCL "
                "for this shape",
                key,
                error,
            )
            return None
        self._pairs[key] = (send, out)
        return self._pairs[key]

    # -------------------------------------------------------------- input --
    def qkv_buffer(self, tokens: int, heads: int, head_dim: int, dtype):
        """The projection's destination, which is also the exchange's source."""
        pair = self._pair(
            "qkv", (1, tokens, heads * 3, head_dim), "scatter_heads", dtype
        )
        if pair is None:
            return None
        # The projection wants [tokens, heads, 3, head_dim] and the exchange
        # wants [1, tokens, 3 * heads, head_dim]: same memory, same order.
        return pair[0].view(tokens, heads, 3, head_dim)

    def exchange_input(self, qkv: torch.Tensor):
        """``[T_local, H, 3, D]`` -> three ``[T_global, H/ws, D]`` views."""
        tokens, heads, three, head_dim = qkv.shape
        assert three == 3, f"expected [T, H, 3, D], got {tuple(qkv.shape)}"
        pair = self._pair(
            "qkv", (1, tokens, heads * 3, head_dim), "scatter_heads", qkv.dtype
        )
        if pair is None:
            return None
        exchanged = self._comm.scatter_heads(
            qkv.view(1, tokens, heads * 3, head_dim), out=pair[1]
        )
        merged = exchanged.view(
            tokens * self.world_size, heads // self.world_size, 3, head_dim
        )
        return merged[:, :, 0], merged[:, :, 1], merged[:, :, 2]

    # ------------------------------------------------------------- output --
    def attn_out_buffer(self, tokens: int, heads: int, head_dim: int, dtype):
        """Attention's destination, which is also the exchange's source."""
        pair = self._pair(
            "attn_out", (1, tokens, heads, head_dim), "gather_heads", dtype
        )
        if pair is None:
            return None
        return pair[0].view(tokens, heads, head_dim)

    def exchange_output(self, attn_out: torch.Tensor):
        """``[T_global, H/ws, D]`` -> ``[T_local, H, D]``, head axis merged."""
        tokens, heads, head_dim = attn_out.shape
        pair = self._pair(
            "attn_out", (1, tokens, heads, head_dim), "gather_heads", attn_out.dtype
        )
        if pair is None:
            return None
        exchanged = self._comm.gather_heads(
            attn_out.view(1, tokens, heads, head_dim), out=pair[1]
        )
        return exchanged.view(
            tokens // self.world_size, heads * self.world_size, head_dim
        )

    # --------------------------------------------------------------- life --
    def shutdown(self) -> None:
        """Collective: every rank must call this before any rank exits."""
        self._pairs.clear()
        self._comm.close()


def get_flashinfer_ulysses_a2a(
    group, device: torch.device, dtype: torch.dtype, capacity: int
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
        if reason != "disabled" and name not in _TRANSPORTS:
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
            group, max_elems=capacity, dtype=dtype, backend="pcie", device=device
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
        logger.warning(
            "flashinfer ulysses could not start on some rank (%s); staying on NCCL",
            error,
        )
        return None

    transport = FlashInferUlyssesA2A(
        comm,
        world_size,
        envs.SGLANG_DIFFUSION_MINIMAX_H3_ULYSSES_MAX_SHAPES,
        capacity,
    )
    _TRANSPORTS[name] = transport
    logger.info(
        "flashinfer ulysses is carrying this group's exchanges "
        "(backend=%s, transport=%s, world_size=%d, capacity=%d)",
        transport.backend,
        transport.transport,
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
