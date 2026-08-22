# SPDX-License-Identifier: Apache-2.0
"""FlashInfer PCIe transport for the MiniMax-H3 Ulysses exchange.

On a single node without NVLink, ``all_to_all_single`` is not the bottleneck the
relayout around it is -- it is the link. FlashInfer's PCIe Ulysses backend moves
the payload with copy engines rather than SM stores, and at world size 8 it
routes the cross-NUMA half over each rank's local mlx5 NIC while the same-NUMA
half stays on CUDA P2P.

Two properties of that backend shape this file.

- **Its layout is the one this model already uses.** ``scatter_heads`` is
  defined as ``out_r[b, j*S + s, hl, d] = x_j[b, s, r*Hl + hl, d]``, which is
  what ``_usp_input_all_to_all_packed_qkv`` computes; ``gather_heads`` is the
  matching inverse of ``_usp_output_all_to_all(head_dim=2)``. So this is a
  transport swap, not a layout change, and the results are bit-identical. Q/K/V
  stay three separate exchanges: the backend is link-bound, so fusing them into
  one operand buys it nothing and costs a pack.
- **Registration is collective and permanent.** Constructing the communicator
  and allocating an output both require every rank to arrive together, and an
  allocation lives until ``close()``. Every gate below is therefore a function
  of values that are identical on every rank -- world size, token count, dtype
  -- so ranks never disagree about which transport an exchange uses.

Outputs are registered buffers, reused by the next call at the same shape. A
caller must consume them within the layer, which the attention core does.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# FlashInfer's PCIe backend, single node.
_SUPPORTED_WORLD_SIZES = (2, 4, 8)
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)

_TRANSPORTS: dict[str, "FlashInferUlyssesA2A | None"] = {}


def _unusable(world_size: int, dtype: torch.dtype) -> str | None:
    """Why this exchange cannot use the PCIe backend, or None if it can.

    Only conditions that exist before a group does, and only ones every rank
    evaluates identically. Anything shape-dependent is checked per call.
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
    """One communicator plus its registered output buffers, per shape."""

    def __init__(self, comm, world_size: int, max_shapes: int, max_elems: int) -> None:
        self._comm = comm
        self.world_size = world_size
        self._max_shapes = max_shapes
        self.max_elems = max_elems
        # (tokens, heads, head_dim) -> (sends, scattered, gather_out)
        self._buffers: dict[tuple[int, int, int], tuple] = {}
        self._declined: set[tuple[int, int, int]] = set()

    @property
    def backend(self) -> str:
        return self._comm.backend

    @property
    def transport(self) -> str | None:
        return self._comm.transport

    def _slots(self, key: tuple[int, int, int], sample: torch.Tensor):
        """Send buffers and registered outputs for one shape, on first sight.

        The send buffers exist for two reasons. Q/K/V arrive as strided views
        into the fused projection output -- ``v`` always, ``q`` and ``k``
        whenever the fused norm/RoPE kernel writes in place -- and the backend
        needs contiguous operands. And a registered output binds the input
        memory region to the pointer it was registered with, so a moving input
        would fall back to a staged copy on the hybrid route for good. One copy
        into a fixed buffer buys both, and it is the copy the NCCL path already
        makes to build its destination-major send buffer.

        Allocation is collective, and the key is derived from values identical
        on every rank, so every rank allocates the same shapes in the same
        order or none of them.
        """
        cached = self._buffers.get(key)
        if cached is not None:
            return cached
        if key in self._declined:
            return None
        if len(self._buffers) >= self._max_shapes:
            self._declined.add(key)
            logger.info(
                "flashinfer ulysses: %d registered shapes already, staying on "
                "NCCL for %s",
                self._max_shapes,
                key,
            )
            return None
        try:
            sends = [torch.empty_like(sample) for _ in range(3)]
            scattered = [
                self._comm.allocate_output(send, "scatter_heads") for send in sends
            ]
            gathered = self._comm.allocate_output(scattered[0], "gather_heads")
        except Exception as error:  # noqa: BLE001
            # Allocation is collective and its failure mode is collective too,
            # so declining here declines on every rank.
            self._declined.add(key)
            logger.warning(
                "flashinfer ulysses: could not register outputs for %s (%s); "
                "staying on NCCL for this shape",
                key,
                error,
            )
            return None
        self._buffers[key] = (sends, scattered, gathered)
        return self._buffers[key]

    def exchange_input(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """``[T_local, H, D]`` x3 -> ``[T_global, H/ws, D]`` x3, or None."""
        tokens, heads, head_dim = q.shape
        if heads % self.world_size:
            return None
        key = (tokens, heads, head_dim)
        slots = self._slots(key, q[None])
        if slots is None:
            return None
        sends, scattered, _ = slots
        for send, tensor in zip(sends, (q, k, v)):
            send.copy_(tensor[None])
        return tuple(
            self._comm.scatter_heads(send, out=out)[0]
            for send, out in zip(sends, scattered)
        )

    def exchange_output(self, attn_out: torch.Tensor, key: tuple[int, int, int]):
        """``[T_global, H/ws, D]`` -> ``[T_local, H, D]``, or None.

        ``attn_out`` is whatever the attention backend allocated, so on the
        hybrid route this exchange takes the staged input path unless the
        caching allocator happens to hand back the registered pointer. That
        costs one device-to-device copy of the operand, which is why the send
        buffers above are worth their own copy and this one is not.
        """
        slots = self._buffers.get(key)
        if slots is None:
            return None
        return self._comm.gather_heads(attn_out[None], out=slots[2])[0]

    def shutdown(self) -> None:
        """Collective: every rank must call this before any rank exits."""
        self._buffers.clear()
        self._comm.close()


def get_flashinfer_ulysses_a2a(
    group,
    device: torch.device,
    dtype: torch.dtype,
    needed: int,
    capacity: int,
) -> FlashInferUlyssesA2A | None:
    """The transport for this process group, constructing it once.

    ``capacity`` is the declared maximum and sizes every registration;
    ``needed`` is this operand. An operand larger than the capacity a group was
    built with stays on NCCL. It does not rebuild: capacity cannot be raised in
    place, and every other communication workspace in both trees declines
    rather than re-registering -- measured here, rebuilding cost more than the
    fallback it was avoiding.

    Over-declaring costs device memory and one cold-path registration. It costs
    nothing per exchange: the transport derives its copy widths and RDMA
    payload from the call's own geometry, and reads the capacity only to
    allocate, to register, and to bounds-check.
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
        cached = _TRANSPORTS[name]
        if cached is None or needed <= cached.max_elems:
            return cached
        logger.warning(
            "flashinfer ulysses: %d elements exceeds the declared capacity of "
            "%d, so this shape stays on NCCL. Set "
            "SGLANG_DIFFUSION_MINIMAX_H3_ULYSSES_MAX_SEQ_LEN to the largest "
            "packed sequence this deployment serves.",
            needed,
            cached.max_elems,
        )
        return None

    max_elems = max(capacity, needed)

    from flashinfer.comm import UlyssesCommunicator

    # The constructor raises rather than returning None, and FlashInfer makes
    # topology failures collective. Vote anyway: a rank-local failure that it
    # did not classify as collective would otherwise leave some ranks on the
    # PCIe path and some on NCCL, which is a hang rather than a slowdown.
    comm = None
    error: BaseException | None = None
    try:
        comm = UlyssesCommunicator(
            group,
            max_elems=max_elems,
            dtype=dtype,
            backend="pcie",
            device=device,
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
        max_elems,
    )
    _TRANSPORTS[name] = transport
    logger.info(
        "flashinfer ulysses is carrying this group's exchanges "
        "(backend=%s, transport=%s, world_size=%d)",
        transport.backend,
        transport.transport,
        world_size,
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
