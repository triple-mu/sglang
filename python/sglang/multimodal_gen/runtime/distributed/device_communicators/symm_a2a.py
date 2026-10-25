# SPDX-License-Identifier: Apache-2.0
"""Copy-engine Ulysses all-to-all over symmetric memory, any group size on one host.

Every rank's workspace is mapped into every peer's address space (torch
symmetric memory), so the seq<->head relayout IS the transfer: one
``cuMemcpy2DAsync`` per peer reads this rank's block straight out of the
caller's (possibly strided) tensor and lands it at its final position in the
peer's output slot. No pack or unpack kernels, no staging copy, and no SMs.
Synchronization is a one-element NCCL all-reduce on the Ulysses group before
the copies (every rank's reader of the previous exchange has retired in stream
order) and after them (every peer's copies into my slots have landed). PCIe
Blackwell has peer access but no native peer atomics, so there is no flag or
CAS protocol here.

Four fixed output slots: q, k, v and the attention output. A slot's view
aliases the next exchange into that slot, so a caller consumes it before then.
"""

import logging
import math

import torch
import torch.distributed as dist

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.distributed.device_communicators.ipc_a2a import (
    _Unsupported,
)
from sglang.multimodal_gen.runtime.distributed.device_communicators.ipc_a2a_multi import (
    _member_devices,
    _StreamMemOps,
)

logger = logging.getLogger(__name__)

# q, k, v and the attention output
SLOTS = 4
_ALIGN = 256


def _layout(x, world, rank, split_dim):
    """``cuMemcpy2DAsync`` parameters of one ``[b, s, n, d]`` exchange, in bytes.

    Returns ``(out_shape, (src_peer_step, src_pitch, src_batch_step),
    (dst_offset, dst_pitch, dst_batch_step), width, height)``: peer ``p``'s
    block of batch row ``bb`` is ``height`` rows of ``width`` bytes, read from
    ``x.data_ptr() + p * src_peer_step + bb * src_batch_step`` with pitch
    ``src_pitch`` and written at ``slot + dst_offset + bb * dst_batch_step``
    with pitch ``dst_pitch`` in ``p``'s workspace. ``ValueError`` means the
    copy engine cannot express this layout. Validity depends on shape, stride
    and dtype only, so every rank reaches the same verdict (rank enters the
    offsets alone and can never make one rank fall through by itself).
    """
    if x.ndim != 4 or x.numel() == 0 or x.shape[split_dim] % world:
        raise ValueError("needs a non-empty [b, s, n, d] tensor divisible on the split dim")
    b, s, n, d = x.shape
    st, es = x.stride(), x.element_size()
    if st[3] != 1 or st[2] != d or min(st) < 0:
        raise ValueError("needs contiguous [heads, dim] rows")
    if (s > 1 and st[1] < n * d) or (b > 1 and st[0] < (s - 1) * st[1] + n * d):
        raise ValueError("overlapping or broadcast rows are unsupported")
    pitch, batch = max(st[1], n * d) * es, st[0] * es
    if split_dim == 2:
        # seq_to_head: [b, s, n, d] -> [b, s * world, n / world, d]
        row = n // world * d * es
        return (
            (b, s * world, n // world, d),
            (row, pitch, batch),
            (rank * s * row, row, s * world * row),
            row,
            s,
        )
    # head_to_seq: [b, s, n, d] -> [b, s / world, n * world, d]
    row, s_local = n * d * es, s // world
    return (
        (b, s_local, n * world, d),
        (s_local * pitch, pitch, batch),
        (rank * row, world * row, s_local * world * row),
        row,
        s_local,
    )


class SymmA2AState:
    def __init__(self):
        self.group = None
        self.rank = None
        self.world = 0
        self.inited = False
        self.failed = False
        self.memops = None
        self.signal = None
        self.workspace = None
        self.handle = None
        self.peers = []
        # bytes per slot
        self.stride = 0
        # smallest message that did not fit on some rank; agreed by all
        self.declined = 0
        # layout rejections already logged
        self.skipped = set()
        self.calls = 0

    def reset(self) -> None:
        """Drop the workspace of a model-parallel group being replaced."""
        self._release()
        self.__init__()

    def drop_staging(self) -> None:
        """Release the workspace (every rank calls this at the same point); the
        next exchange allocates one sized for its own messages."""
        self._release()

    def _release(self) -> None:
        if self.workspace is None:
            return
        # the trailing barrier of the last exchange ordered every peer's copies
        # before this stream; the host sync retires them before the unmap
        torch.cuda.synchronize()
        self.handle = self.workspace = None
        self.peers = []
        self.stride = 0

    def init(self, group) -> None:
        self.rank = dist.get_rank(group=group)
        self.world = dist.get_world_size(group=group)
        self.group = group
        dev = torch.cuda.current_device()
        devices = _member_devices(group, dev)
        peers = [d for member, d in enumerate(devices) if member != self.rank]
        missing = [d for d in peers if not torch.cuda.can_device_access_peer(dev, d)]
        error = memops = None
        try:
            if dist.get_backend(group) != "nccl":
                raise _Unsupported("the barrier needs an NCCL Ulysses group")
            memops = _StreamMemOps()
        except _Unsupported as e:
            error = e
        # agree before the collectives below: a rank that bailed out alone
        # would leave the others blocked inside them
        usable = torch.tensor(
            [int(not missing and error is None)], dtype=torch.int32, device="cuda"
        )
        dist.all_reduce(usable, op=dist.ReduceOp.MIN, group=group)
        if usable.item() == 0:
            if missing:
                raise _Unsupported(f"CUDA device {dev} cannot reach peers {missing}")
            raise _Unsupported(str(error) if error else "a Ulysses peer is unsupported")
        self.memops = memops
        self.signal = torch.ones(1, dtype=torch.int32, device="cuda")
        # warm the communicator before any capture
        self._barrier()
        self.inited = True

    def _barrier(self) -> None:
        # ponytail: two of these per exchange (~30 us each on PCIe); a
        # cuStreamWriteValue32/WaitValue32 flag protocol over _StreamMemOps is
        # the upgrade if they show up in a profile
        dist.all_reduce(self.signal, op=dist.ReduceOp.MAX, group=self.group)

    def _fits_on_every_rank(self, nbytes: int) -> bool:
        """Whether every rank can spare twice `nbytes`; a group collective."""
        free, _ = torch.cuda.mem_get_info()
        cached = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
        fits = torch.tensor(
            [int(free + cached >= 2 * nbytes)], dtype=torch.int32, device="cuda"
        )
        dist.all_reduce(fits, op=dist.ReduceOp.MIN, group=self.group)
        return bool(fits.item())

    def _ensure(self, nbytes: int) -> bool:
        """A workspace whose slots hold `nbytes`; False when none can be had now."""
        if nbytes <= self.stride:
            return True
        if torch.cuda.is_current_stream_capturing():
            return False
        if self.declined and nbytes >= self.declined:
            return False
        stride = -(-nbytes // _ALIGN) * _ALIGN
        if not self._fits_on_every_rank(SLOTS * stride):
            logger.info(
                "symmetric-memory all-to-all workspace (%.2f GiB) does not fit on "
                "every rank; using NCCL for messages of %d bytes and up",
                SLOTS * stride / 2**30,
                nbytes,
            )
            self.declined = nbytes
            return False
        import torch.distributed._symmetric_memory as symm_mem

        # the previous exchange's trailing barrier retired every copy into the
        # old workspace on every rank
        self._release()
        # symmetric memory is allocated outside the caching allocator
        torch.cuda.empty_cache()
        self.workspace = symm_mem.empty(SLOTS * stride, dtype=torch.uint8, device="cuda")
        self.handle = symm_mem.rendezvous(self.workspace, self.group)
        self.peers = [ptr + self.handle.offset for ptr in self.handle.buffer_ptrs]
        self.stride = stride
        logger.info(
            "Ulysses all-to-all over symmetric memory on the copy engine: "
            "%d ranks, %d slots x %.0f MiB",
            self.world,
            SLOTS,
            stride / 2**20,
        )
        return True

    def _view(self, slot: int, shape, dtype: torch.dtype) -> torch.Tensor:
        nbytes = math.prod(shape) * dtype.itemsize
        return self.workspace.narrow(0, slot * self.stride, nbytes).view(dtype).view(shape)

    def exchange(self, tensors, slots, split_dim: int):
        """One all-to-all per tensor into its slot, under a single barrier pair.

        ``split_dim=2`` scatters heads and gathers the sequence
        (``[b, s_local, H, D] -> [b, s_global, H / world, D]``); ``split_dim=1``
        is the inverse. Returns views of the slots, or None, before any
        collective, when this call cannot use the transport (the caller keeps
        its NCCL path). Inference only.
        """
        try:
            plans = [_layout(x, self.world, self.rank, split_dim) for x in tensors]
        except ValueError as e:
            if str(e) not in self.skipped:
                self.skipped.add(str(e))
                logger.info("symmetric-memory all-to-all skipped (%s); using NCCL", e)
            return None
        if torch.is_grad_enabled() and any(x.requires_grad for x in tensors):
            return None
        if not self._ensure(max(x.nbytes for x in tensors)):
            return None
        lo = self.workspace.data_ptr()
        hi = lo + SLOTS * self.stride
        if any(lo <= x.data_ptr() < hi for x in tensors):
            # an input that lives in a slot would be overwritten mid-copy
            return None
        stream = torch.cuda.current_stream()
        copy2d, rank, world = self.memops.copy2d, self.rank, self.world
        self._barrier()
        for step in range(world):
            # my own block first, then rank + 1, rank + 2, ...: every step is a
            # permutation, so no destination takes two senders at once
            peer = (rank + step) % world
            for x, slot, (_, src, dst, width, height) in zip(tensors, slots, plans):
                dst_ptr = self.peers[peer] + slot * self.stride + dst[0]
                src_ptr = x.data_ptr() + peer * src[0]
                for bb in range(x.shape[0]):
                    copy2d(
                        stream,
                        dst_ptr + bb * dst[2],
                        dst[1],
                        src_ptr + bb * src[2],
                        src[1],
                        width,
                        height,
                    )
        self._barrier()
        self.calls += 1
        return tuple(
            self._view(slot, plan[0], x.dtype)
            for x, slot, plan in zip(tensors, slots, plans)
        )


SYMM_A2A = SymmA2AState()


def symm_a2a_ready(group) -> bool:
    """True when the transport is enabled and initialized on every member of
    `group` (initializes lazily on the first eager call, never under capture)."""
    from sglang.multimodal_gen.runtime.platforms import current_platform

    if not envs.SGLANG_DIFFUSION_SYMM_A2A:
        return False
    if SYMM_A2A.group is not None and SYMM_A2A.group is not group:
        SYMM_A2A.reset()
    if SYMM_A2A.failed:
        return False
    if SYMM_A2A.inited:
        return True
    if not current_platform.is_cuda() or torch.cuda.is_current_stream_capturing():
        return False
    ok = True
    try:
        SYMM_A2A.init(group)
    except _Unsupported as e:
        logger.info("symmetric-memory all-to-all unavailable (%s); using NCCL", e)
        ok = False
    except Exception as e:
        logger.debug("symmetric-memory all-to-all initialization failed", exc_info=True)
        logger.warning("symmetric-memory all-to-all unavailable (%s); using NCCL", e)
        ok = False
    # every member must take the same path, or a rank waits at a barrier nobody joins
    agreed = torch.tensor([int(ok)], dtype=torch.int32, device="cuda")
    dist.all_reduce(agreed, op=dist.ReduceOp.MIN, group=group)
    if agreed.item() == 0:
        SYMM_A2A.reset()
        SYMM_A2A.failed = True
        return False
    return True
