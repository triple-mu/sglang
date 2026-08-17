# SPDX-License-Identifier: Apache-2.0
"""fast-ulysses transport for the MiniMax-H3 Ulysses exchange.

What this replaces is not only NCCL's ``all_to_all_single`` but the relayout on
either side of it. NCCL needs a contiguous destination-major send buffer, so
the input exchange has to build one and the output exchange has to merge the
head axis afterwards. fast-ulysses folds both into its copy strides and the
NIC's MKey layout: the projection's own ``[tokens, heads, 3, head_dim]`` buffer
is sent as it stands, and so is attention's output.

Two of its constraints shape everything here:

- **An output workspace is permanently bound to one input allocation.** The
  mlx5 backend caches the input memory region, and the MKeys that gather from
  it, on the bare pointer; a moved input would gather whatever now lives there.
  Hence the persistent buffers below, which the projection and attention write
  into instead of allocating their own.
- **Allocating a workspace is collective.** Every rank must reach it with the
  same shape in the same order. Every gate here is therefore a function of
  values that are identical on every rank -- world size, token count, dtype --
  so the ranks never disagree about which transport an exchange uses.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# Distinct token counts that get their own buffers and workspaces. Nothing is
# ever released -- fast-ulysses holds every registration for the group's
# lifetime -- so a server that sees unbounded shapes has to stop somewhere.
_MAX_SHAPES = 4

_TRANSPORTS: dict[str, "FastUlyssesA2A | None"] = {}


def _unusable(world_size: int, dtype: torch.dtype) -> str | None:
    """Why this exchange cannot use fast-ulysses, or None if it can.

    Only the conditions that exist before a group does. Everything that depends
    on the shape or the transport is asked of the group itself -- a limit
    transcribed here would be a copy the library cannot see, and copies drift.
    """
    if not envs.SGLANG_DIFFUSION_MINIMAX_H3_FAST_ULYSSES:
        return "disabled"
    try:
        import fast_ulysses
    except ImportError as error:
        return f"fast-ulysses is not installed ({error})"
    if world_size < 2 or not fast_ulysses.supports_world_size(world_size):
        return (
            f"world size {world_size} is not one of "
            f"{tuple(fast_ulysses.SUPPORTED_WORLD_SIZES)}"
        )
    if not fast_ulysses.supports_dtype(dtype):
        return f"dtype {dtype} is unsupported"
    if torch.is_grad_enabled():
        return "autograd is enabled"
    if torch.compiler.is_compiling():
        return "running under torch.compile"
    if torch.cuda.is_current_stream_capturing():
        return "running under CUDA graph capture"
    return None


class FastUlyssesA2A:
    """One fast-ulysses group, plus the persistent buffers its API requires."""

    def __init__(self, native) -> None:
        self._group = native
        self.device = native.device
        self.backend = self._group.backend
        self.world_size = self._group.world_size
        self._stream = self._group.stream
        self._buffers: dict[tuple, torch.Tensor] = {}
        self._workspaces: dict[tuple, torch.Tensor] = {}
        self._budget_spent = False

    def _over_budget(self, what: str, shape) -> None:
        if not self._budget_spent:
            self._budget_spent = True
            logger.warning(
                "fast-ulysses has its %d shapes; %s %s and every later new "
                "shape fall back to NCCL",
                _MAX_SHAPES,
                what,
                shape,
            )
        return None

    def _buffer(self, role: str, shape: tuple[int, ...], dtype) -> torch.Tensor | None:
        key = (role, shape, dtype)
        buffer = self._buffers.get(key)
        if buffer is None:
            if len(self._buffers) >= 2 * _MAX_SHAPES:
                return self._over_budget(role, shape)
            buffer = torch.empty(shape, dtype=dtype, device=self.device)
            self._buffers[key] = buffer
        return buffer

    def _workspace(self, mode: int, source: torch.Tensor) -> torch.Tensor | None:
        key = (mode, tuple(source.shape), source.dtype)
        workspace = self._workspaces.get(key)
        if workspace is None:
            if len(self._workspaces) >= 2 * _MAX_SHAPES:
                return self._over_budget(f"mode {mode}", tuple(source.shape))
            # Collective: every rank allocates the same shape in the same order.
            workspace = self._group.allocate_output(source, mode)
            self._workspaces[key] = workspace
        return workspace

    def _exchangeable(self, mode: int, shape: tuple[int, ...], dtype) -> bool:
        """Whether the group will carry this exchange, asked rather than assumed.

        The limits differ by transport -- the 16-bit MKey stride is mlx5's and
        only mlx5's -- so a caller that hardcodes them refuses shapes p2p can
        carry. The answer is rank-invariant, so branching on it is symmetric.
        """
        reason = self._group.unsupported_reason(shape, dtype, mode)
        if reason is None:
            return True
        if not self._budget_spent:
            logger.info("fast-ulysses declines %s mode=%d: %s", shape, mode, reason)
        return False

    # ---------------------------------------------------------------- input --
    def qkv_buffer(
        self, tokens: int, heads: int, head_dim: int, dtype
    ) -> torch.Tensor | None:
        """The projection's destination, which is also the exchange's source."""
        if not self._exchangeable(0, (1, tokens, heads * 3, head_dim), dtype):
            return None
        return self._buffer("qkv", (tokens, heads, 3, head_dim), dtype)

    def exchange_input(
        self, qkv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """``[T_local, H, 3, D]`` -> three ``[T_global, H/ws, D]`` views.

        The head axis carries ``3 * H`` entries, so splitting it into world_size
        contiguous blocks hands each rank ``H/ws`` heads with their own Q, K and
        V beside them. Nothing is packed going in and nothing is copied out.
        """
        tokens, heads, three, head_dim = qkv.shape
        assert three == 3, f"expected [T, H, 3, D], got {tuple(qkv.shape)}"
        source = qkv.view(1, tokens, heads * 3, head_dim)
        workspace = self._workspace(0, source)
        if workspace is None:
            return None
        exchanged = self._group.all_to_all_4d(source, mode=0, out=workspace)
        merged = exchanged.view(
            tokens * self.world_size, heads // self.world_size, 3, head_dim
        )
        return merged[:, :, 0], merged[:, :, 1], merged[:, :, 2]

    # --------------------------------------------------------------- output --
    def attn_out_buffer(
        self, tokens: int, heads: int, head_dim: int, dtype
    ) -> torch.Tensor | None:
        """Attention's destination, which is also the exchange's source."""
        if not self._exchangeable(1, (1, tokens, heads, head_dim), dtype):
            return None
        return self._buffer("attn_out", (tokens, heads, head_dim), dtype)

    def exchange_output(self, attn_out: torch.Tensor) -> torch.Tensor | None:
        """``[T_global, H/ws, D]`` -> ``[T_local, H, D]``, head axis merged."""
        tokens, heads, head_dim = attn_out.shape
        source = attn_out.view(1, tokens, heads, head_dim)
        workspace = self._workspace(1, source)
        if workspace is None:
            return None
        exchanged = self._group.all_to_all_4d(source, mode=1, out=workspace)
        return exchanged.view(
            tokens // self.world_size, heads * self.world_size, head_dim
        )

    # ----------------------------------------------------------------- life --
    def on_bound_stream(self) -> bool:
        """fast-ulysses refuses any stream but the one bound at construction."""
        return (
            torch.cuda.current_stream(self.device).cuda_stream
            == self._stream.cuda_stream
        )

    def shutdown(self) -> None:
        self._workspaces.clear()
        self._buffers.clear()
        self._group.destroy()


def get_fast_ulysses_a2a(group, device: torch.device, dtype) -> FastUlyssesA2A | None:
    """The transport for ``group``, or None with the reason logged once.

    Construction is collective, so every rank of the group must reach this with
    the same arguments. They do: the caller only reaches it when Ulysses is
    active, and every gate above is rank-invariant.
    """
    reason = _unusable(dist.get_world_size(group), dtype)
    name = getattr(group, "group_name", None) or str(id(group))
    if reason is not None:
        if reason != "disabled" and name not in _TRANSPORTS:
            _TRANSPORTS[name] = None
            logger.info("fast-ulysses is unusable (%s); staying on NCCL", reason)
        return None

    if name in _TRANSPORTS:
        transport = _TRANSPORTS[name]
        # A different stream would make fast-ulysses raise. The stream is the
        # same on every rank, so falling back on it stays symmetric.
        if transport is not None and not transport.on_bound_stream():
            return None
        return transport

    from fast_ulysses import UlyssesGroup

    # create(), not the constructor: mlx5 setup fails per rank rather than per
    # job, and `except: use NCCL` around a constructor is only correct when the
    # failure is symmetric. create() agrees the outcome before anyone commits,
    # so None here means None on every rank.
    native = UlyssesGroup.create(process_group=group, device=device)
    if native is None:
        _TRANSPORTS[name] = None
        logger.warning("fast-ulysses could not start on some rank; staying on NCCL")
        return None
    transport = FastUlyssesA2A(native)
    _TRANSPORTS[name] = transport
    logger.info(
        "fast-ulysses is carrying the Ulysses exchange (backend=%s, world_size=%d)",
        transport.backend,
        transport.world_size,
    )
    return transport


def shutdown_fast_ulysses_a2a() -> None:
    """Collective: every rank must call this before any rank exits."""
    for name in list(_TRANSPORTS):
        transport = _TRANSPORTS.pop(name)
        if transport is not None:
            transport.shutdown()
