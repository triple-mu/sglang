# SPDX-License-Identifier: Apache-2.0
"""SubBlock sparse attention on the SM120 Sage kernel, fed by a quantised Ulysses exchange.

MiniMax-H3 on SM120 (RTX PRO Blackwell) only. Per DiT layer and denoise step:

1. every rank quantises its own Q/K/V shard once -- INT8 Q and K, FP8 V -- after
   one small fp32 all-gather of the K channel sums and V channel maxima the
   scales depend on;
2. the INT8/FP8 bytes go through the Ulysses all-to-all instead of BF16, so the
   exchange moves about half the bytes;
3. the receiver rebuilds the Sage operands for its heads, ranks key blocks with
   the SubBlock router on dequantised copies, and runs the vendored Cake
   block-sparse Sage kernel. The dense warmup steps keep every key block and go
   through the same kernel, so the BF16 exchange is never needed.

The schedule keys of ``subblock_sparse_attn`` (``sparsity``, ``skip_first_steps``,
``skip_first_layers``, ``n_k``, ``n_q``, ``min_seq_len``) apply unchanged;
``compute_mode`` is fixed. ``lowp_a2a=false`` keeps the BF16 all-to-all and
quantises after it with the same kernels: an ablation knob, not a fallback.
``skip_first_steps`` beyond the step count gives dense Sage attention over the
quantised exchange.

The route is decided from rank-invariant inputs only, so every rank takes the
same branch and no collective is left waiting. Calls the route declines (ring,
non-H3 packed layouts, shards that are not 128-row aligned) run the BF16
exchange and then the local quantisation; layers outside the DiT stack run the
dense backend.
"""

from __future__ import annotations

import threading
from collections import Counter

import msgspec
import torch
import torch.distributed as dist

from sglang.kernels.ops.diffusion import (
    sage_block_sparse_attn_sm120,
    sage_block_sparse_dense_block_index,
    subblock_block_tables,
    subblock_pool_int8,
    ulysses_lowp_finalize_stats,
    ulysses_lowp_k_sum_v_amax,
    ulysses_lowp_payload_spec,
    ulysses_lowp_quant_pack,
    ulysses_lowp_scale_widths,
    ulysses_lowp_unpack_for_sage,
)
from sglang.kernels.ops.diffusion.quantization.ulysses_lowp_sage_jit import (
    SHARD_ALIGNMENT,
    UlyssesLowpSpec,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.subblock_sparse import (
    SubBlockRouter,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.subblock_sparse.router import (
    LOG2E,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.subblock_sparse_attn import (
    SUBBLOCK_SPARSE_BLOCK_SIZE,
    SUBBLOCK_SPARSE_HEAD_DIM,
    SubBlockSparseAttentionImpl,
    SubBlockSparseAttentionMetadata,
    SubBlockSparseAttentionMetadataBuilder,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

BLOCK = SUBBLOCK_SPARSE_BLOCK_SIZE
HEAD_DIM = SUBBLOCK_SPARSE_HEAD_DIM

_COUNTERS: Counter[str] = Counter()
_COUNTERS_LOCK = threading.Lock()


def reset_lowp_counters() -> None:
    with _COUNTERS_LOCK:
        _COUNTERS.clear()


def get_lowp_counters() -> dict[str, int]:
    """Route counters since the last reset: `lowp_a2a_total` and `bf16_a2a_total{reason}`."""
    with _COUNTERS_LOCK:
        return dict(_COUNTERS)


def _count(key: str) -> None:
    with _COUNTERS_LOCK:
        _COUNTERS[key] += 1


def h3_live_rows(
    cu_seqlens_host: tuple[int, ...] | None, max_seqlen: int, total: int
) -> int | None:
    """`used` for H3's packed layout `(0, used, total)` with a zero tail, else None."""
    if cu_seqlens_host is None or len(cu_seqlens_host) != 3:
        return None
    start, used, stop = (int(x) for x in cu_seqlens_host)
    if start != 0 or stop != total or not 0 < used <= total or int(max_seqlen) != used:
        return None
    return used


def lowp_route_reason(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens_host: tuple[int, ...] | None,
    max_seqlen: int,
    ring_active: bool,
    world_size: int,
    enabled: bool,
) -> str | None:
    """Why the quantised exchange cannot run, or None.

    Every input is rank-invariant (equal shards, shared cu_seqlens, one config),
    so all ranks reach the same decision without a collective.
    """
    if not enabled:
        return "disabled"
    if world_size < 2:
        return "not_ulysses"
    if ring_active:
        return "ring"
    if any(t.dtype is not torch.bfloat16 for t in (q, k, v)):
        return "dtype"
    if (
        q.ndim != 3
        or q.shape != k.shape
        or q.shape != v.shape
        or q.shape[-1] != HEAD_DIM
    ):
        return "shape"
    if q.shape[0] % SHARD_ALIGNMENT or q.shape[1] % world_size:
        return "alignment"
    if any(t.stride(-1) != 1 for t in (q, k, v)):
        return "stride"
    if h3_live_rows(cu_seqlens_host, max_seqlen, q.shape[0] * world_size) is None:
        return "packed_layout"
    if not q.is_cuda:
        return "device"
    if torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing():
        return "compile_or_capture"
    return None


def cake_block_tables(
    index: torch.Tensor,
    topk: int,
    num_blocks: int,
    sparse_query_block_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`(q2k_block_index, q2k_block_nums)` for the Cake kernel.

    Query blocks the mask excludes keep every key block, so text and audio rows
    get exact attention inside the same launch.
    """
    batch, heads, q_blocks, _ = index.shape
    if sparse_query_block_mask is None:
        nums = torch.full(
            (batch, heads, q_blocks), topk, dtype=torch.int32, device=index.device
        )
        return index.contiguous(), nums
    mask = sparse_query_block_mask.to(device=index.device, dtype=torch.bool).reshape(-1)
    if mask.numel() != q_blocks:
        raise ValueError(
            f"sparse query-block mask has {mask.numel()} entries for {q_blocks} query blocks"
        )
    if index.is_cuda:
        return subblock_block_tables(index, mask, num_blocks)
    # CPU callers (tests) build the same tables with torch ops.
    tables = (
        torch.arange(num_blocks, device=index.device, dtype=index.dtype)
        .view(1, 1, 1, num_blocks)
        .expand(batch, heads, q_blocks, num_blocks)
        .clone()
    )
    tables[..., :topk] = torch.where(
        mask.view(1, 1, q_blocks, 1), index, tables[..., :topk]
    )
    nums = (
        torch.where(mask.view(1, 1, q_blocks), topk, num_blocks)
        .expand(batch, heads, q_blocks)
        .to(torch.int32)
        .contiguous()
    )
    return tables, nums


class _SageOperands(msgspec.Struct, frozen=True):
    """Cake kernel inputs for this rank's heads; V is `[1, h, 128, S]` in Sage's 16-token order."""

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    q_scale: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor


def _unpack(
    recv: torch.Tensor,
    spec: UlyssesLowpSpec,
    *,
    used: int,
    v_scale: torch.Tensor,
    device: torch.device,
) -> _SageOperands:
    """Rebuild the kernel operands; rows at or past `used` come out as zeros."""
    from sglang.multimodal_gen.runtime.layers.usp import _a2a_staging_buffer

    cut = -(-used // BLOCK) * BLOCK
    q_width, k_width = ulysses_lowp_scale_widths(cut)
    h, s = spec.local_heads, spec.global_sequence
    out = (
        _a2a_staging_buffer("sage_lowp_q", (1, h, s, HEAD_DIM), torch.int8, device),
        _a2a_staging_buffer("sage_lowp_k", (1, h, s, HEAD_DIM), torch.int8, device),
        _a2a_staging_buffer(
            "sage_lowp_v", (1, h, HEAD_DIM, s), torch.float8_e4m3fn, device
        ),
        _a2a_staging_buffer(
            "sage_lowp_q_scale", (1, h, q_width), torch.float32, device
        ),
        _a2a_staging_buffer(
            "sage_lowp_k_scale", (1, h, k_width), torch.float32, device
        ),
    )
    ulysses_lowp_unpack_for_sage(
        recv, spec=spec, scale_sequence=cut, used_sequence=used, out=out
    )
    return _SageOperands(*out, v_scale=v_scale.contiguous())


def _quantize_local(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, used: int
) -> _SageOperands:
    """Quantise a whole sequence held by one rank with the exchange kernels at world size one."""
    from sglang.multimodal_gen.runtime.layers.usp import _a2a_staging_buffer

    total, heads, _ = q.shape
    if total % SHARD_ALIGNMENT:
        raise ValueError(
            f"the packed sequence must be a multiple of {SHARD_ALIGNMENT} rows for "
            f"the Sage quantiser, got {total}"
        )
    spec = ulysses_lowp_payload_spec(
        batch=1, local_sequence=total, num_heads=heads, world_size=1
    )
    q4, k4, v4 = (t.unsqueeze(0) for t in (q, k, v))
    stats = ulysses_lowp_k_sum_v_amax(k4, v4)
    k_mean, v_scale = ulysses_lowp_finalize_stats(
        stats, world_size=1, used_sequence=used, dtype=q.dtype
    )
    payload = _a2a_staging_buffer(
        "sage_lowp_send", spec.payload_shape, torch.uint8, q.device
    )
    ulysses_lowp_quant_pack(
        q4,
        k4,
        v4,
        k_mean,
        v_scale,
        rank=0,
        world_size=1,
        used_sequence=used,
        out=payload,
    )
    return _unpack(payload, spec, used=used, v_scale=v_scale, device=q.device)


class SubBlockSparseSageSM120Impl(SubBlockSparseAttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool = False,
        softmax_scale: float | None = None,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        super().__init__(
            num_heads,
            head_size,
            causal=causal,
            softmax_scale=softmax_scale,
            num_kv_heads=num_kv_heads,
            prefix=prefix,
            **extra_impl_args,
        )
        from sglang.multimodal_gen.runtime.server_args import get_global_server_args

        config = get_global_server_args().attention_backend_config or {}
        self.lowp_a2a = bool(config.get("lowp_a2a", True))
        if self.layer_enabled:
            # Cake takes a per-row block count, so the budget needs no rounding.
            self.router = SubBlockRouter(
                n_k=self.schedule.n_k,
                n_q=self.schedule.n_q,
                block_size_k=BLOCK,
                budget_granularity=1,
            )

    def _sage_ready(self, query: torch.Tensor, used: int) -> bool:
        return (
            self.layer_enabled
            and not self.causal
            and query.dtype is torch.bfloat16
            and query.shape[-1] == HEAD_DIM
            and used >= self.schedule.min_seq_len
        )

    def forward(self, query, key, value, attn_metadata=None) -> torch.Tensor:
        """`[B, S, H, D]` attention outside the DiT (the video VAE reaches every backend): dense."""
        return self.dense_impl.forward(query, key, value, attn_metadata)

    def forward_varlen(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        cu_seqlens_host: tuple[int, ...] | None = None,
        first_segment_sparse_query_block_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Packed `[T, H, D]` rows holding the whole sequence: one rank, or after a BF16 all-to-all."""
        total = query.shape[0]
        used = h3_live_rows(cu_seqlens_host, max_seqlen, total)
        if used is None or not self._sage_ready(query, used):
            return self.dense_impl.forward_varlen(
                query,
                key,
                value,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                cu_seqlens_host=cu_seqlens_host,
            )
        operands = _quantize_local(query, key, value, used=used)
        return self._attend(
            operands,
            used=used,
            total=total,
            sparse_query_block_mask=first_segment_sparse_query_block_mask,
        )

    def forward_ulysses_lowp(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        cu_seqlens_host: tuple[int, ...] | None,
        max_seqlen: int,
        ring_active: bool,
        sparse_query_block_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Quantise this rank's shard, exchange INT8/FP8 bytes, attend for this rank's heads.

        Returns `[S_global, H_local, D]`, or None when the caller must run the
        BF16 exchange instead.
        """
        from sglang.multimodal_gen.runtime.distributed.device_communicators.rdma_ulysses_a2a import (
            active_rdma_ulysses_a2a,
        )
        from sglang.multimodal_gen.runtime.distributed.parallel_state import (
            get_sp_group,
            get_ulysses_parallel_rank,
            get_ulysses_parallel_world_size,
        )
        from sglang.multimodal_gen.runtime.layers.usp import (
            _a2a_staging_buffer,
            _usp_all_to_all_single,
        )

        world_size = get_ulysses_parallel_world_size()
        local_sequence, heads, _ = query.shape
        reason = lowp_route_reason(
            query,
            key,
            value,
            cu_seqlens_host=cu_seqlens_host,
            max_seqlen=max_seqlen,
            ring_active=ring_active,
            world_size=world_size,
            enabled=self.lowp_a2a,
        )
        used = h3_live_rows(cu_seqlens_host, max_seqlen, local_sequence * world_size)
        if reason is None and not self._sage_ready(query, used):
            reason = "dense_layer"
        if reason is not None:
            _count(f"bf16_a2a_total{{{reason}}}")
            return None
        _count("lowp_a2a_total")

        rank = get_ulysses_parallel_rank()
        device = query.device
        group = get_sp_group().ulysses_group
        transport = active_rdma_ulysses_a2a(group)
        spec = ulysses_lowp_payload_spec(
            batch=1,
            local_sequence=local_sequence,
            num_heads=heads,
            world_size=world_size,
        )
        q4, k4, v4 = (t.unsqueeze(0) for t in (query, key, value))
        stats = ulysses_lowp_k_sum_v_amax(k4, v4)
        if transport is not None:
            gathered = transport.exchange_stats(stats)
        else:
            gathered = _a2a_staging_buffer(
                "sage_lowp_stats", (world_size, *stats.shape), torch.float32, device
            )
            dist.all_gather_into_tensor(gathered, stats, group=group)
        k_mean, v_scale = ulysses_lowp_finalize_stats(
            gathered, world_size=world_size, used_sequence=used, dtype=query.dtype
        )
        # The RDMA transport reads its own landing buffer, so pack straight into it.
        send = (
            transport.packed_input_buffer(spec.payload_shape)
            if transport is not None
            else _a2a_staging_buffer(
                "sage_lowp_send", spec.payload_shape, torch.uint8, device
            )
        )
        ulysses_lowp_quant_pack(
            q4,
            k4,
            v4,
            k_mean,
            v_scale,
            rank=rank,
            world_size=world_size,
            used_sequence=used,
            out=send,
        )
        recv = (
            transport.exchange_chunks(send)
            if transport is not None
            else _usp_all_to_all_single(send, role="sage_lowp_recv")
        )
        h = spec.local_heads
        operands = _unpack(
            recv,
            spec,
            used=used,
            v_scale=v_scale[:, rank * h : (rank + 1) * h],
            device=device,
        )
        logger.info_once(
            f"SubBlock Sage SM120: quantised Ulysses exchange active (L={local_sequence}, "
            f"S={spec.global_sequence}, {spec.chunk_bytes * world_size} uint8 bytes per rank)"
        )
        return self._attend(
            operands,
            used=used,
            total=spec.global_sequence,
            sparse_query_block_mask=sparse_query_block_mask,
        )

    def gather_output(self, out: torch.Tensor) -> torch.Tensor:
        """Ulysses output all-to-all of `[S_global, H_local, D]` -> `[S_local, H, D]`."""
        from sglang.multimodal_gen.runtime.distributed.device_communicators.rdma_ulysses_a2a import (
            active_rdma_ulysses_a2a,
        )
        from sglang.multimodal_gen.runtime.distributed.parallel_state import (
            get_sp_group,
        )
        from sglang.multimodal_gen.runtime.layers.usp import _usp_output_all_to_all

        transport = active_rdma_ulysses_a2a(get_sp_group().ulysses_group)
        if transport is None:
            return _usp_output_all_to_all(out[None], head_dim=2)[0]
        landing = transport.gather_landing(tuple(out.shape), out.dtype)
        landing.copy_(out)
        return transport.gather_heads(landing)

    def _attend(
        self,
        op: _SageOperands,
        *,
        used: int,
        total: int,
        sparse_query_block_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Route (or keep every block) and run the Cake kernel; returns `[total, h, D]`."""
        from sglang.multimodal_gen.runtime.layers.usp import _a2a_staging_buffer

        cut = -(-used // BLOCK) * BLOCK
        heads = op.q.shape[1]
        num_blocks = cut // BLOCK
        if self._step_enabled():
            # Q's per-group scale and K's channel mean shift every key block of a
            # query row alike, so only K's per-tile scale enters the ranking.
            gq, gk, sub_q, sub_k = self.router.cell_geometry(used, used)
            pooled_q, pooled_k = subblock_pool_int8(
                op.q,
                op.k,
                op.k_scale,
                used=used,
                sub_q=sub_q,
                sub_k=sub_k,
                cells_q=gq * self.router.n_q,
                cells_k=gk * self.router.n_k,
                q_factor=self.softmax_scale * LOG2E,
            )
            plan = self.router.route_pooled(
                pooled_q,
                pooled_k,
                batch=1,
                seq_q=used,
                seq_k=used,
                sparsity=self.schedule.sparsity,
            )
            index, nums = cake_block_tables(
                plan.index, plan.topk, num_blocks, sparse_query_block_mask
            )
            logger.info_once(
                f"SubBlock Sage SM120 sparse attention active: S={used} heads={heads} "
                f"keeping {plan.topk}/{num_blocks} key blocks per selected query block, "
                f"dense for the first {self.schedule.skip_first_steps} denoise steps"
            )
        else:
            index, nums = sage_block_sparse_dense_block_index(
                1, heads, cut, cut, op.q.device
            )
        out = _a2a_staging_buffer(
            "sage_lowp_out", (1, heads, total, HEAD_DIM), torch.bfloat16, op.q.device
        )
        sage_block_sparse_attn_sm120(
            op.q,
            op.k,
            op.v,
            op.q_scale,
            op.k_scale,
            op.v_scale,
            index,
            nums,
            out=out,
            seqlen_q=cut,
            seqlen_k=cut,
            softmax_scale=self.softmax_scale,
        )
        if used < total:
            out[:, :, used:] = 0
        return out[0].transpose(0, 1)


class SubBlockSparseSageSM120Backend(AttentionBackend):
    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [HEAD_DIM]

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.SUBBLOCK_SPARSE_SAGE_SM120

    @staticmethod
    def get_impl_cls() -> type[SubBlockSparseSageSM120Impl]:
        return SubBlockSparseSageSM120Impl

    @staticmethod
    def get_metadata_cls() -> type[SubBlockSparseAttentionMetadata]:
        return SubBlockSparseAttentionMetadata

    @staticmethod
    def get_builder_cls() -> type[SubBlockSparseAttentionMetadataBuilder]:
        return SubBlockSparseAttentionMetadataBuilder


__all__ = [
    "SubBlockSparseSageSM120Backend",
    "SubBlockSparseSageSM120Impl",
    "cake_block_tables",
    "get_lowp_counters",
    "h3_live_rows",
    "lowp_route_reason",
    "reset_lowp_counters",
]
