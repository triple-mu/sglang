"""Low-precision Ulysses all-to-all for the SM120 Sage block-sparse kernel.

The sender quantises its Q/K/V shard once (INT8 Q with a scale per 32 tokens,
channel-mean-centred INT8 K with a scale per 64 tokens, per-channel FP8 V) and
packs it destination-major, so the all-to-all moves about half the bytes of
the BF16 exchange and the receiver hands the chunks to the attention kernel
without dequantising. Only two per-channel statistics need the whole sequence
(the K mean and the V amax); they go through one small fp32 all-gather that
the caller runs:

    send = ulysses_lowp_k_sum_v_amax(k, v)                    # [2, B, H, 128] fp32
    gathered = all_gather(send)                                # [W, 2, B, H, 128]
    k_mean, v_scale = ulysses_lowp_finalize_stats(gathered, world_size=W,
                                                  used_sequence=used, dtype=q.dtype)
    ulysses_lowp_quant_pack(q, k, v, k_mean, v_scale, rank=r, world_size=W,
                            used_sequence=used, out=payload)   # [W, chunk_bytes] uint8
    recv = all_to_all(payload)                                 # or the RDMA transport
    ulysses_lowp_unpack_for_sage(recv, spec=spec, scale_sequence=cut,
                                 used_sequence=used, out=(q8, k8, v8, qs, ks))

Contract (local grid, ALIGN-128): every rank's shard `[B, L, H, 128]` has
`L % 128 == 0` and `H % world_size == 0`, so no quantisation group straddles a
rank and the per-group scales computed on the shard are final. Q/K/V are
accepted as strided views (dense head_dim, 16-byte-aligned strides), which
admits the fused projection's `[T, H, 3, D]` slices. Rows at or beyond
`used_sequence` (zero padding of the packed sequence) are kept out of the
group amax and come out of the unpack as zeros.

The arithmetic follows flashinfer's `ulysses_lowp` port of the SageAttention
quantiser and is compiled with `--use_fast_math` like the original; against a
plain torch reference the INT8 values agree to within one step on a small
fraction of elements and the FP8 values within one ulp.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import msgspec
import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

Q_GROUP = 32
K_GROUP = 64
HEAD_DIM = 128
SHARD_ALIGNMENT = 128
V_SCALE_MAX = 2.25
_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)
_REQUIRED_CAPABILITY = (12, 0)


class UlyssesLowpSpec(msgspec.Struct, frozen=True):
    """Byte layout of the destination-major payload for one shard geometry."""

    batch: int
    local_sequence: int
    num_heads: int
    local_heads: int
    world_size: int
    main_bytes: int
    q_scale_offset: int
    k_scale_offset: int
    chunk_bytes: int

    @property
    def global_sequence(self) -> int:
        return self.local_sequence * self.world_size

    @property
    def payload_shape(self) -> tuple[int, int]:
        return (self.world_size, self.chunk_bytes)


def ulysses_lowp_payload_spec(
    *, batch: int, local_sequence: int, num_heads: int, world_size: int
) -> UlyssesLowpSpec:
    """Mirror of `chunk_spec` in `ulysses_lowp_sage.cuh`."""
    if local_sequence <= 0 or local_sequence % SHARD_ALIGNMENT:
        raise ValueError(
            f"local_sequence must be a positive multiple of {SHARD_ALIGNMENT}, got {local_sequence}"
        )
    if world_size < 1 or num_heads % world_size:
        raise ValueError(
            f"num_heads {num_heads} must split evenly over world_size {world_size}"
        )
    local_heads = num_heads // world_size
    main_bytes = batch * local_sequence * local_heads * HEAD_DIM
    q_scale_offset = 3 * main_bytes
    k_scale_offset = (
        q_scale_offset + batch * local_heads * (local_sequence // Q_GROUP) * 4
    )
    raw = k_scale_offset + batch * local_heads * (local_sequence // K_GROUP) * 4
    return UlyssesLowpSpec(
        batch=batch,
        local_sequence=local_sequence,
        num_heads=num_heads,
        local_heads=local_heads,
        world_size=world_size,
        main_bytes=main_bytes,
        q_scale_offset=q_scale_offset,
        k_scale_offset=k_scale_offset,
        chunk_bytes=(raw + 127) // 128 * 128,
    )


def ulysses_lowp_scale_widths(rows: int) -> tuple[int, int]:
    """Q / K scale slots the Sage kernel derives from `rows` live tokens."""
    return (rows + 127) // 128 * 4, (rows + 63) // 64


def can_use_ulysses_lowp_sage(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, world_size: int
) -> bool:
    """Cheap, rank-invariant gate; the launcher checks the tensors in full."""
    if not (q.is_cuda and torch.version.hip is None):
        return False
    if tuple(torch.cuda.get_device_capability(q.device)) != _REQUIRED_CAPABILITY:
        return False
    if q.dtype not in _SUPPORTED_DTYPES or q.dtype != k.dtype or q.dtype != v.dtype:
        return False
    if q.dim() != 4 or q.shape != k.shape or q.shape != v.shape:
        return False
    batch, local_sequence, heads, head_dim = q.shape
    return (
        head_dim == HEAD_DIM
        and local_sequence % SHARD_ALIGNMENT == 0
        and world_size >= 1
        and heads % world_size == 0
        and all(t.stride(-1) == 1 for t in (q, k, v))
    )


@cache_once
def _jit_module(dtype: torch.dtype, use_pdl: bool) -> Module:
    if dtype not in _SUPPORTED_DTYPES:
        raise RuntimeError(f"ulysses_lowp_sage: unsupported dtype {dtype}")
    if (
        torch.version.hip is not None
        or tuple(torch.cuda.get_device_capability()) != _REQUIRED_CAPABILITY
    ):
        raise RuntimeError("ulysses_lowp_sage targets compute capability 12.0 only")
    args = make_cpp_args(dtype, use_pdl)
    return load_jit(
        "diffusion_ulysses_lowp_sage",
        *args,
        cuda_files=["diffusion/ulysses_lowp_sage.cuh"],
        cuda_wrappers=[
            ("k_sum_v_amax", f"ulysses_lowp_sage::Kernels<{args}>::k_sum_v_amax"),
            ("quant_pack", f"ulysses_lowp_sage::Kernels<{args}>::quant_pack"),
            ("unpack_for_sage", f"ulysses_lowp_sage::Kernels<{args}>::unpack_for_sage"),
        ],
        # The Sage quantiser's rounding chain was pinned under fast math.
        extra_cuda_cflags=["--use_fast_math"],
    )


def _module(dtype: torch.dtype) -> Module:
    return _jit_module(dtype, is_arch_support_pdl())


def ulysses_lowp_k_sum_v_amax(k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Per-channel K sum and V amax of this shard as one `[2, B, H, 128]` fp32 tensor.

    That tensor is the all-gather payload; it is identical in size on every rank.
    """
    batch, _, heads, head_dim = k.shape
    stats = torch.empty(2, batch, heads, head_dim, dtype=torch.float32, device=k.device)
    _module(k.dtype).k_sum_v_amax(k, v, stats[0], stats[1])
    return stats


def ulysses_lowp_finalize_stats(
    gathered: torch.Tensor,
    *,
    world_size: int,
    used_sequence: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Turn the gathered `[W, 2, B, H, 128]` statistics into `k_mean` and `v_scale`.

    The K mean divides by the live row count (padding rows contribute exactly
    zero to the sum) and is rounded to the activation dtype, which is the
    value the pack kernel subtracts. Reductions run in fixed rank order, so
    every rank derives bit-identical results.
    """
    g = gathered.view(world_size, 2, *gathered.shape[-3:])
    k_mean = (g[:, 0].sum(dim=0) / used_sequence).to(dtype).contiguous()
    v_scale = (g[:, 1].amax(dim=0) / V_SCALE_MAX).contiguous()
    return k_mean, v_scale


def ulysses_lowp_quant_pack(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_mean: torch.Tensor,
    v_scale: torch.Tensor,
    *,
    rank: int,
    world_size: int,
    used_sequence: int,
    out: torch.Tensor,
) -> torch.Tensor:
    """Quantise the shard and write the destination-major payload into `out`."""
    _module(q.dtype).quant_pack(
        q, k, v, k_mean, v_scale, out, int(rank), int(world_size), int(used_sequence)
    )
    return out


def ulysses_lowp_unpack_for_sage(
    recv: torch.Tensor,
    *,
    spec: UlyssesLowpSpec,
    scale_sequence: int,
    used_sequence: int,
    out: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rebuild `(q_int8, k_int8, v_fp8, q_scale, k_scale)` in the Sage kernel's layout.

    Q/K are `[B, h, S, 128]`, V is `[B, h, 128, S]`, and the scales are sized by
    `ulysses_lowp_scale_widths(scale_sequence)`; `recv` may carry any dtype
    view of the payload bytes, the module is chosen by the shard dtype the
    caller packed with (the unpack itself moves bytes).
    """
    q, k, v, q_scale, k_scale = out
    _module(torch.bfloat16).unpack_for_sage(
        recv,
        q,
        k,
        v,
        q_scale,
        k_scale,
        int(spec.local_sequence),
        int(spec.world_size),
        int(scale_sequence),
        int(used_sequence),
    )
    return out


__all__ = [
    "HEAD_DIM",
    "K_GROUP",
    "Q_GROUP",
    "SHARD_ALIGNMENT",
    "UlyssesLowpSpec",
    "V_SCALE_MAX",
    "can_use_ulysses_lowp_sage",
    "ulysses_lowp_finalize_stats",
    "ulysses_lowp_k_sum_v_amax",
    "ulysses_lowp_payload_spec",
    "ulysses_lowp_quant_pack",
    "ulysses_lowp_scale_widths",
    "ulysses_lowp_unpack_for_sage",
]
