"""Pure-torch reference for the SageAttention SM120 operand contract.

Mirrors flashinfer's Triton ``quantize_sage_qkv_sm120`` (the producer the
SM120 Sage block-sparse kernels were written against) so kernel tests can build
operands and check dequantised results without a GPU-specific quantiser:

* Q: INT8, one scale per 32-query group (``amax / 127``, amax floored at 1e-7),
  ``q_scale`` laid out ``[B, H, ceil(S_q / 128) * 4]``.
* K: channel mean over the live rows (kept in bf16, as the producer does)
  subtracted, then INT8 with one scale per 64-key block.
* V: FP8 E4M3, one scale per channel (``amax / 2.25``), stored
  ``[B, H, 128, S_k_padded]`` with the 16-token physical permutation the FP8 PV
  MMA expects; padding rows are zero.
"""

from __future__ import annotations

import torch

Q_GROUP = 32
Q_TILE = 128
K_BLOCK = 64
HEAD_DIM = 128
V_SCALE_MAX = 2.25
_AMAX_FLOOR = 1.0e-7


def sage_v_physical_rows(seqlen_padded: int, device: torch.device) -> torch.Tensor:
    """Logical token ``t`` of a 64-block lands at ``physical_row[t]`` in the V layout."""
    t = torch.arange(seqlen_padded, device=device)
    local = t % K_BLOCK
    mod = local % 16
    return (
        (t // K_BLOCK) * K_BLOCK
        + (local // 16) * 16
        + (mod // 8) * 2
        + ((mod // 2) % 4) * 4
        + mod % 2
    )


def quantize_sage_q(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``q`` bf16/fp32 ``[B, H, S, 128]`` -> (int8 ``[B, H, S, 128]``, fp32 scales)."""
    batch, heads, seqlen, dim = q.shape
    assert dim == HEAD_DIM
    groups = -(-seqlen // Q_TILE) * (Q_TILE // Q_GROUP)
    padded = groups * Q_GROUP
    qf = torch.zeros(batch, heads, padded, dim, dtype=torch.float32, device=q.device)
    qf[:, :, :seqlen] = q.float()
    grouped = qf.view(batch, heads, groups, Q_GROUP, dim)
    amax = grouped.abs().amax(dim=(-1, -2)).clamp_min(_AMAX_FLOOR)
    scale = amax / 127.0
    q_int8 = (
        torch.round(grouped / scale[..., None, None])
        .clamp_(-127.0, 127.0)
        .to(torch.int8)
        .view(batch, heads, padded, dim)[:, :, :seqlen]
        .contiguous()
    )
    return q_int8, scale.contiguous()


def quantize_sage_kv(
    k: torch.Tensor, v: torch.Tensor, *, used_rows: int | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``k``/``v`` ``[B, H, S, 128]`` -> (k_int8 ``[B,H,S,128]``, v_fp8 ``[B,H,128,S_pad]``,
    k_scale ``[B,H,S_pad/64]``, v_scale ``[B,H,128]``).

    ``used_rows`` (default ``S``) is the live prefix: rows beyond it are treated
    as padding (excluded from the statistics, quantised to zero).
    """
    batch, heads, seqlen, dim = k.shape
    assert dim == HEAD_DIM and v.shape == k.shape
    used = seqlen if used_rows is None else used_rows
    blocks = -(-seqlen // K_BLOCK)
    padded = blocks * K_BLOCK
    valid = torch.arange(padded, device=k.device) < used

    kf = torch.zeros(batch, heads, padded, dim, dtype=torch.float32, device=k.device)
    vf = torch.zeros_like(kf)
    kf[:, :, :seqlen] = k.float()
    vf[:, :, :seqlen] = v.float()
    kf = kf * valid[None, None, :, None]
    vf = vf * valid[None, None, :, None]

    k_mean = (kf.sum(dim=2) / used).to(torch.bfloat16).float()
    v_scale = (vf.abs().amax(dim=2).clamp_min(_AMAX_FLOOR) / V_SCALE_MAX).contiguous()

    k_centered = (kf - k_mean[:, :, None, :]) * valid[None, None, :, None]
    blocked = k_centered.view(batch, heads, blocks, K_BLOCK, dim)
    k_amax = blocked.abs().amax(dim=(-1, -2)).clamp_min(_AMAX_FLOOR)
    k_scale = (k_amax / 127.0).contiguous()
    k_int8 = (
        torch.round(blocked / k_scale[..., None, None])
        .clamp_(-127.0, 127.0)
        .to(torch.int8)
        .view(batch, heads, padded, dim)[:, :, :seqlen]
        .contiguous()
    )

    v_quant = (vf / v_scale[:, :, None, :]).to(torch.float8_e4m3fn)
    v_fp8 = torch.zeros(
        batch, heads, dim, padded, dtype=torch.float8_e4m3fn, device=k.device
    )
    v_fp8[:, :, :, sage_v_physical_rows(padded, k.device)] = v_quant.transpose(2, 3)
    return k_int8, v_fp8.contiguous(), k_scale, v_scale


def dequantize_sage(
    q_int8: torch.Tensor,
    k_int8: torch.Tensor,
    v_fp8: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    *,
    seqlen_q: int,
    seqlen_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """fp32 ``[B, H, S, 128]`` operands the kernel effectively computes with.

    K comes back mean-centred: the kernel never adds the mean back, and softmax
    is invariant to the per-query constant it contributes.
    """
    batch, heads = q_int8.shape[:2]
    q = q_int8[:, :, :seqlen_q].float()
    qs = q_scale.repeat_interleave(Q_GROUP, dim=-1)[:, :, :seqlen_q]
    q = q * qs[..., None]
    k = k_int8[:, :, :seqlen_k].float()
    ks = k_scale.repeat_interleave(K_BLOCK, dim=-1)[:, :, :seqlen_k]
    k = k * ks[..., None]
    rows = sage_v_physical_rows(v_fp8.shape[-1], v_fp8.device)[:seqlen_k]
    v = v_fp8[:, :, :, rows].float().transpose(2, 3) * v_scale[:, :, None, :]
    return q, k, v


def block_sparse_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_block_index: torch.Tensor,
    q2k_block_nums: torch.Tensor,
    *,
    softmax_scale: float,
) -> torch.Tensor:
    """fp32 masked attention over 64-token blocks; rows with zero kept blocks give zeros."""
    batch, heads, seqlen_q, _ = q.shape
    seqlen_k = k.shape[2]
    q_blocks = -(-seqlen_q // K_BLOCK)
    k_blocks = seqlen_k // K_BLOCK
    keep = torch.zeros(
        batch, heads, q_blocks, k_blocks, dtype=torch.bool, device=q.device
    )
    capacity = q2k_block_index.shape[-1]
    slot_live = (
        torch.arange(capacity, device=q.device)[None, None, None, :]
        < q2k_block_nums[..., None]
    )
    index = q2k_block_index.long().clamp(0, k_blocks - 1)
    keep.scatter_(-1, index.masked_fill(~slot_live, 0), slot_live)
    # A row that keeps nothing must not wrongly pick up block 0 from the fill.
    keep[..., 0] &= (slot_live & (index == 0)).any(dim=-1)
    mask = keep.repeat_interleave(K_BLOCK, dim=2).repeat_interleave(K_BLOCK, dim=3)
    mask = mask[:, :, :seqlen_q, :seqlen_k]
    scores = torch.einsum("bhqd,bhkd->bhqk", q, k) * softmax_scale
    scores = scores.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0)
    return torch.einsum("bhqk,bhkd->bhqd", probs, v)
