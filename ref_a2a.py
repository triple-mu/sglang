from __future__ import annotations
import torch
import torch.distributed as dist

__all__ = ["all_to_all_4d"]


def _offsets(splits: list[int]) -> list[int]:
    out, acc = [], 0
    for s in splits:
        out.append(acc)
        acc += s
    return out


def _even(total: int, n: int) -> list[int]:
    """The split the caller gets when they pass none: the remainder goes to the first ranks."""
    q, r = divmod(total, n)
    return [q + 1] * r + [q] * (n - r)


def all_to_all_4d(
    x: torch.Tensor,
    *,
    mode: int = 0,
    group: dist.ProcessGroup | None = None,
    seq_splits: list[int] | None = None,
    head_splits: list[int] | None = None,
) -> torch.Tensor:
    """mode 0 scatters heads and gathers sequence; mode 1 inverts it.
    Args:
        x: 4D CUDA tensor. mode 0 takes ``(b, s_local, n_global, d)``; mode 1 takes
            ``(b, s_global, n_local, d)``.
        mode: 0 or 1.
        group: the process group; ``None`` uses ``dist.group.WORLD``.
        seq_splits: rank p's sequence shard. ``None`` derives an even split of the axis this call
            can see, which is only possible in the direction where that axis is global.
        head_splits: rank p's head count, same rule.
    Returns:
        A new tensor. Collective over ``group``; every rank must call it with the same shapes.
    """
    pg = group if group is not None else dist.group.WORLD
    ws = dist.get_world_size(pg)
    rank = dist.get_rank(pg)
    if x.dim() != 4:
        raise ValueError(f"expected a 4D tensor, got {tuple(x.shape)}")
    if mode not in (0, 1):
        raise ValueError(f"mode must be 0 or 1, got {mode}")
    x = x.contiguous()
    b, d = x.shape[0], x.shape[-1]
    # A direction can derive the split of the axis it holds WHOLE. The other axis it sees only as
    # this rank's shard, so the default there is "every rank holds the same" -- which is the even
    # case. Uneven shards must be passed, because no rank can infer another's.
    if mode == 0:
        head_splits = head_splits or _even(x.shape[2], ws)  # n_global is whole here
        seq_splits = seq_splits or [x.shape[1]] * ws
    else:
        seq_splits = seq_splits or _even(x.shape[1], ws)  # s_global is whole here
        head_splits = head_splits or [x.shape[2]] * ws
    if len(seq_splits) != ws or len(head_splits) != ws:
        raise ValueError(
            f"splits must have {ws} entries, got {len(seq_splits)} and {len(head_splits)}"
        )
    n_local, s_local = head_splits[rank], seq_splits[rank]
    s_global, n_global = sum(seq_splits), sum(head_splits)
    ho, so = _offsets(head_splits), _offsets(seq_splits)
    if mode == 0:
        if tuple(x.shape) != (b, s_local, n_global, d):
            raise ValueError(
                f"mode 0 expects (b, {s_local}, {n_global}, d), got {tuple(x.shape)}"
            )
        in_splits = [b * s_local * head_splits[p] * d for p in range(ws)]
        out_splits = [b * seq_splits[p] * n_local * d for p in range(ws)]
        # THE ONE COPY. Even head splits make it a single permute kernel; uneven ones make it a
        # gather of ws contiguous blocks, which is the same traffic.
        if len(set(head_splits)) == 1:
            send = (
                x.view(b, s_local, ws, head_splits[0], d)
                .permute(2, 0, 1, 3, 4)
                .reshape(-1)
            )
        else:
            send = torch.cat(
                [
                    x[:, :, ho[p] : ho[p] + head_splits[p], :].reshape(-1)
                    for p in range(ws)
                ]
            )
        recv = torch.empty(sum(out_splits), dtype=x.dtype, device=x.device)
        dist.all_to_all_single(recv, send, out_splits, in_splits, group=pg)
        if b == 1:
            return recv.view(
                b, s_global, n_local, d
            )  # free: the blocks already stack along s
        out = torch.empty(b, s_global, n_local, d, dtype=x.dtype, device=x.device)
        off = 0
        for p in range(ws):
            out[:, so[p] : so[p] + seq_splits[p]] = recv[
                off : off + out_splits[p]
            ].view(b, seq_splits[p], n_local, d)
            off += out_splits[p]
        return out
    if tuple(x.shape) != (b, s_global, n_local, d):
        raise ValueError(
            f"mode 1 expects (b, {s_global}, {n_local}, d), got {tuple(x.shape)}"
        )
    in_splits = [b * seq_splits[p] * n_local * d for p in range(ws)]
    out_splits = [b * s_local * head_splits[p] * d for p in range(ws)]
    if b == 1:
        send = x.reshape(-1)  # free: sequence is already the outer axis
    else:
        send = torch.cat(
            [x[:, so[p] : so[p] + seq_splits[p]].reshape(-1) for p in range(ws)]
        )
    recv = torch.empty(sum(out_splits), dtype=x.dtype, device=x.device)
    dist.all_to_all_single(recv, send, out_splits, in_splits, group=pg)
    # THE ONE COPY: interleave the source ranks on the head axis.
    if len(set(head_splits)) == 1:
        return (
            recv.view(ws, b, s_local, head_splits[0], d)
            .permute(1, 2, 0, 3, 4)
            .reshape(b, s_local, n_global, d)
        )
    out = torch.empty(b, s_local, n_global, d, dtype=x.dtype, device=x.device)
    off = 0
    for p in range(ws):
        out[:, :, ho[p] : ho[p] + head_splits[p]] = recv[
            off : off + out_splits[p]
        ].view(b, s_local, head_splits[p], d)
        off += out_splits[p]
    return out
