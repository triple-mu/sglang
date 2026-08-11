"""Time MiniMax-H3's two Ulysses collectives against fast-ulysses.

The point of comparison is the path H3 actually runs today -- the packed-QKV
input exchange and the output exchange out of ``usp.py``, with their Triton
pack and JIT merge kernels -- not a generic ``all_to_all_single``. Each is
split into relayout and transfer so the trace-level question ("how much of the
collective is data movement the caller only pays because of the layout")
has a number here too.

Usage:
    NCCL_NVLS_ENABLE=0 torchrun --nproc_per_node=<ws> bench_usp_a2a.py [--steps N]
"""

from __future__ import annotations

import argparse
import os
import statistics

import torch
import torch.distributed as dist

# H3's DiT: 50 blocks, 56 heads, head_dim 128, bf16.
H3_NUM_HEADS = 56
H3_HEAD_DIM = 128
H3_NUM_LAYERS = 50
# 1344x768, 5s at 24fps, plus the 227-token text tail, rounded to H3's
# packed-sequence alignment of 64.
H3_SEQ_LEN = 38051


def _median_ms(fn, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        # Every rank must enter the collective, so the barrier goes before the
        # timed region rather than inside it.
        dist.barrier()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def _align_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=25)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=H3_SEQ_LEN)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        maybe_init_distributed_environment_and_model_parallel,
    )

    world_size = int(os.environ["WORLD_SIZE"])
    maybe_init_distributed_environment_and_model_parallel(
        tp_size=1,
        sp_size=world_size,
        ulysses_degree=world_size,
        ring_degree=1,
    )

    from sglang.multimodal_gen.runtime.layers.usp import (
        _usp_input_all_to_all_packed_qkv,
        _usp_output_all_to_all,
    )

    rank = dist.get_rank()
    device = torch.device("cuda", local_rank)
    dtype = torch.bfloat16

    # sglang tail-pads the sequence so every rank holds the same shard.
    seq_global = _align_up(args.seq_len, world_size)
    seq_local = seq_global // world_size
    h_local = H3_NUM_HEADS // world_size

    q, k, v = (
        torch.randn(seq_local, H3_NUM_HEADS, H3_HEAD_DIM, dtype=dtype, device=device)
        for _ in range(3)
    )
    attn_out = torch.randn(
        1, seq_global, h_local, H3_HEAD_DIM, dtype=dtype, device=device
    )

    results: dict[str, float] = {}

    results["sglang_in"] = _median_ms(
        lambda: _usp_input_all_to_all_packed_qkv(q, k, v), args.iters, args.warmup
    )
    results["sglang_out"] = _median_ms(
        lambda: _usp_output_all_to_all(attn_out, head_dim=2), args.iters, args.warmup
    )

    # Break sglang's two calls into relayout vs transfer, which is the split
    # the end-to-end trace question is really about.
    from sglang.kernels.ops.diffusion.triton.ulysses_qkv import (
        pack_qkv_destination_major,
    )
    from sglang.kernels.ops.diffusion.usp_relayout import usp_merge_heads
    from sglang.multimodal_gen.runtime.layers.usp import (
        _a2a_staging_buffer,
        _usp_all_to_all_single,
    )

    pack_dst = torch.empty(
        (world_size, seq_local, h_local, 3 * H3_HEAD_DIM), dtype=dtype, device=device
    )
    results["sglang_in_relayout"] = _median_ms(
        lambda: pack_qkv_destination_major(q, k, v, world_size, out=pack_dst),
        args.iters,
        args.warmup,
    )
    results["sglang_in_transfer"] = _median_ms(
        lambda: _usp_all_to_all_single(pack_dst, role="bench_in"),
        args.iters,
        args.warmup,
    )

    out_permuted = attn_out.permute(1, 0, 2, 3).contiguous()
    results["sglang_out_relayout_pre"] = _median_ms(
        lambda: attn_out.permute(1, 0, 2, 3).contiguous(), args.iters, args.warmup
    )
    results["sglang_out_transfer"] = _median_ms(
        lambda: _usp_all_to_all_single(out_permuted, role="bench_out"),
        args.iters,
        args.warmup,
    )
    merge_src = out_permuted.reshape(world_size, seq_local, 1, h_local, H3_HEAD_DIM)
    results["sglang_out_relayout_post"] = _median_ms(
        lambda: usp_merge_heads(merge_src), args.iters, args.warmup
    )

    # fast-ulysses takes the same exchange as one 4D call: q/k/v concatenated
    # on the last axis is exactly the packing sglang expresses as a shape.
    from fast_ulysses import UlyssesGroup

    from sglang.multimodal_gen.runtime.distributed.parallel_state import get_sp_group

    # Bind to the same process group usp.py exchanges over, so this measures
    # what an in-tree integration would.
    group = UlyssesGroup(process_group=get_sp_group().ulysses_group)
    qkv = torch.cat([q, k, v], dim=-1).unsqueeze(0).contiguous()

    def _fu_in() -> None:
        packed = torch.cat([q, k, v], dim=-1).unsqueeze(0)
        out = group.all_to_all_4d(packed, mode=0)
        out.split(H3_HEAD_DIM, dim=-1)

    def _fu_in_precat() -> None:
        # The cat hoisted out: what the call costs if the QKV projection can be
        # made to emit this layout directly.
        group.all_to_all_4d(qkv, mode=0)

    results["fu_in"] = _median_ms(_fu_in, args.iters, args.warmup)
    results["fu_in_precat"] = _median_ms(_fu_in_precat, args.iters, args.warmup)
    results["fu_out"] = _median_ms(
        lambda: group.all_to_all_4d(attn_out, mode=1), args.iters, args.warmup
    )

    # `out=` writes straight into the symmetric window, removing the copy-out.
    # Collective, so the buffers are allocated once, outside the timed loop.
    fu_in_buf = group.empty_output(qkv, mode=0)
    fu_out_buf = group.empty_output(attn_out, mode=1)
    results["fu_in_zerocopy"] = _median_ms(
        lambda: group.all_to_all_4d(qkv, mode=0, out=fu_in_buf),
        args.iters,
        args.warmup,
    )
    results["fu_out_zerocopy"] = _median_ms(
        lambda: group.all_to_all_4d(attn_out, mode=1, out=fu_out_buf),
        args.iters,
        args.warmup,
    )

    # The stage split, straight from the operator.
    _, stages_in = group._timed(qkv, mode=0)
    _, stages_out = group._timed(attn_out, mode=1)

    if rank == 0:
        mb_in = qkv.numel() * qkv.element_size() / 1e6
        mb_out = attn_out.numel() * attn_out.element_size() / 1e6
        print(f"\n=== ws={world_size}  seq_global={seq_global} "
              f"heads={H3_NUM_HEADS} d={H3_HEAD_DIM} bf16 ===")
        print(f"payload per rank: in={mb_in:.0f} MB  out={mb_out:.0f} MB")
        print(f"{'stage':<28}{'ms':>10}")
        for name in (
            "sglang_in",
            "sglang_in_relayout",
            "sglang_in_transfer",
            "fu_in",
            "fu_in_precat",
            "fu_in_zerocopy",
            "sglang_out",
            "sglang_out_relayout_pre",
            "sglang_out_transfer",
            "sglang_out_relayout_post",
            "fu_out",
            "fu_out_zerocopy",
        ):
            print(f"{name:<28}{results[name]:>10.3f}")

        sglang_total = results["sglang_in"] + results["sglang_out"]
        relayout = (
            results["sglang_in_relayout"]
            + results["sglang_out_relayout_pre"]
            + results["sglang_out_relayout_post"]
        )
        variants = {
            "fast-ulysses (as-is)": results["fu_in"] + results["fu_out"],
            "  + cat hoisted": results["fu_in_precat"] + results["fu_out"],
            "  + out= zero-copy": results["fu_in_zerocopy"]
            + results["fu_out_zerocopy"],
        }
        print(f"\n{'per-block sglang':<28}{sglang_total:>10.3f} ms"
              f"   (relayout {relayout:.3f} = {100 * relayout / sglang_total:.0f}%)")
        for name, value in variants.items():
            print(f"{name:<28}{value:>10.3f} ms   {sglang_total / value:.2f}x")

        best = min(variants.values())
        print(f"\nper denoise step ({H3_NUM_LAYERS} blocks):")
        print(f"  sglang        {sglang_total * H3_NUM_LAYERS:8.1f} ms")
        print(f"  best variant  {best * H3_NUM_LAYERS:8.1f} ms"
              f"   saves {(sglang_total - best) * H3_NUM_LAYERS:.1f} ms/step")
        print(f"\nfast-ulysses internal stages (ms): in={stages_in} out={stages_out}")

    group.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
