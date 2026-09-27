# SPDX-License-Identifier: Apache-2.0
"""Where a layer-step of subblock_sparse_sage_sm120 spends its time on one GPU.

Times the local path (`forward_varlen` after a BF16 exchange) piece by piece at
the MiniMax-H3 768p 5 s geometry, for a dense-table step and a routed step.

    CUDA_VISIBLE_DEVICES=0 python test/manual/diffusion/bench_subblock_sparse_sage_sm120_impl.py
"""

from __future__ import annotations

import time
from unittest.mock import patch

import torch

from sglang.kernels.ops.diffusion import sage_block_sparse_attn_sm120
from sglang.multimodal_gen.runtime.layers.attention.backends import (
    subblock_sparse_sage_sm120 as mod,
)

HEADS, HEAD_DIM, SEQ, USED = 7, 128, 37888, 37874
SCALE = HEAD_DIM**-0.5


class _FakeServerArgs:
    attention_backend_config = {"sparsity": 0.75, "skip_first_steps": 2, "min_seq_len": 4096}


def _impl():
    with (
        patch(
            "sglang.multimodal_gen.runtime.server_args.get_global_server_args",
            return_value=_FakeServerArgs(),
        ),
        patch.object(mod.SubBlockSparseSageSM120Impl, "_build_dense_impl", return_value=None),
    ):
        return mod.SubBlockSparseSageSM120Impl(
            num_heads=HEADS, head_size=HEAD_DIM, causal=False, softmax_scale=SCALE, prefix="blocks.3.attn"
        )


def _step(step):
    class _Ctx:
        current_timestep = step

    return patch(
        "sglang.multimodal_gen.runtime.layers.attention.backends.subblock_sparse_attn.get_forward_context",
        return_value=_Ctx(),
    )


def timed(label, fn, iters=10):
    fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        out = fn()
    end.record()
    torch.cuda.synchronize()
    print(f"{label:<44s} {start.elapsed_time(end) / iters:8.3f} ms")
    return out


def main():
    torch.manual_seed(0)
    impl = _impl()
    qkv = torch.randn(SEQ, 3, HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    qkv[USED:] = 0
    q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
    cu = torch.tensor([0, USED, SEQ], dtype=torch.int32, device="cuda")
    kw = dict(cu_seqlens=cu, max_seqlen=USED, cu_seqlens_host=(0, USED, SEQ))
    mask = torch.ones(-(-USED // 64), dtype=torch.bool, device="cuda")
    mask[:64] = False  # the leading text/audio blocks stay dense, as in H3

    with _step(0):
        timed("forward_varlen dense table", lambda: impl.forward_varlen(q, k, v, **kw))
    with _step(5):
        timed("forward_varlen routed, no mask", lambda: impl.forward_varlen(q, k, v, **kw))
        timed(
            "forward_varlen routed, video mask",
            lambda: impl.forward_varlen(q, k, v, first_segment_sparse_query_block_mask=mask, **kw),
        )

    op = timed("_quantize_local (W=1 kernels)", lambda: mod._quantize_local(q, k, v, used=USED))
    cut = -(-USED // 64) * 64
    route = timed(
        "_dequantize_for_routing",
        lambda: mod._dequantize_for_routing(op.q[:, :, :USED], op.k[:, :, :USED], op.k_scale),
    )
    plan = timed(
        "router.route",
        lambda: impl.router.route(
            route[0].transpose(1, 2), route[1].transpose(1, 2), sparsity=0.75, softmax_scale=SCALE
        ),
    )
    num_blocks = cut // 64
    timed("cake_block_tables no mask", lambda: mod.cake_block_tables(plan.index, plan.topk, num_blocks, None))
    tables = timed(
        "cake_block_tables video mask",
        lambda: mod.cake_block_tables(plan.index, plan.topk, num_blocks, mask),
    )
    out = torch.empty(1, HEADS, SEQ, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    dense_index, dense_nums = mod.sage_block_sparse_dense_block_index(1, HEADS, cut, cut, q.device)
    sparse_index, sparse_nums = mod.cake_block_tables(plan.index, plan.topk, num_blocks, None)

    def cake(index, nums):
        return sage_block_sparse_attn_sm120(
            op.q, op.k, op.v, op.q_scale, op.k_scale, op.v_scale, index, nums,
            out=out, seqlen_q=cut, seqlen_k=cut, softmax_scale=SCALE,
        )

    timed("cake dense table (592/592)", lambda: cake(dense_index, dense_nums))
    timed(f"cake sparse table ({plan.topk}/592)", lambda: cake(sparse_index, sparse_nums))
    timed("cake heterogeneous (video mask)", lambda: cake(*tables))
    print(f"plan: topk={plan.topk} of {num_blocks} blocks")


if __name__ == "__main__":
    main()
