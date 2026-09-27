# SPDX-License-Identifier: Apache-2.0
"""Time the Ulysses exchange primitives the MiniMax-H3 attention core issues at 768p 5 s.

Spawns one process per GPU, builds the Ulysses group, and times on the host
clock (max over ranks via all-reduce) the BF16 packed QKV exchange, the uint8
payload exchange, the stats all-gather, and both attention paths of
subblock_sparse_sage_sm120 end to end.

    python python/sglang/multimodal_gen/test/single_test_file/bench_ulysses_exchange_8_gpu.py [world_size]
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import torch

HEADS, HEAD_DIM, SEQ, USED = 56, 128, 37888, 37874
SCALE = HEAD_DIM**-0.5
ITERS = 10


def _worker() -> int:
    from unittest.mock import patch

    import torch.distributed as dist

    from sglang.kernels.ops.diffusion import ulysses_lowp_payload_spec
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        get_sp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.multimodal_gen.runtime.layers.attention.backends.subblock_sparse_sage_sm120 import (
        SubBlockSparseSageSM120Impl,
    )
    from sglang.multimodal_gen.runtime.layers.usp import (
        _usp_all_to_all_single,
        _usp_input_all_to_all_packed_qkv,
        _usp_output_all_to_all,
    )

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    torch.inference_mode().__enter__()  # the attention core runs under inference mode
    init_distributed_environment(world_size=world, rank=rank, local_rank=rank)
    initialize_model_parallel(
        sequence_parallel_degree=world, ulysses_degree=world, ring_degree=1
    )
    group = get_sp_group().ulysses_group
    local = SEQ // world
    h = HEADS // world

    class _FakeServerArgs:
        attention_backend_config = {
            "sparsity": 0.75,
            "skip_first_steps": 2,
            "min_seq_len": 4096,
        }

    class _Ctx:
        current_timestep = 0

    with (
        patch(
            "sglang.multimodal_gen.runtime.server_args.get_global_server_args",
            return_value=_FakeServerArgs(),
        ),
        patch.object(
            SubBlockSparseSageSM120Impl, "_build_dense_impl", return_value=None
        ),
    ):
        impl = SubBlockSparseSageSM120Impl(
            num_heads=h,
            head_size=HEAD_DIM,
            causal=False,
            softmax_scale=SCALE,
            prefix="blocks.3.attn",
        )

    g = torch.Generator(device="cuda").manual_seed(rank)
    qkv = torch.randn(
        local, 3, HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16, generator=g
    )
    q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
    spec = ulysses_lowp_payload_spec(
        batch=1, local_sequence=local, num_heads=HEADS, world_size=world
    )
    payload = torch.zeros(spec.payload_shape, dtype=torch.uint8, device="cuda")
    stats = torch.zeros(2, 1, HEADS, HEAD_DIM, dtype=torch.float32, device="cuda")
    gathered = torch.zeros(world, *stats.shape, dtype=torch.float32, device="cuda")
    attn_out = torch.randn(SEQ, h, HEAD_DIM, device="cuda", dtype=torch.bfloat16)

    def timed(label, fn):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        dist.barrier(group=group)
        start = time.perf_counter()
        for _ in range(ITERS):
            fn()
        torch.cuda.synchronize()
        elapsed = torch.tensor(
            [(time.perf_counter() - start) / ITERS * 1e3], device="cuda"
        )
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX, group=group)
        if rank == 0:
            print(f"{label:<52s} {elapsed.item():8.3f} ms", flush=True)

    timed(
        "bf16 packed QKV a2a (_usp_input_all_to_all_packed_qkv)",
        lambda: _usp_input_all_to_all_packed_qkv(q, k, v),
    )
    timed(
        f"uint8 payload a2a ({payload.numel() / 1e6:.1f} MB/rank)",
        lambda: _usp_all_to_all_single(payload, role="bench_u8"),
    )
    recv = torch.empty_like(payload)
    timed(
        "uint8 payload a2a, plain dist.all_to_all_single",
        lambda: dist.all_to_all_single(recv.view(-1), payload.view(-1), group=group),
    )
    half = torch.empty(payload.numel() // 2, dtype=torch.bfloat16, device="cuda")
    timed(
        "bf16 same bytes a2a, plain dist.all_to_all_single",
        lambda: dist.all_to_all_single(torch.empty_like(half), half, group=group),
    )
    timed(
        "stats all_gather_into_tensor",
        lambda: dist.all_gather_into_tensor(gathered, stats, group=group),
    )
    timed(
        "bf16 output a2a (_usp_output_all_to_all)",
        lambda: _usp_output_all_to_all(attn_out[None], head_dim=2),
    )

    kw = dict(
        cu_seqlens_host=(0, USED, SEQ),
        max_seqlen=USED,
        ring_active=False,
        sparse_query_block_mask=None,
    )
    with patch(
        "sglang.multimodal_gen.runtime.layers.attention.backends.subblock_sparse_attn.get_forward_context",
        return_value=_Ctx(),
    ):
        timed(
            "forward_ulysses_lowp (dense table)",
            lambda: impl.forward_ulysses_lowp(q, k, v, **kw),
        )

        def bf16_path():
            qg, kg, vg = _usp_input_all_to_all_packed_qkv(q, k, v)
            return impl.forward_varlen(
                qg,
                kg,
                vg,
                cu_seqlens=None,
                max_seqlen=USED,
                cu_seqlens_host=(0, USED, SEQ),
            )

        timed("bf16 a2a + forward_varlen (dense table)", bf16_path)
        _Ctx.current_timestep = 5
        timed(
            "forward_ulysses_lowp (routed)",
            lambda: impl.forward_ulysses_lowp(q, k, v, **kw),
        )
        timed("bf16 a2a + forward_varlen (routed)", bf16_path)
        timed(
            "forward_ulysses_lowp + gather_output over NCCL (routed)",
            lambda: impl.gather_output(impl.forward_ulysses_lowp(q, k, v, **kw)),
        )
    if os.environ.get("BENCH_RDMA") == "1":
        from sglang.multimodal_gen.runtime.distributed.device_communicators.rdma_ulysses_a2a import (
            get_rdma_ulysses_a2a,
            shutdown_rdma_ulysses_a2a,
        )

        transport = get_rdma_ulysses_a2a(
            group,
            torch.device("cuda", rank),
            max_seq_len=SEQ,
            heads=HEADS,
            head_dim=HEAD_DIM,
            strict=True,
        )
        landing = transport.packed_input_buffer(tuple(payload.shape))
        landing.copy_(payload)
        timed(
            f"RDMA exchange_chunks ({payload.numel() / 1e6:.1f} MB/rank, zero copy)",
            lambda: transport.exchange_chunks(landing),
        )
        timed(
            "RDMA exchange_chunks (staged)", lambda: transport.exchange_chunks(payload)
        )
        timed("RDMA exchange_stats", lambda: transport.exchange_stats(stats))
        gather_landing = transport.gather_landing(tuple(attn_out.shape), attn_out.dtype)
        gather_landing.copy_(attn_out)
        timed(
            "RDMA gather_heads (zero copy)",
            lambda: transport.gather_heads(gather_landing),
        )
        with patch(
            "sglang.multimodal_gen.runtime.layers.attention.backends.subblock_sparse_attn.get_forward_context",
            return_value=_Ctx(),
        ):
            _Ctx.current_timestep = 0
            timed(
                "forward_ulysses_lowp over RDMA (dense table)",
                lambda: impl.forward_ulysses_lowp(q, k, v, **kw),
            )
            timed(
                "forward_ulysses_lowp + gather_output over RDMA (dense)",
                lambda: impl.gather_output(impl.forward_ulysses_lowp(q, k, v, **kw)),
            )
            _Ctx.current_timestep = 5
            timed(
                "forward_ulysses_lowp + gather_output over RDMA (routed)",
                lambda: impl.gather_output(impl.forward_ulysses_lowp(q, k, v, **kw)),
            )
        shutdown_rdma_ulysses_a2a()
    dist.barrier(group=group)
    return 0


if __name__ == "__main__":
    if "RANK" in os.environ:
        sys.exit(_worker())
    world = int(sys.argv[1]) if len(sys.argv) > 1 else torch.cuda.device_count()
    procs = []
    for rank in range(world):
        env = os.environ.copy()
        env.update(
            {
                "RANK": str(rank),
                "LOCAL_RANK": str(rank),
                "WORLD_SIZE": str(world),
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": "29771",
            }
        )
        procs.append(subprocess.Popen([sys.executable, __file__], env=env))
    sys.exit(max(p.wait() for p in procs))
