# SPDX-License-Identifier: Apache-2.0
"""The quantised Ulysses exchange of subblock_sparse_sage_sm120 against its own local path.

Two ranks quantise their shards, exchange INT8/FP8 bytes and attend; each rank's
heads must match one rank quantising the whole sequence (the only difference is
the summation order of the K channel mean), and the route guard must decline
non-H3 layouts on every rank without touching a collective.

    pytest -v python/sglang/multimodal_gen/test/single_test_file/test_subblock_sparse_sage_sm120_2_gpu.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest

import torch

from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.test.test_utils import CustomTestCase

_WORLD = 2
HEADS = 8
HEAD_DIM = 128
SEQ = 2048
USED = 2000
SCALE = HEAD_DIM**-0.5


def _worker() -> int:
    from unittest.mock import patch

    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.multimodal_gen.runtime.layers.attention.backends.subblock_sparse_sage_sm120 import (
        SubBlockSparseSageSM120Impl,
        get_lowp_counters,
        reset_lowp_counters,
    )

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    init_distributed_environment(world_size=world, rank=rank, local_rank=rank)
    initialize_model_parallel(
        sequence_parallel_degree=world, ulysses_degree=world, ring_degree=1
    )

    class _FakeServerArgs:
        attention_backend_config = {"sparsity": 0.5, "skip_first_steps": 10}

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
            num_heads=HEADS // world,
            head_size=HEAD_DIM,
            causal=False,
            softmax_scale=SCALE,
            prefix="blocks.5.attn",
        )

    g = torch.Generator(device="cuda").manual_seed(0)
    qkv = torch.randn(
        SEQ, 3, HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16, generator=g
    )
    qkv[USED:] = 0
    q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
    local = SEQ // world
    shard = slice(rank * local, (rank + 1) * local)
    h = HEADS // world
    heads = slice(rank * h, (rank + 1) * h)
    failures = []

    with patch(
        "sglang.multimodal_gen.runtime.layers.attention.backends.subblock_sparse_attn.get_forward_context",
        return_value=_Ctx(),
    ):
        reset_lowp_counters()
        out = impl.forward_ulysses_lowp(
            q[shard],
            k[shard],
            v[shard],
            cu_seqlens_host=(0, USED, SEQ),
            max_seqlen=USED,
            ring_active=False,
            sparse_query_block_mask=None,
        )
        if out is None or tuple(out.shape) != (SEQ, h, HEAD_DIM):
            failures.append(
                f"lowp route declined or misshaped: {None if out is None else tuple(out.shape)}"
            )
        else:
            local_out = impl.forward_varlen(
                q,
                k,
                v,
                cu_seqlens=torch.tensor(
                    [0, USED, SEQ], dtype=torch.int32, device="cuda"
                ),
                max_seqlen=USED,
                cu_seqlens_host=(0, USED, SEQ),
            )[:, heads]
            got, want = out[:USED].float(), local_out[:USED].float()
            rel = ((got - want).norm() / want.norm()).item()
            if rel > 1e-2:
                failures.append(f"lowp vs local rel-L2 {rel:.4f}")
            qf, kf, vf = (t[:USED, heads].transpose(0, 1).float() for t in (q, k, v))
            ref = (
                torch.softmax(qf @ kf.transpose(-1, -2) * SCALE, dim=-1) @ vf
            ).transpose(0, 1)
            rel_ref = ((got - ref).norm() / ref.norm()).item()
            if rel_ref > 6e-2:
                failures.append(f"lowp vs fp32 rel-L2 {rel_ref:.4f}")
            if out[USED:].float().any():
                failures.append("padding rows are not zero")
        declined = impl.forward_ulysses_lowp(
            q[shard],
            k[shard],
            v[shard],
            cu_seqlens_host=(0, 500, 1000, SEQ),
            max_seqlen=500,
            ring_active=False,
            sparse_query_block_mask=None,
        )
        if declined is not None:
            failures.append("a two-document layout was not declined")
        counters = get_lowp_counters()
        if counters != {"lowp_a2a_total": 1, "bf16_a2a_total{packed_layout}": 1}:
            failures.append(f"unexpected counters {counters}")

    for failure in failures:
        print(f"FAILURE rank{rank}: {failure}", flush=True)
    torch.distributed.barrier()
    return 1 if failures else 0


class TestSubBlockSparseSageSM120TwoRanks(CustomTestCase):
    def test_quantised_exchange_matches_local_quantisation(self):
        if not current_platform.is_cuda():
            self.skipTest("CUDA-only test")
        if torch.cuda.device_count() < _WORLD:
            self.skipTest(f"needs {_WORLD} GPUs")
        if torch.cuda.get_device_capability() != (12, 0):
            self.skipTest("needs SM120 GPUs")
        procs = []
        for rank in range(_WORLD):
            env = os.environ.copy()
            env.update(
                {
                    "RANK": str(rank),
                    "LOCAL_RANK": str(rank),
                    "WORLD_SIZE": str(_WORLD),
                    "MASTER_ADDR": "127.0.0.1",
                    "MASTER_PORT": "29763",
                }
            )
            procs.append(
                subprocess.Popen(
                    [sys.executable, __file__],
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            )
        outputs = [p.communicate(timeout=900)[0] for p in procs]
        codes = [p.returncode for p in procs]
        if any(codes):
            self.fail("worker failed:\n" + "\n".join(outputs))


if __name__ == "__main__":
    if "RANK" in os.environ:
        sys.exit(_worker())
    unittest.main()
