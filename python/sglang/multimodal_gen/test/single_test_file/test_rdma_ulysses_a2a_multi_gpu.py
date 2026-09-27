# SPDX-License-Identifier: Apache-2.0
"""The RDMA Ulysses transport against NCCL on one node with an mlx5 RoCE port per rank.

Runs manually on the 8x RTX PRO 5000 box (2, 4 or 8 ranks); skipped without
rdma-core, enough GPUs, or mlx5 devices. Checks, on every rank: chunk exchange
equals all_to_all_single across two geometries (the second forces a rebind),
zero-copy packing into the landing buffer, gather_heads equals the NCCL output
all-to-all for bf16 and uint8, request counters, and that a rank publishing the
sticky abort makes every peer's next exchange fail within the timeout while the
group can still shut the transport down.

    python python/sglang/multimodal_gen/test/single_test_file/test_rdma_ulysses_a2a_multi_gpu.py [world_size]
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

import torch

from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.test.test_utils import CustomTestCase

HEADS, HEAD_DIM = 56, 128


def _worker() -> int:
    import torch.distributed as dist

    from sglang.multimodal_gen.runtime.distributed.device_communicators.rdma_ulysses_a2a import (
        get_rdma_ulysses_a2a,
        shutdown_rdma_ulysses_a2a,
    )
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        get_sp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.multimodal_gen.runtime.layers.usp import _usp_output_all_to_all

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    abort_test = os.environ.get("RDMA_ABORT_TEST") == "1"
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    init_distributed_environment(world_size=world, rank=rank, local_rank=rank)
    initialize_model_parallel(
        sequence_parallel_degree=world, ulysses_degree=world, ring_degree=1
    )
    group = get_sp_group().ulysses_group
    failures = []

    with torch.inference_mode():
        transport = get_rdma_ulysses_a2a(
            group,
            device,
            max_seq_len=37888,
            heads=HEADS,
            head_dim=HEAD_DIM,
            strict=True,
        )
        g = torch.Generator(device="cuda").manual_seed(100 + rank)

        for chunk in (12736640, 4 * 1024 * 1024 + 128, 12736640):
            payload = torch.randint(
                0, 256, (world, chunk), generator=g, device="cuda", dtype=torch.int32
            ).to(torch.uint8)
            want = torch.empty_like(payload)
            dist.all_to_all_single(want.view(-1), payload.view(-1), group=group)
            got = transport.exchange_chunks(payload)
            if not torch.equal(got, want):
                failures.append(
                    f"chunk exchange C={chunk} differs from all_to_all_single"
                )
            landing = transport.packed_input_buffer((world, chunk))
            landing.copy_(payload)
            got = transport.exchange_chunks(landing)
            if not torch.equal(got, want):
                failures.append(f"zero-copy chunk exchange C={chunk} differs")

        for s_global, dtype in (
            (37888, torch.bfloat16),
            (16384, torch.bfloat16),
            (37888, torch.uint8),
        ):
            h = HEADS // world
            if dtype is torch.uint8:
                x = torch.randint(
                    0,
                    256,
                    (s_global, h, HEAD_DIM),
                    generator=g,
                    device="cuda",
                    dtype=torch.int32,
                ).to(torch.uint8)
            else:
                x = torch.randn(
                    s_global, h, HEAD_DIM, generator=g, device="cuda", dtype=dtype
                )
            want = _usp_output_all_to_all(x[None], head_dim=2)[0]
            got = transport.gather_heads(x)
            if not torch.equal(got, want):
                failures.append(f"gather_heads S={s_global} {dtype} differs from NCCL")
            landing = transport.gather_landing(tuple(x.shape), dtype)
            landing.copy_(x.transpose(0, 1).transpose(0, 1))
            got = transport.gather_heads(landing)
            if not torch.equal(got, want):
                failures.append(f"zero-copy gather_heads S={s_global} {dtype} differs")

        stats = torch.randn(
            2, 1, HEADS, HEAD_DIM, generator=g, device="cuda", dtype=torch.float32
        )
        want = torch.empty(world, *stats.shape, device="cuda")
        dist.all_gather_into_tensor(want, stats, group=group)
        got = transport.exchange_stats(stats)
        if not torch.equal(got, want):
            failures.append("exchange_stats differs from all_gather_into_tensor")

        expected = 6 + 6 + 1
        if transport.exchanges != expected:
            failures.append(f"exchange counter {transport.exchanges} != {expected}")

        if abort_test:
            dist.barrier(group=group)
            payload = torch.zeros((world, 4096), dtype=torch.uint8, device="cuda")
            if rank == 0:
                index, _, _ = transport._slots["chunks"]
                transport._module.publish_abort_for_test(transport._handle, index)
            dist.barrier(group=group)
            try:
                transport.exchange_chunks(payload)
            except Exception as error:  # noqa: BLE001
                if "abort" not in str(error) and "poisoned" not in str(error):
                    failures.append(f"unexpected abort error: {error}")
            else:
                failures.append("exchange after an abort did not fail")

        try:
            shutdown_rdma_ulysses_a2a()
        except Exception as error:  # noqa: BLE001
            failures.append(f"shutdown failed: {error}")

    for failure in failures:
        print(f"FAILURE rank{rank}: {failure}", flush=True)
    dist.barrier(group=group)
    return 1 if failures else 0


def _run(world: int, *, abort_test: bool) -> tuple[list[int], list[str]]:
    procs = []
    for rank in range(world):
        env = os.environ.copy()
        env.update(
            {
                "RANK": str(rank),
                "LOCAL_RANK": str(rank),
                "WORLD_SIZE": str(world),
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": "29781" if not abort_test else "29782",
                "RDMA_ABORT_TEST": "1" if abort_test else "0",
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
    outputs = [p.communicate(timeout=1800)[0] for p in procs]
    return [p.returncode for p in procs], outputs


class TestRdmaUlyssesA2A(CustomTestCase):
    def _skip_unless_possible(self, world: int) -> None:
        from sglang.kernels.ops.communication.rdma_ulysses import missing_rdma_libraries

        if not current_platform.is_cuda():
            self.skipTest("CUDA-only test")
        if torch.cuda.device_count() < world:
            self.skipTest(f"needs {world} GPUs")
        if missing_rdma_libraries():
            self.skipTest("needs rdma-core")
        if not any(
            p.name.startswith("mlx5_")
            for p in Path("/sys/class/infiniband").glob("mlx5_*")
        ):
            self.skipTest("needs mlx5 devices")

    def test_exchanges_match_nccl(self):
        world = int(os.environ.get("RDMA_TEST_WORLD", "0")) or torch.cuda.device_count()
        self._skip_unless_possible(world)
        codes, outputs = _run(world, abort_test=False)
        if any(codes):
            self.fail("worker failed:\n" + "\n".join(outputs))

    def test_abort_is_group_wide_and_teardown_survives(self):
        world = int(os.environ.get("RDMA_TEST_WORLD", "0")) or torch.cuda.device_count()
        self._skip_unless_possible(world)
        codes, outputs = _run(world, abort_test=True)
        if any(codes):
            self.fail("worker failed:\n" + "\n".join(outputs))


if __name__ == "__main__":
    if "RANK" in os.environ:
        sys.exit(_worker())
    if len(sys.argv) > 1:
        os.environ["RDMA_TEST_WORLD"] = sys.argv.pop(1)
    unittest.main()
