"""The symmetric-memory copy-engine all-to-all must match the NCCL exchange bit
for bit.

It only engages with SGLANG_DIFFUSION_SYMM_A2A on a same-host Ulysses group
with peer-to-peer access, so nothing in the single-GPU suite reaches it. Runs
on 2 ranks, and on 4 when four GPUs are visible:

    pytest -v python/sglang/multimodal_gen/test/single_test_file/test_symm_a2a_multi_gpu.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest

import torch

from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.test.test_utils import CustomTestCase


def _attend(q, k, v):
    # elementwise across the whole gathered sequence, so a misplaced shard shows up
    return (q.float() * 2 + k.flip(0).float() - v.roll(1, 0).float()).to(q.dtype)


def _worker() -> int:
    import torch.distributed as dist

    from sglang.multimodal_gen import envs
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        maybe_init_distributed_environment_and_model_parallel,
    )

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    maybe_init_distributed_environment_and_model_parallel(
        tp_size=1, sp_size=world, ulysses_degree=world
    )

    from sglang.multimodal_gen.runtime.distributed.device_communicators.symm_a2a import (
        SLOTS,
        SYMM_A2A,
    )
    from sglang.multimodal_gen.runtime.layers.usp import (
        _usp_input_all_to_all_packed_qkv,
        _usp_input_all_to_all_qkv,
        _usp_output_all_to_all,
        uncached_a2a_staging,
    )

    peers = [d for d in range(world) if d != rank]
    if not all(torch.cuda.can_device_access_peer(rank, d) for d in peers):
        print("SKIP no peer-to-peer access between the devices", flush=True)
        return 0

    failures = []

    def check(ok, what):
        if not ok:
            failures.append(what)

    def arms(fn, *args):
        """The NCCL reference and the transport on the same inputs; returns
        (reference, result, exchanges the transport ran)."""
        envs.SGLANG_DIFFUSION_SYMM_A2A = False
        ref = fn(*args)
        envs.SGLANG_DIFFUSION_SYMM_A2A = True
        before = SYMM_A2A.calls
        got = fn(*args)
        envs.SGLANG_DIFFUSION_SYMM_A2A = False
        return ref, got, SYMM_A2A.calls - before

    def same(a, b):
        if isinstance(a, torch.Tensor):
            return a.shape == b.shape and torch.equal(a, b)
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))

    def in_workspace(t):
        lo = SYMM_A2A.workspace.data_ptr()
        return lo <= t.data_ptr() < lo + SLOTS * SYMM_A2A.stride

    def rand(*shape):
        return torch.randn(*shape, dtype=torch.bfloat16, device="cuda")

    # (s_local, heads, head_dim); heads % world == 0, odd row counts too
    for s_local, heads, head_dim in [(96, 16, 64), (40, 32, 128), (7, 8, 64)]:
        tag = f"{s_local}x{heads}x{head_dim}"
        for call in range(3):  # slots are reused and the counters advance
            torch.manual_seed(1000 * call + s_local + rank)
            # MiniMax-H3's fused-projection strides (3HD, D, 1)
            q, k, v = rand(s_local, 3, heads, head_dim).unbind(1)
            assert not q.is_contiguous()
            ref, got, n = arms(_usp_input_all_to_all_packed_qkv, q, k, v)
            check(n == 1, f"packed qkv {tag} call {call}: ran {n} exchanges")
            check(same(ref, got), f"packed qkv {tag} call {call}")
            check(all(in_workspace(t) for t in got), f"packed qkv {tag}: not slot views")
            out = _attend(*got)  # consumes the slots
            ref, got, n = arms(_usp_output_all_to_all, out[None], 2)
            check(n == 1 and same(ref, got), f"output {tag} call {call}")
            check(in_workspace(got), f"output {tag}: not a slot view")
        for batch in (1, 2):  # the 4D owned path and the per-batch copy loop
            torch.manual_seed(7 * s_local + batch + rank)
            q4, k4, v4 = (rand(batch, s_local, heads, head_dim) for _ in range(3))
            ref, got, n = arms(_usp_input_all_to_all_qkv, q4, k4, v4)
            check(n == 1 and same(ref, got), f"qkv 4D b={batch} {tag}")
            check(not any(in_workspace(t) for t in got), f"qkv 4D b={batch}: must be owned")
            o4 = rand(batch, s_local * world, heads // world, head_dim)
            ref, got, n = arms(_usp_output_all_to_all, o4, 2)
            check(n == 1 and same(ref, got), f"output 4D b={batch} {tag}")
            # a sequence-offset slice, as callers pass out[:, num_replicated:]
            ref, got, n = arms(_usp_output_all_to_all, o4[:, world:], 2)
            check(n == 1 and same(ref, got), f"output 4D slice b={batch} {tag}")

    # GQA: fewer k/v heads than q heads, each tensor its own layout
    torch.manual_seed(42 + rank)
    ref, got, n = arms(_usp_input_all_to_all_qkv, rand(1, 32, 16, 64), rand(1, 32, 8, 64), rand(1, 32, 8, 64))
    check(n == 1 and same(ref, got), "qkv 4D GQA")

    # growth: a larger message reallocates the workspace, then small shapes still work
    stride_before = SYMM_A2A.stride
    torch.manual_seed(5 + rank)
    q, k, v = rand(1024, 3, 16, 64).unbind(1)
    ref, got, n = arms(_usp_input_all_to_all_packed_qkv, q, k, v)
    check(n == 1 and same(ref, got), "packed qkv after growth")
    check(SYMM_A2A.stride > stride_before, "workspace did not grow")
    q, k, v = rand(16, 3, 16, 64).unbind(1)
    ref, got, n = arms(_usp_input_all_to_all_packed_qkv, q, k, v)
    check(n == 1 and same(ref, got), "small packed qkv after growth")

    # decline: a message that does not fit on some rank stays on NCCL and is
    # remembered, without a collective, on later calls
    fits = SYMM_A2A._fits_on_every_rank
    SYMM_A2A._fits_on_every_rank = lambda nbytes: False
    try:
        q, k, v = rand(4096, 3, 16, 64).unbind(1)
        ref, got, n = arms(_usp_input_all_to_all_packed_qkv, q, k, v)
        check(n == 0 and same(ref, got), "declined message")
        check(SYMM_A2A.declined == q.nbytes, "decline not remembered")
        ref, got, n = arms(_usp_input_all_to_all_packed_qkv, q, k, v)
        check(n == 0 and same(ref, got), "declined message, second call")
    finally:
        SYMM_A2A._fits_on_every_rank = fits
        SYMM_A2A.declined = 0

    # the pipelined attention's first-sight reference keeps plain NCCL tensors
    with uncached_a2a_staging():
        q, k, v = rand(32, 3, 16, 64).unbind(1)
        ref, got, n = arms(_usp_input_all_to_all_packed_qkv, q, k, v)
        check(n == 0 and same(ref, got), "uncached staging engaged the transport")

    # skew: queued exchanges with one rank running late must still pair up
    issued = []
    envs.SGLANG_DIFFUSION_SYMM_A2A = True
    for call in range(6):
        torch.manual_seed(500 + call + rank)
        q, k, v = rand(64, 3, 16, 64).unbind(1)
        if rank == 0:
            torch.cuda._sleep(20_000_000)
        got = _usp_input_all_to_all_packed_qkv(q, k, v)
        issued.append(((q, k, v), _attend(*got)))
    envs.SGLANG_DIFFUSION_SYMM_A2A = False
    for (q, k, v), out in issued:
        ref = _usp_input_all_to_all_packed_qkv(q, k, v)
        check(torch.equal(_attend(*ref), out), "skewed exchange diverged")

    # a comparison where every arm quietly fell back would pass while testing nothing
    if not SYMM_A2A.inited or SYMM_A2A.failed or SYMM_A2A.workspace is None:
        failures.append(
            f"transport never engaged (inited={SYMM_A2A.inited} "
            f"failed={SYMM_A2A.failed} workspace={SYMM_A2A.workspace is not None})"
        )

    verdict = torch.tensor([len(failures)], device="cuda")
    dist.all_reduce(verdict)
    if failures:
        print(f"rank{rank} MISMATCH: {failures}", flush=True)
    if rank == 0:
        print(
            f"SYMM_A2A_PARITY {'FAIL' if verdict.item() else 'PASS'} world={world}",
            flush=True,
        )
    SYMM_A2A.reset()
    dist.barrier()
    dist.destroy_process_group()
    return 1 if verdict.item() else 0


class TestSymmA2AMultiGpu(CustomTestCase):
    def _run(self, world: int, port: int):
        if not current_platform.is_cuda():
            self.skipTest("the symmetric-memory transport is unavailable on this platform")
        if torch.cuda.device_count() < world:
            self.skipTest(f"needs {world} GPUs")
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                f"--nproc-per-node={world}",
                f"--master-port={port}",
                __file__,
                "--worker",
            ],
            capture_output=True,
            text=True,
            timeout=900,
            env=os.environ.copy(),
        )
        print(proc.stdout[-4000:])
        if proc.returncode != 0:
            print(proc.stderr[-4000:], file=sys.stderr)
        self.assertEqual(proc.returncode, 0, "symmetric-memory all-to-all diverged")
        if "SKIP" not in proc.stdout:
            self.assertIn(f"SYMM_A2A_PARITY PASS world={world}", proc.stdout)

    def test_two_ranks_match_nccl_bitwise(self):
        self._run(2, 29577)

    def test_four_ranks_match_nccl_bitwise(self):
        self._run(4, 29587)


if __name__ == "__main__":
    if "--worker" in sys.argv:
        raise SystemExit(_worker())
    unittest.main()
