"""The fast-ulysses Ulysses exchange must match the existing path bit for bit.

fast-ulysses moves the same bytes on the copy engines and folds the relayout
into copy strides, so it is a pure speed change: any difference in the output
is a bug, not a tolerance. Nothing in the single-GPU suite reaches this code,
so without this test a regression lands silently.

Runs H3's two collectives twice per shape -- once over NCCL, once over
fast-ulysses -- and requires identical bytes:

    pytest -v python/sglang/multimodal_gen/test/single_test_file/test_fast_ulysses_a2a_2_gpu.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest

import torch

from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.test.test_utils import CustomTestCase

_WORLD = int(os.environ.get("SGLANG_TEST_FAST_ULYSSES_WORLD", "2"))


def _worker() -> int:
    """One rank: compare both Ulysses exchanges against their NCCL result."""
    import torch.distributed as dist

    from sglang.multimodal_gen import envs
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        maybe_init_distributed_environment_and_model_parallel,
    )

    world = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    maybe_init_distributed_environment_and_model_parallel(
        tp_size=1, sp_size=world, ulysses_degree=world
    )

    from sglang.multimodal_gen.runtime.layers import usp as usp_mod

    def both_paths(fn, *args, **kwargs):
        """Run once on the existing collective, once on fast-ulysses."""
        usp_mod._FAST_ULYSSES_PROBED = False
        usp_mod._FAST_ULYSSES_GROUPS.clear()
        envs.SGLANG_DIFFUSION_FAST_ULYSSES = False
        # The IPC transport is a third path; hold it off so this compares the
        # two arms under test rather than whichever 2-rank fast path won.
        ipc_was = envs.SGLANG_DIFFUSION_IPC_A2A
        envs.SGLANG_DIFFUSION_IPC_A2A = False
        baseline = fn(*args, **kwargs)

        usp_mod._FAST_ULYSSES_PROBED = False
        usp_mod._FAST_ULYSSES_GROUPS.clear()
        envs.SGLANG_DIFFUSION_FAST_ULYSSES = True
        fast = fn(*args, **kwargs)
        engaged = bool(usp_mod._FAST_ULYSSES_GROUPS)

        envs.SGLANG_DIFFUSION_FAST_ULYSSES = False
        envs.SGLANG_DIFFUSION_IPC_A2A = ipc_was
        return baseline, fast, engaged

    failures = []
    engagements = []

    # (s_local, h_global, head_dim); h_global must split across the group and
    # head_dim * 2 bytes must be 16-byte aligned. The last row is H3's own
    # geometry at a shortened sequence.
    shapes = [
        (256, 8, 64),
        (1152, 24, 128),
        (512, 56, 128),
    ]
    for s_local, h_global, head_dim in shapes:
        if h_global % world:
            continue
        torch.manual_seed(1234 + rank)
        q, k, v = (
            torch.randn(
                s_local, h_global, head_dim, dtype=torch.bfloat16, device="cuda"
            )
            for _ in range(3)
        )
        base, fast, engaged = both_paths(
            usp_mod._usp_input_all_to_all_packed_qkv, q, k, v
        )
        engagements.append(engaged)
        for i, (b, f) in enumerate(zip(base, fast)):
            if b.shape != f.shape:
                failures.append(
                    f"input a2a qkv[{i}] shape {tuple(b.shape)} != {tuple(f.shape)} "
                    f"for {(s_local, h_global, head_dim)}"
                )
            elif not torch.equal(b, f):
                failures.append(f"input a2a qkv[{i}] {(s_local, h_global, head_dim)}")

        # The output exchange consumes [b, s_global, h_local, d].
        out = torch.randn(
            1,
            s_local * world,
            h_global // world,
            head_dim,
            dtype=torch.bfloat16,
            device="cuda",
        )
        base, fast, engaged = both_paths(
            usp_mod._usp_output_all_to_all, out, head_dim=2
        )
        engagements.append(engaged)
        if base.shape != fast.shape:
            failures.append(
                f"output a2a shape {tuple(base.shape)} != {tuple(fast.shape)} "
                f"for {(s_local, h_global, head_dim)}"
            )
        elif not torch.equal(base, fast):
            failures.append(f"output a2a {(s_local, h_global, head_dim)}")

    # A comparison where both arms quietly ran the same code would pass while
    # proving nothing, so require evidence the transport actually engaged.
    if not any(engagements):
        failures.append(
            "fast-ulysses never engaged; both arms ran the existing collective"
        )

    # `head_dim=1` is [b, h_local, s_global, d], which fast-ulysses does not
    # accept. It must stay on the existing collective rather than be reshaped
    # into the fast path, which would cost back the permute this path removes.
    usp_mod._FAST_ULYSSES_PROBED = False
    usp_mod._FAST_ULYSSES_GROUPS.clear()
    envs.SGLANG_DIFFUSION_FAST_ULYSSES = True
    x_hd1 = torch.randn(
        1, 8 // world, 64 * world, 64, dtype=torch.bfloat16, device="cuda"
    )
    try:
        usp_mod._usp_output_all_to_all(x_hd1, head_dim=1)
    except Exception as exc:  # noqa: BLE001 - any failure here is the finding
        failures.append(f"head_dim=1 did not stay on the existing path: {exc!r}")
    envs.SGLANG_DIFFUSION_FAST_ULYSSES = False

    verdict = torch.tensor([len(failures)], device="cuda")
    dist.all_reduce(verdict)
    if failures:
        print(f"rank{rank} MISMATCH: {failures}", flush=True)
    if rank == 0:
        print(
            f"FAST_ULYSSES_PARITY {'FAIL' if verdict.item() else 'PASS'} "
            f"(world={world}, engaged={sum(engagements)}/{len(engagements)})",
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()
    return 1 if verdict.item() else 0


class TestFastUlyssesA2A(CustomTestCase):
    world_size = _WORLD

    def test_fast_ulysses_matches_nccl_bitwise(self):
        if not current_platform.is_cuda():
            self.skipTest("fast-ulysses is a CUDA/NVLink transport")
        if torch.cuda.device_count() < self.world_size:
            self.skipTest(f"needs {self.world_size} GPUs")
        try:
            import fast_ulysses  # noqa: F401
        except ImportError:
            self.skipTest("fast-ulysses is not installed")

        env = dict(os.environ)
        # This transport does not use NVLink SHARP, but the NCCL arm shares the
        # process and fails init where multicast cannot be bound.
        env.setdefault("NCCL_NVLS_ENABLE", "0")
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                f"--nproc-per-node={self.world_size}",
                "--master-port=29519",
                __file__,
                "--worker",
            ],
            capture_output=True,
            text=True,
            timeout=1200,
            env=env,
        )
        print(proc.stdout[-4000:])
        if proc.returncode != 0:
            print(proc.stderr[-4000:], file=sys.stderr)
        self.assertEqual(proc.returncode, 0, "fast-ulysses parity worker failed")
        self.assertIn("FAST_ULYSSES_PARITY PASS", proc.stdout)


if __name__ == "__main__":
    if "--worker" in sys.argv:
        sys.exit(_worker())
    unittest.main()
