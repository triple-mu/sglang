"""The fast-ulysses gate must decide from process-wide constants only.

Both constructing the group and allocating the window behind a new shape are
collective. If eligibility could turn on a property of the tensor being
exchanged, one rank could take the fast path while another fell back, and that
deadlocks silently -- neither side raises and neither side times out. So the
gate is tested for what it reads, not only for what it returns.
"""

import unittest
from unittest.mock import patch

import torch

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.layers import usp as usp_mod

_USP = "sglang.multimodal_gen.runtime.layers.usp"


class TestFastUlyssesGate(unittest.TestCase):
    def setUp(self) -> None:
        self._reset_probe()
        self.addCleanup(self._reset_probe)

    @staticmethod
    def _reset_probe() -> None:
        usp_mod._FAST_ULYSSES_PROBED = False
        usp_mod._FAST_ULYSSES_GROUPS.clear()

    def test_disabled_by_default(self) -> None:
        self.assertFalse(envs.SGLANG_DIFFUSION_FAST_ULYSSES)

    def test_env_off_returns_none_without_importing(self) -> None:
        # An import attempt would be a side effect on a path that is supposed to
        # be free when the switch is off.
        import builtins

        real_import = builtins.__import__
        seen = []

        def spy(name, *args, **kwargs):
            seen.append(name)
            return real_import(name, *args, **kwargs)

        with (
            patch.object(envs, "SGLANG_DIFFUSION_FAST_ULYSSES", False),
            patch(f"{_USP}.get_ulysses_parallel_world_size", return_value=4),
            patch.object(builtins, "__import__", spy),
        ):
            self.assertIsNone(usp_mod._fast_ulysses_group())
        self.assertNotIn("fast_ulysses", seen)

    def test_world_size_outside_range_returns_none(self) -> None:
        # 1 has no exchange; above 8 is structural (`BarPeers::p[8]`).
        for world_size in (1, 9, 16):
            with self.subTest(world_size=world_size):
                self._reset_probe()
                with (
                    patch.object(envs, "SGLANG_DIFFUSION_FAST_ULYSSES", True),
                    patch(
                        f"{_USP}.get_ulysses_parallel_world_size",
                        return_value=world_size,
                    ),
                ):
                    self.assertIsNone(usp_mod._fast_ulysses_group())

    def test_probe_happens_once(self) -> None:
        calls = []

        def counting_world_size():
            calls.append(1)
            return 1

        with (
            patch.object(envs, "SGLANG_DIFFUSION_FAST_ULYSSES", True),
            patch(f"{_USP}.get_ulysses_parallel_world_size", counting_world_size),
        ):
            for _ in range(5):
                usp_mod._fast_ulysses_group()
        # Caching is what keeps every rank's decision identical for the whole
        # run; re-deciding per call would reintroduce the divergence risk.
        self.assertEqual(len(calls), 1)

    def test_unimportable_package_falls_back(self) -> None:
        import builtins

        real_import = builtins.__import__

        def refuse(name, *args, **kwargs):
            if name == "fast_ulysses":
                raise ImportError("simulated: not installed")
            return real_import(name, *args, **kwargs)

        with (
            patch.object(envs, "SGLANG_DIFFUSION_FAST_ULYSSES", True),
            patch(f"{_USP}.get_ulysses_parallel_world_size", return_value=4),
            patch(f"{_USP}.get_sp_group"),
            patch(f"{_USP}._vote_unanimous", return_value=False) as vote,
            patch.object(builtins, "__import__", refuse),
        ):
            self.assertIsNone(usp_mod._fast_ulysses_group())
        # The rank must report its own failure into the vote, not just return.
        self.assertEqual(vote.call_args.args[1], False)

    def test_one_rank_failing_import_disables_every_rank(self) -> None:
        # The deadlock guard. This rank CAN import fast-ulysses, but a peer
        # cannot. It must not enter the constructor, which is collective: a
        # rank that entered while a peer skipped would wait forever, with no
        # exception and no timeout.
        fake_module = unittest.mock.MagicMock()
        with (
            patch.object(envs, "SGLANG_DIFFUSION_FAST_ULYSSES", True),
            patch(f"{_USP}.get_ulysses_parallel_world_size", return_value=4),
            patch(f"{_USP}.get_sp_group"),
            patch(f"{_USP}._vote_unanimous", return_value=False),
            patch.dict("sys.modules", {"fast_ulysses": fake_module}),
        ):
            self.assertIsNone(usp_mod._fast_ulysses_group())
        fake_module.UlyssesGroup.assert_not_called()

    def test_refused_topology_falls_back(self) -> None:
        # fast-ulysses refuses a group that is not NVLink-joined. That must
        # degrade to the existing collective, not propagate. No second vote is
        # needed: the NVLink probe reads the same fabric on every rank.
        fake_module = unittest.mock.MagicMock()
        fake_module.UlyssesGroup.side_effect = RuntimeError("not NVLink-joined")
        with (
            patch.object(envs, "SGLANG_DIFFUSION_FAST_ULYSSES", True),
            patch(f"{_USP}.get_ulysses_parallel_world_size", return_value=4),
            patch(f"{_USP}.get_sp_group"),
            patch(f"{_USP}._vote_unanimous", return_value=True),
            patch.dict("sys.modules", {"fast_ulysses": fake_module}),
        ):
            self.assertIsNone(usp_mod._fast_ulysses_group())

    def test_async_split_gets_one_group_per_role(self) -> None:
        # The regression lock for the overlap. fast-ulysses keys its staging
        # buffer by (shape, dtype) *per group*, and q/k/v are the same shape:
        # on a shared group the second issue would wait, on the caller's
        # stream, for the first exchange to finish end to end -- which is
        # exactly the overlap this path exists to get.
        for async_on, expected in ((False, ["default"]), (True, ["default", "q", "k", "v"])):
            with self.subTest(async_on=async_on):
                self._reset_probe()
                fake_module = unittest.mock.MagicMock()
                fake_module.UlyssesGroup.side_effect = lambda **_: unittest.mock.MagicMock()
                with (
                    patch.object(envs, "SGLANG_DIFFUSION_FAST_ULYSSES", True),
                    patch.object(
                        envs, "SGLANG_DIFFUSION_FAST_ULYSSES_ASYNC_QKV", async_on
                    ),
                    patch(f"{_USP}.get_ulysses_parallel_world_size", return_value=4),
                    patch(f"{_USP}.get_sp_group"),
                    patch(f"{_USP}._vote_unanimous", return_value=True),
                    patch.dict("sys.modules", {"fast_ulysses": fake_module}),
                ):
                    usp_mod._fast_ulysses_group()
                self.assertEqual(list(usp_mod._FAST_ULYSSES_GROUPS), expected)
                # Distinct objects, not the same group handed out four times.
                groups = list(usp_mod._FAST_ULYSSES_GROUPS.values())
                self.assertEqual(len({id(g) for g in groups}), len(groups))
        self._reset_probe()

    def test_out_buffer_cache_is_capped(self) -> None:
        # Symmetric windows are never released, so an uncapped cache leaks
        # ~180MB per rank per new shape on a multi-resolution server.
        group = unittest.mock.MagicMock()
        group.empty_output.side_effect = lambda x, mode=0: torch.empty(0)
        usp_mod._FAST_ULYSSES_OUT_BUFFERS.clear()
        self.addCleanup(usp_mod._FAST_ULYSSES_OUT_BUFFERS.clear)
        cap = 3
        with (
            patch.object(envs, "SGLANG_DIFFUSION_FAST_ULYSSES_MAX_BUFFERS", cap),
            patch("torch.cuda.is_current_stream_capturing", return_value=False),
        ):
            for size in range(cap):
                got = usp_mod._fast_ulysses_out_buffer(
                    group, "role", torch.empty(size + 1), 0
                )
                self.assertIsNotNone(got)
            # Past the cap the exchange pays its copy-out instead of leaking.
            beyond = usp_mod._fast_ulysses_out_buffer(
                group, "role", torch.empty(cap + 99), 0
            )
        self.assertIsNone(beyond)
        self.assertEqual(len(usp_mod._FAST_ULYSSES_OUT_BUFFERS), cap)

    def test_gate_reads_no_tensor_state(self) -> None:
        # The guard against silent deadlock, asserted on the source: the gate
        # must not branch on anything a tensor carries.
        import inspect

        source = inspect.getsource(usp_mod._fast_ulysses_group)
        for forbidden in (
            "is_contiguous",
            "is_cuda",
            ".dtype",
            ".shape",
            ".stride",
            ".ndim",
            "is_compiling",
        ):
            self.assertNotIn(
                forbidden,
                source,
                f"_fast_ulysses_group must not gate on {forbidden}: a per-tensor "
                "condition can diverge across ranks and hang the collective",
            )


if __name__ == "__main__":
    unittest.main()
