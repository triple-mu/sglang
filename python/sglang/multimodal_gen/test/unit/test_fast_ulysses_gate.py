"""The fast-ulysses gate must decide from process-wide constants only.

Both constructing the collective and allocating the window behind a new shape
are collective. If eligibility could turn on a property of the tensor being
exchanged, one rank could take the fast path while another fell back, and that
deadlocks silently -- neither side raises and neither side times out. So the
gate is tested for what it reads, not only for what it returns.

The collective is owned by the sequence-parallel group coordinator, next to the
process group it exchanges over, so these exercise the property there rather
than a module-level cache.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.distributed.group_coordinator import (
    SequenceParallelGroupCoordinator,
)
from sglang.multimodal_gen.runtime.layers import usp as usp_mod

_USP = "sglang.multimodal_gen.runtime.layers.usp"


def _probe(world_size: int = 4):
    """A stand-in carrying only what the property reads."""
    return SimpleNamespace(
        ulysses_world_size=world_size,
        ulysses_group=object(),
        _fast_ulysses_group=None,
        _fast_ulysses_probed=False,
    )


def _resolve(obj):
    return SequenceParallelGroupCoordinator.fast_ulysses_group.fget(obj)


class TestFastUlyssesGate(unittest.TestCase):
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

        obj = _probe()
        with (
            patch.object(envs, "SGLANG_DIFFUSION_FAST_ULYSSES", False),
            patch.object(builtins, "__import__", spy),
        ):
            self.assertIsNone(_resolve(obj))
        self.assertNotIn("fast_ulysses", seen)

    def test_world_size_outside_range_returns_none(self) -> None:
        # 1 has no exchange; above 8 is structural (`BarPeers::p[8]`).
        for world_size in (1, 9, 16):
            with self.subTest(world_size=world_size):
                obj = _probe(world_size)
                with patch.object(envs, "SGLANG_DIFFUSION_FAST_ULYSSES", True):
                    self.assertIsNone(_resolve(obj))

    def test_probe_happens_once(self) -> None:
        obj = _probe(world_size=1)  # ineligible, so the probe returns early
        with patch.object(envs, "SGLANG_DIFFUSION_FAST_ULYSSES", True):
            for _ in range(5):
                _resolve(obj)
        # Caching is what keeps every rank's decision identical for the whole
        # run; re-deciding per call would reintroduce the divergence risk.
        self.assertTrue(obj._fast_ulysses_probed)

    def test_one_rank_failing_import_disables_every_rank(self) -> None:
        # The deadlock guard. This rank CAN import fast-ulysses, but a peer
        # cannot. It must not enter the constructor, which is collective: a rank
        # that entered while a peer skipped would wait forever, with no
        # exception and no timeout.
        fake_module = unittest.mock.MagicMock()
        obj = _probe()

        def vote_no(tensor, op=None, group=None):
            tensor.fill_(0)  # a peer said no

        with (
            patch.object(envs, "SGLANG_DIFFUSION_FAST_ULYSSES", True),
            patch.dict("sys.modules", {"fast_ulysses": fake_module}),
            patch("torch.cuda.current_device", return_value=0),
            patch("torch.tensor", return_value=torch.zeros(1, dtype=torch.int32)),
            patch("torch.distributed.all_reduce", vote_no),
        ):
            self.assertIsNone(_resolve(obj))
        fake_module.UlyssesGroup.assert_not_called()

    def test_refused_topology_falls_back(self) -> None:
        # fast-ulysses refuses a group that is not NVLink-joined. That must
        # degrade to the existing collective, not propagate. No second vote is
        # needed: the NVLink probe reads the same fabric on every rank.
        fake_module = unittest.mock.MagicMock()
        fake_module.UlyssesGroup.side_effect = RuntimeError("not NVLink-joined")
        obj = _probe()

        with (
            patch.object(envs, "SGLANG_DIFFUSION_FAST_ULYSSES", True),
            patch.dict("sys.modules", {"fast_ulysses": fake_module}),
            patch("torch.cuda.current_device", return_value=0),
            patch("torch.tensor", return_value=torch.ones(1, dtype=torch.int32)),
            patch("torch.distributed.all_reduce", lambda *a, **k: None),
        ):
            self.assertIsNone(_resolve(obj))

    def test_one_collective_serves_every_role(self) -> None:
        # Staging is pooled per shape inside fast-ulysses, so concurrent
        # same-shaped exchanges already get different slots. A group per role
        # would only duplicate a comm stream, a window map and a barrier state.
        fake_module = unittest.mock.MagicMock()
        built = unittest.mock.MagicMock()
        fake_module.UlyssesGroup.return_value = built
        obj = _probe()

        with (
            patch.object(envs, "SGLANG_DIFFUSION_FAST_ULYSSES", True),
            patch.dict("sys.modules", {"fast_ulysses": fake_module}),
            patch("torch.cuda.current_device", return_value=0),
            patch("torch.tensor", return_value=torch.ones(1, dtype=torch.int32)),
            patch("torch.distributed.all_reduce", lambda *a, **k: None),
        ):
            first = _resolve(obj)
            second = _resolve(obj)
        self.assertIs(first, built)
        self.assertIs(second, built)
        self.assertEqual(fake_module.UlyssesGroup.call_count, 1)

    def test_gate_reads_no_tensor_state(self) -> None:
        # The guard against silent deadlock, asserted on the source: the gate
        # must not branch on anything a tensor carries.
        import inspect

        source = inspect.getsource(
            SequenceParallelGroupCoordinator.fast_ulysses_group.fget
        )
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
                f"the gate must not read {forbidden}: a per-tensor condition can "
                "diverge across ranks and hang the collective",
            )

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


if __name__ == "__main__":
    unittest.main()
