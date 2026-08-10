import unittest
from unittest import mock

import torch

from sglang.multimodal_gen.runtime.managers.gpu_worker import GPUWorker
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class _FakeReq:
    def __init__(self, *, is_warmup: bool) -> None:
        self.is_warmup = is_warmup


class TestResolvedCudaProfilerRanks(unittest.TestCase):
    def _args(self, ranks) -> ServerArgs:
        args = ServerArgs.__new__(ServerArgs)
        args.cuda_profiler_ranks = ranks
        return args

    def test_unset_defaults_to_rank_zero_only(self):
        self.assertEqual(
            self._args(None).resolved_cuda_profiler_ranks(), frozenset({0})
        )
        self.assertEqual(self._args([]).resolved_cuda_profiler_ranks(), frozenset({0}))

    def test_explicit_ranks_are_deduplicated(self):
        self.assertEqual(
            self._args([4, 0, 4]).resolved_cuda_profiler_ranks(), frozenset({0, 4})
        )


class TestCudaProfilerRange(unittest.TestCase):
    """`--enable-cuda-profiler-range` must open exactly one range per timed request.

    An `nsys profile --capture-range=cudaProfilerApi` run records nothing until
    `cudaProfilerStart` fires, so the gate here is what keeps weight loading,
    FSDP sharding and the warmup requests out of the report — and the range must
    close even when the forward raises, or the profiler keeps recording teardown.
    """

    def _worker(self, *, enabled: bool, ranks=None, rank: int = 0) -> GPUWorker:
        worker = GPUWorker.__new__(GPUWorker)
        worker.rank = rank
        server_args = ServerArgs.__new__(ServerArgs)
        server_args.enable_cuda_profiler_range = enabled
        server_args.cuda_profiler_ranks = ranks
        worker.server_args = server_args
        return worker

    def _run(self, worker: GPUWorker, req: _FakeReq, *, raises: bool = False):
        with mock.patch.object(torch.cuda, "profiler") as profiler, mock.patch(
            "sglang.multimodal_gen.runtime.managers.gpu_worker.current_platform"
        ) as platform:
            platform.is_cuda_alike.return_value = True
            if raises:
                with self.assertRaises(RuntimeError):
                    with worker._maybe_cuda_profiler_range(req):
                        raise RuntimeError("forward blew up")
            else:
                with worker._maybe_cuda_profiler_range(req):
                    pass
            return profiler

    def test_timed_request_opens_and_closes_the_range(self):
        profiler = self._run(self._worker(enabled=True), _FakeReq(is_warmup=False))
        profiler.start.assert_called_once_with()
        profiler.stop.assert_called_once_with()

    def test_warmup_request_is_not_profiled(self):
        profiler = self._run(self._worker(enabled=True), _FakeReq(is_warmup=True))
        profiler.start.assert_not_called()
        profiler.stop.assert_not_called()

    def test_disabled_flag_is_not_profiled(self):
        profiler = self._run(self._worker(enabled=False), _FakeReq(is_warmup=False))
        profiler.start.assert_not_called()
        profiler.stop.assert_not_called()

    def test_rank_outside_the_selection_stays_silent(self):
        # The range is per-process, so an unlisted rank contributes nothing to
        # the report even though the profiler followed the fork.
        profiler = self._run(
            self._worker(enabled=True, ranks=[0, 4], rank=1),
            _FakeReq(is_warmup=False),
        )
        profiler.start.assert_not_called()

        profiler = self._run(
            self._worker(enabled=True, ranks=[0, 4], rank=4),
            _FakeReq(is_warmup=False),
        )
        profiler.start.assert_called_once_with()

    def test_range_closes_when_the_forward_raises(self):
        profiler = self._run(
            self._worker(enabled=True), _FakeReq(is_warmup=False), raises=True
        )
        profiler.start.assert_called_once_with()
        profiler.stop.assert_called_once_with()

    def test_non_cuda_platform_is_not_profiled(self):
        worker = self._worker(enabled=True)
        with mock.patch.object(torch.cuda, "profiler") as profiler, mock.patch(
            "sglang.multimodal_gen.runtime.managers.gpu_worker.current_platform"
        ) as platform:
            platform.is_cuda_alike.return_value = False
            with worker._maybe_cuda_profiler_range(_FakeReq(is_warmup=False)):
                pass
        profiler.start.assert_not_called()


if __name__ == "__main__":
    unittest.main()
