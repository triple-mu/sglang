"""The Nsight capture range must be opened by every rank, not just rank 0.

The diffusion GPU workers are separate spawned processes and nsys tracks a
``--capture-range=cudaProfilerApi`` range per process, so a rank that never
calls ``cudaProfilerStart`` contributes nothing to the report. Gating the
bracket on ``get_world_rank() == 0`` therefore produces a single-GPU timeline
of a multi-GPU run -- measured, with a two-process probe: 20 kernel instances
captured instead of 40.
"""

import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.multimodal_gen.runtime.pipelines_core.executors.pipeline_executor import (
    PipelineExecutor,
)
from sglang.multimodal_gen.runtime.utils.cuda_profiler import maybe_cuda_profiler_range


def _should_bracket(payload, *, enabled: bool) -> bool:
    """Call the predicate with only the collaborator it actually reads.

    ``_is_warmup_payload`` is a staticmethod, so a bare namespace carrying it
    is a complete stand-in for the executor here.
    """
    probe = SimpleNamespace(_is_warmup_payload=PipelineExecutor._is_warmup_payload)
    return PipelineExecutor._should_bracket_cuda_profiler(
        probe, payload, SimpleNamespace(enable_cuda_profiler_range=enabled)
    )


class TestCudaProfilerRange(unittest.TestCase):
    def test_every_rank_brackets_the_real_request(self) -> None:
        payload = SimpleNamespace(is_warmup=False)
        for rank in (0, 1, 3, 7):
            with self.subTest(rank=rank):
                with patch(
                    "sglang.multimodal_gen.runtime.pipelines_core.executors."
                    "pipeline_executor.get_world_rank",
                    return_value=rank,
                ):
                    self.assertTrue(_should_bracket(payload, enabled=True))

    def test_warmup_is_excluded(self) -> None:
        self.assertFalse(
            _should_bracket(SimpleNamespace(is_warmup=True), enabled=True)
        )

    def test_grouped_payload_warmup_is_excluded(self) -> None:
        batches = [SimpleNamespace(is_warmup=True), SimpleNamespace(is_warmup=True)]
        self.assertFalse(_should_bracket(batches, enabled=True))

    def test_flag_off_disables_the_bracket(self) -> None:
        self.assertFalse(
            _should_bracket(SimpleNamespace(is_warmup=False), enabled=False)
        )

    def test_bracket_does_not_gate_on_rank(self) -> None:
        # The regression lock. A rank check reintroduces the single-GPU report,
        # which is silent: the run succeeds and the trace merely omits N-1 GPUs.
        source = inspect.getsource(PipelineExecutor._should_bracket_cuda_profiler)
        body = source.split('"""')[-1]
        self.assertNotIn("get_world_rank", body)

    def test_disabled_range_does_not_touch_cudart(self) -> None:
        with patch("torch.cuda.cudart") as cudart:
            with maybe_cuda_profiler_range(False):
                pass
        cudart.assert_not_called()

    def test_enabled_range_stops_even_when_the_body_raises(self) -> None:
        # A failing request must still close the range, or the report is
        # unreadable.
        with patch("torch.cuda.cudart") as cudart:
            with self.assertRaises(RuntimeError):
                with maybe_cuda_profiler_range(True):
                    raise RuntimeError("boom")
        cudart.return_value.cudaProfilerStart.assert_called_once()
        cudart.return_value.cudaProfilerStop.assert_called_once()


if __name__ == "__main__":
    unittest.main()
