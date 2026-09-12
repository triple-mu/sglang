# SPDX-License-Identifier: Apache-2.0
"""Request-level NVTX ranges that let Nsight Systems pick the steady-state request."""

from types import SimpleNamespace
from unittest import mock

from sglang.multimodal_gen.runtime.managers.gpu_worker import GPUWorker
from sglang.multimodal_gen.runtime.utils import nvtx_pytorch_hooks
from sglang.multimodal_gen.runtime.utils.nvtx_pytorch_hooks import (
    request_nvtx_marker,
)


def _worker(enabled: bool) -> GPUWorker:
    worker = object.__new__(GPUWorker)
    worker.server_args = SimpleNamespace(enable_nvtx_marker=enabled)
    worker.is_output_rank = True
    worker._nvtx_request_ordinal = 0
    return worker


def _run_forwards(worker: GPUWorker, warmup_flags) -> list[str]:
    pushed: list[str] = []
    with (
        mock.patch.object(nvtx_pytorch_hooks.nvtx, "range_push", pushed.append),
        mock.patch.object(nvtx_pytorch_hooks.nvtx, "range_pop", lambda: None),
    ):
        for index, is_warmup in enumerate(warmup_flags):
            req = SimpleNamespace(is_warmup=is_warmup, request_id=f"req-{index}")
            with worker._request_nvtx_range(req, [req]):
                pass
    return pushed


def test_request_ranges_count_only_real_requests():
    """`--nvtx-capture=request#2` must select the second real request: warmup
    forwards neither emit a range nor advance the ordinal."""
    pushed = _run_forwards(_worker(enabled=True), [True, False, True, False, False])
    assert pushed == [
        request_nvtx_marker(1),
        request_nvtx_marker(2),
        request_nvtx_marker(3),
    ]
    assert request_nvtx_marker(2) == "request#2"


def test_request_ranges_stay_silent_without_the_flag():
    worker = _worker(enabled=False)
    assert _run_forwards(worker, [False, False]) == []
    assert worker._nvtx_request_ordinal == 0
