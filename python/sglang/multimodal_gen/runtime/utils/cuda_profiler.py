# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""CUDA profiler capture-range control for SGLang Diffusion.

Mirrors ``sglang.srt.utils.profile_utils._ProfilerCudart`` for the diffusion
runtime: the driver-level start/stop pair is what
``nsys profile -c cudaProfilerApi`` uses as its capture range, so wrapping only
the real request keeps model loading and warmup out of the trace.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import torch

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


@contextlib.contextmanager
def maybe_cuda_profiler_range(enabled: bool) -> Iterator[None]:
    """Bracket a block with ``cudaProfilerStart`` / ``cudaProfilerStop``.

    When ``enabled`` is ``False`` this is a no-op, so callers can pass a
    per-request gate directly. ``cudaProfilerStop`` runs from ``finally`` so a
    failing request still closes the capture range and leaves a readable
    report behind.
    """
    if not enabled:
        yield
        return
    logger.info("Call cudaProfilerStart")
    torch.cuda.cudart().cudaProfilerStart()
    try:
        yield
    finally:
        torch.cuda.cudart().cudaProfilerStop()
        logger.info("Call cudaProfilerStop")
