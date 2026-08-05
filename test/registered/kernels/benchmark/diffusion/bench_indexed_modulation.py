"""Triton vs CUDA JIT for the MiniMax-H3 indexed AdaLN modulation ops.

The production shape is the packed per-rank T2VA sequence: rows=9456,
hidden=5376, 9 modulation rows.
"""

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.benchmark.utils import create_random
from sglang.kernels.ops.diffusion.indexed_modulation import (
    _jit_indexed_modulation_module,
)
from sglang.kernels.ops.diffusion.triton import indexed_modulation as triton_impl
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=20, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

NUM_PARAMS = 9


def _jit_scale_shift(x, shift, scale, indices):
    _jit_indexed_modulation_module().indexed_scale_shift(x, shift, scale, indices)


def _jit_gate(x, gate, other, indices):
    _jit_indexed_modulation_module().indexed_gate(x, gate, other, indices)


FN_MAP = {
    ("scale_shift", "jit"): _jit_scale_shift,
    ("scale_shift", "triton"): triton_impl.indexed_scale_shift_bf16_,
    ("gate", "jit"): _jit_gate,
    ("gate", "triton"): triton_impl.indexed_gate_bf16_,
}


@marker.parametrize("op", ["scale_shift", "gate"])
@marker.parametrize(
    "rows,hidden", [(9456, 5376), (2364, 5376), (9456, 3072)], [(9456, 5376)]
)
@marker.benchmark("impl", ["triton", "jit"])
def benchmark(rows: int, hidden: int, op: str, impl: str):
    x = create_random(rows, hidden)
    indices = torch.randint(0, NUM_PARAMS, (rows,), device=x.device, dtype=torch.int64)
    # The production tables are chunk() views of one AdaLN linear output.
    packed = create_random(NUM_PARAMS, 6 * hidden)
    if op == "scale_shift":
        first, second = packed.chunk(6, dim=-1)[:2]
    else:
        first, second = packed.chunk(6, dim=-1)[0], create_random(rows, hidden)

    return marker.do_bench(
        FN_MAP[(op, impl)],
        input_args=(x, first, second, indices),
        graph_clone_args=(0, 2),
        memory_args=(x, first, second, indices),
        memory_output=(x,),
    )


if __name__ == "__main__":
    benchmark.run()
