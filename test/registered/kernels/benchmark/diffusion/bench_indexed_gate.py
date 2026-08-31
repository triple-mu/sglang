"""Indexed gated residual (MiniMax-H3 adaLN): cpp-jit default vs Triton.

Production eager-loop shape: x/other [20992, 5376] bf16 in place, gate rows
indexed per token by the packed (timestep, modality) group id.
"""

import importlib

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.diffusion import indexed_gate_bf16_
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=20, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

# Backend-vs-backend bench: the Triton fallback is reached by name, the
# facade only exposes the dispatching entry.
_TRITON = importlib.import_module(
    "sglang.kernels.ops.diffusion.modulate.indexed_modulation_triton"
)

FN_MAP = {
    "jit": indexed_gate_bf16_,  # dispatches to the cpp-jit kernel
    "triton": _TRITON.indexed_gate_bf16_,
}


@marker.parametrize("rows,groups", [(20992, 9), (4096, 6)], [(4096, 6)])
@marker.benchmark("impl", ["jit", "triton"])
def benchmark(rows: int, groups: int, impl: str):
    hidden = 5376
    x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    other = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn(groups, hidden, device="cuda", dtype=torch.bfloat16)
    indices = torch.randint(0, groups, (rows,), device="cuda")
    return marker.do_bench(
        FN_MAP[impl],
        input_args=(x, gate, other, indices),
        # x is read and written in place, other is read: clone both per graph
        # iteration; gate/indices are tiny and L2-resident in production too.
        graph_clone_args=(0, 2),
        memory_output=(x,),
    )


if __name__ == "__main__":
    benchmark.run()
