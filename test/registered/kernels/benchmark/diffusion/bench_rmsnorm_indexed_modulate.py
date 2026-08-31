"""Fused RMSNorm + indexed adaLN scale/shift (MiniMax-H3): cpp-jit vs Triton.

Production fused-adaLN loop shape: rows [20992, 5376] bf16, w_eff fp32 and
shift/gate bf16 rows indexed per token by the packed (timestep, modality)
group id.  Plan A is the bare norm+modulate, Plan B absorbs the preceding
indexed gated-residual add (residual updated in place).
"""

import importlib

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.diffusion import (
    gate_residual_rmsnorm_indexed_scale_shift_,
    rmsnorm_indexed_scale_shift,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=20, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

# Backend-vs-backend bench: the Triton fallback is reached by name, the
# facade only exposes the dispatching entry.
_TRITON = importlib.import_module(
    "sglang.kernels.ops.diffusion.norm.rmsnorm_indexed_modulate_triton"
)

PLAN_A_FN = {
    "jit": rmsnorm_indexed_scale_shift,  # dispatches to the cpp-jit kernel
    "triton": _TRITON.rmsnorm_indexed_scale_shift,
}
PLAN_B_FN = {
    "jit": gate_residual_rmsnorm_indexed_scale_shift_,
    "triton": _TRITON.gate_residual_rmsnorm_indexed_scale_shift_,
}


@marker.parametrize("plan", ["a", "b"])
@marker.parametrize("rows,groups", [(20992, 9), (4096, 6)], [(4096, 6)])
@marker.benchmark("impl", ["jit", "triton"])
def benchmark(plan: str, rows: int, groups: int, impl: str):
    hidden = 5376
    x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    update = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn(groups, hidden, device="cuda", dtype=torch.bfloat16)
    w_eff = torch.randn(groups, hidden, device="cuda", dtype=torch.float32)
    shift = torch.randn(groups, hidden, device="cuda", dtype=torch.bfloat16)
    indices = torch.randint(0, groups, (rows,), device="cuda")
    if plan == "a":
        return marker.do_bench(
            PLAN_A_FN[impl],
            input_args=(x, w_eff, shift, indices),
            input_kwargs={"eps": 1e-5},
            # x is read; group rows are tiny and L2-resident in production too.
            graph_clone_args=(0,),
        )
    return marker.do_bench(
        PLAN_B_FN[impl],
        input_args=(x, update, gate, w_eff, shift, indices),
        input_kwargs={"eps": 1e-5},
        # x (residual) is read and written in place, update is read: clone
        # both per graph iteration.  The returned (out, residual) tuple plus
        # the default memory_args count x twice -- its read and its write.
        graph_clone_args=(0, 1),
    )


if __name__ == "__main__":
    benchmark.run()
