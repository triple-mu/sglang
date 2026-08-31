import torch
import torch.nn.functional as F

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.activation.activation import (
    silu_and_mul_with_activation_rounding,
)
from sglang.kernels.ops.diffusion import fused_silu_mul_per_token_quant_fp8
from sglang.kernels.ops.diffusion.activation.silu_mul_quant_triton import (
    fused_silu_mul_per_token_quant_fp8 as triton_fused_silu_mul_per_token_quant_fp8,
)
from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_quant_fp8
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=20, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

DEVICE = "cuda"
# MiniMax-H3 production fc1 output width: [gate | up] halves of 14336.
PACKED = 28672


def _eager_then_quant(x: torch.Tensor):
    # The fp8 serving chain today: eager SwiGLU (reuse_fc1_activation is off
    # under quantization) + standalone per-token quant before fc2.
    gate, up = x.chunk(2, dim=-1)
    return sglang_per_token_quant_fp8(F.silu(gate) * up)


def _act_and_mul_then_quant(x: torch.Tensor):
    return sglang_per_token_quant_fp8(silu_and_mul_with_activation_rounding(x))


FN_MAP = {
    # the public symbol dispatches C++ JIT -> Triton; the explicit line pins
    # the Triton backend for comparison
    "fused": fused_silu_mul_per_token_quant_fp8,
    "fused_triton": triton_fused_silu_mul_per_token_quant_fp8,
    "eager+quant": _eager_then_quant,
    "act_and_mul+quant": _act_and_mul_then_quant,
}


# Rows cover both reference dispatch regimes; 20992 is the production
# fl2va per-rank token count (ulysses=2).
@marker.parametrize("rows", [1797, 20992], [1797])
@marker.benchmark("impl", ["fused", "fused_triton", "eager+quant", "act_and_mul+quant"])
def benchmark(rows: int, impl: str) -> marker.BenchResult:
    x = torch.randn(rows, PACKED, dtype=torch.bfloat16, device=DEVICE)
    # All impls report the fused op's effective payload (read x, write q + s),
    # so the GB/s column compares end-to-end efficiency, not kernel count.
    return marker.do_bench(
        FN_MAP[impl],
        input_args=(x,),
        graph_clone_args=(0,),
        memory_args=(x,),
        memory_output="out",
    )


if __name__ == "__main__":
    benchmark.run()
