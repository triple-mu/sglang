import torch
from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.diffusion import per_head_quant_fp8
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=20, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

DEVICE = "cuda"
# MiniMax-H3 production post-a2a geometry: 28 local heads x 128, fl2va
# global sequence 41984.
HEADS, HEAD_DIM = 28, 128


def _torch_chain(x: torch.Tensor):
    amax = x.float().abs().amax(dim=(0, 2))
    scale = (amax / 448.0).clamp(min=1e-12)
    payload = (
        (x.float() / scale.view(1, -1, 1)).clamp(-448, 448).to(torch.float8_e4m3fn)
    )
    return payload, scale


FN_MAP = {
    "fused": per_head_quant_fp8,
    "torch": _torch_chain,
}


@marker.parametrize("rows", [1797, 41984], [1797])
@marker.benchmark("impl", ["fused", "torch"])
def benchmark(rows: int, impl: str) -> marker.BenchResult:
    x = torch.randn(rows, HEADS, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
    return marker.do_bench(
        FN_MAP[impl],
        input_args=(x,),
        graph_clone_args=(0,),
        memory_args=(x,),
        memory_output="out",
    )


if __name__ == "__main__":
    benchmark.run()
