import torch

from sglang.kernels.jit.benchmark import marker

# Deep backend imports on purpose: the bench compares the C++ JIT default
# against its Triton fallback, so the facade's dispatch must be bypassed.
from sglang.kernels.ops.diffusion.norm import group_norm_silu_twopass_jit as jit_backend
from sglang.kernels.ops.diffusion.norm import (
    group_norm_silu_twopass_triton as triton_backend,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=15, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

DEVICE = "cuda"
NUM_GROUPS, EPS = 32, 1e-6

FN_MAP = {
    "cpp": jit_backend.group_norm_silu_4d,
    "triton": triton_backend.group_norm_silu_4d,
}


# MiniMax-H3 VAE encoder tile sites: the dominant 256^2 clip site and the
# deepest 16^2 site (fp32 is the encoder default, bf16 is flag-gated).
@marker.parametrize(
    "n,c,h,w",
    [(17, 128, 256, 256), (9, 256, 128, 128), (5, 1024, 16, 16)],
    [(5, 1024, 16, 16)],
)
@marker.parametrize("dtype_name", ["fp32", "bf16"], ["bf16"])
@marker.benchmark("impl", ["cpp", "triton"])
def benchmark(n: int, c: int, h: int, w: int, dtype_name: str, impl: str):
    dtype = torch.float32 if dtype_name == "fp32" else torch.bfloat16
    x = torch.randn(n, c, h, w, device=DEVICE, dtype=dtype).to(
        memory_format=torch.channels_last
    )
    weight = torch.randn(c, device=DEVICE, dtype=dtype)
    bias = torch.randn(c, device=DEVICE, dtype=dtype)
    return marker.do_bench(
        FN_MAP[impl],
        input_args=(x, weight, bias, NUM_GROUPS, EPS, True),
        graph_clone_args=(0,),
        memory_args=(x,),
        memory_output="out",
    )


if __name__ == "__main__":
    with torch.inference_mode():
        benchmark.run()
