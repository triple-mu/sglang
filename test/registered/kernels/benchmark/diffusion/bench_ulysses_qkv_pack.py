"""Ulysses destination-major QKV pack: cpp-jit default vs Triton fallback.

Production MiniMax-H3 768p shape: rows=20992, 56 heads x 128, bf16, with
q/k/v as row-strided split views of the fused qkv GEMM output.
"""

import importlib

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.diffusion import pack_qkv_destination_major
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=20, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

# Backend-vs-backend bench: the Triton fallback is reached by name, the
# facade only exposes the dispatching entry.
_TRITON = importlib.import_module(
    "sglang.kernels.ops.diffusion.layout.ulysses_qkv_triton"
)


def _strided_qkv(rows: int, heads: int, head_size: int):
    qkv = torch.randn(rows, 3 * heads * head_size, device="cuda", dtype=torch.bfloat16)
    hd = heads * head_size
    return (
        qkv[:, :hd].view(rows, heads, head_size),
        qkv[:, hd : 2 * hd].view(rows, heads, head_size),
        qkv[:, 2 * hd :].view(rows, heads, head_size),
    )


FN_MAP = {
    "jit": pack_qkv_destination_major,  # dispatches to the cpp-jit kernel
    "triton": _TRITON.pack_qkv_destination_major,
}


@marker.parametrize("rows", [20992, 7936], [7936])
@marker.parametrize("world", [2, 4], [2])
@marker.benchmark("impl", ["jit", "triton"])
def benchmark(rows: int, world: int, impl: str):
    heads, head_size = 56, 128
    q, k, v = _strided_qkv(rows, heads, head_size)
    out = torch.empty(
        world, rows, heads // world, 3 * head_size, device="cuda", dtype=torch.bfloat16
    )
    return marker.do_bench(
        FN_MAP[impl],
        input_args=(q, k, v, world),
        input_kwargs={"out": out},
        # q/k/v are strided views of one 900MB buffer (far beyond L2); cloning
        # them would change the strides the kernel is dispatched on.
        graph_clone_args=(),
        memory_args=(q, k, v),
        memory_output=(out,),
    )


if __name__ == "__main__":
    benchmark.run()
