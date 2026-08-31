import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.diffusion import (
    merge_two_sources_per_token_quant_fp8,
    usp_merge_heads,
    usp_merge_heads_per_token_quant_fp8,
)

# Deep import on purpose (allowlisted in test_import_surface): pins the Triton
# backend so the columns compare it against the C++ JIT default dispatch.
from sglang.kernels.ops.diffusion.layout import usp_merge_quant_triton
from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_quant_fp8
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=20, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

DEVICE = "cuda"
# MiniMax-H3 production output merge: ulysses=2, 28 local heads, head_dim 128.
WORLD, H_LOCAL, HEAD_DIM = 2, 28, 128


def _merge_then_quant(x: torch.Tensor):
    # The fp8 serving chain today: bf16 head-merge copy + standalone
    # per-token quant before out_proj.
    tokens = x.shape[1] * x.shape[2]
    return sglang_per_token_quant_fp8(usp_merge_heads(x).reshape(tokens, -1))


def _fused_two_source(x: torch.Tensor):
    # The 2-rank IPC form: the same rows as two [T, C] head-shard sources.
    tokens = x.shape[1]
    return merge_two_sources_per_token_quant_fp8(
        x[0, :, 0].reshape(tokens, -1), x[1, :, 0].reshape(tokens, -1)
    )


def _fused_two_source_triton(x: torch.Tensor):
    tokens = x.shape[1]
    return usp_merge_quant_triton.merge_two_sources_per_token_quant_fp8(
        x[0, :, 0].reshape(tokens, -1), x[1, :, 0].reshape(tokens, -1)
    )


FN_MAP = {
    "fused": usp_merge_heads_per_token_quant_fp8,
    "fused_triton": usp_merge_quant_triton.usp_merge_heads_per_token_quant_fp8,
    "merge+quant": _merge_then_quant,
    "fused_two_source": _fused_two_source,
    "fused_two_source_triton": _fused_two_source_triton,
}


# Rows cover both reference dispatch regimes; 20992 is the production
# fl2va per-rank token count (ulysses=2).
@marker.parametrize("seq", [1797, 20992], [1797])
@marker.benchmark(
    "impl",
    [
        "fused",
        "fused_triton",
        "merge+quant",
        "fused_two_source",
        "fused_two_source_triton",
    ],
)
def benchmark(seq: int, impl: str) -> marker.BenchResult:
    x = torch.randn(
        WORLD, seq, 1, H_LOCAL, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE
    )
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
