import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.diffusion import (
    can_use_sage_block_sparse_attn_sm120,
    sage_block_sparse_attn_sm120,
    sage_block_sparse_dense_block_index,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.sage_sm120_reference import (
    K_BLOCK,
    quantize_sage_kv,
    quantize_sage_q,
)

register_cuda_ci(
    est_time=30, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

HEADS = 7  # MiniMax-H3: 56 heads over Ulysses 8


def _routing(batch, heads, seqlen, density, device):
    if density >= 1.0:
        return sage_block_sparse_dense_block_index(batch, heads, seqlen, seqlen, device)
    blocks = seqlen // K_BLOCK
    keep = max(1, int(blocks * density))
    g = torch.Generator(device="cpu").manual_seed(0)
    index = (
        torch.stack(
            [
                torch.randperm(blocks, generator=g)[:keep]
                for _ in range(batch * heads * blocks)
            ]
        )
        .view(batch, heads, blocks, keep)
        .to(device, torch.int32)
    )
    nums = torch.full((batch, heads, blocks), keep, dtype=torch.int32, device=device)
    return index, nums


@marker.parametrize("seqlen", [4096, 16384, 37888], [4096])
@marker.parametrize("density", [1.0, 0.5, 0.25], [0.25])
@marker.benchmark("impl", ["cake_sm120"])
def benchmark(seqlen: int, density: float, impl: str):
    if not can_use_sage_block_sparse_attn_sm120():
        return marker.skip("compute capability 12.0 only")
    device = torch.device("cuda")
    g = torch.Generator(device="cpu").manual_seed(1)
    q = torch.randn(1, HEADS, seqlen, 128, generator=g).to(device, torch.bfloat16)
    k = torch.randn(1, HEADS, seqlen, 128, generator=g).to(device, torch.bfloat16)
    v = torch.randn(1, HEADS, seqlen, 128, generator=g).to(device, torch.bfloat16)
    q_int8, q_scale = quantize_sage_q(q)
    k_int8, v_fp8, k_scale, v_scale = quantize_sage_kv(k, v)
    index, nums = _routing(1, HEADS, seqlen, density, device)
    out = torch.empty_like(q)

    def run(q_int8, k_int8, v_fp8, q_scale, k_scale, v_scale, index, nums, out):
        return sage_block_sparse_attn_sm120(
            q_int8, k_int8, v_fp8, q_scale, k_scale, v_scale, index, nums, out=out
        )

    return marker.do_bench(
        run,
        input_args=(q_int8, k_int8, v_fp8, q_scale, k_scale, v_scale, index, nums, out),
        graph_clone_args=(),
        disable_log_bandwidth=True,
    )


if __name__ == "__main__":
    benchmark.run()
