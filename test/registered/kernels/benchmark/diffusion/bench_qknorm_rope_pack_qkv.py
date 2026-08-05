"""Two-kernel QKNorm+RoPE then destination-major pack vs the fused single pass.

The chain moves q and k three times (norm+rope read, norm+rope write, pack
read); the fused kernel reads q/k/v once and writes the packed buffer once.
Inputs are strided views of one fused qkv projection output, as the DiT
produces them, so the whole qkv buffer is what gets cloned between CUDA-graph
iterations. The GB/s column counts the real traffic, which is why the chain
gets an explicit `extra_memory_footprint` for its extra q/k pass.
"""

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.benchmark.utils import create_random
from sglang.kernels.ops.diffusion.qknorm_rope import fused_inplace_qknorm_rope
from sglang.kernels.ops.diffusion.qknorm_rope_pack_qkv import (
    fused_qknorm_rope_pack_qkv,
)
from sglang.kernels.ops.diffusion.triton.ulysses_qkv import (
    pack_qkv_destination_major,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=20, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

HEAD_DIM = 128
ROPE_DIM = 96
EPS = 1e-5


def _cos_sin_cache(rope_dim: int, max_position: int) -> torch.Tensor:
    inv_freq = 1.0 / (
        10000.0
        ** (torch.arange(0, rope_dim, 2, dtype=torch.float32, device="cuda") / rope_dim)
    )
    t = torch.arange(max_position, dtype=torch.float32, device="cuda")
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(torch.bfloat16)


def _split_heads(qkv: torch.Tensor, num_heads: int):
    inner = num_heads * HEAD_DIM
    return tuple(
        chunk.view(qkv.shape[0], num_heads, HEAD_DIM)
        for chunk in qkv.split(inner, dim=-1)
    )


def chain_qknorm_rope_pack(
    qkv, q_weight, k_weight, cos_sin_cache, positions, num_heads, world_size
):
    q, k, v = _split_heads(qkv, num_heads)
    fused_inplace_qknorm_rope(
        q,
        k,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        is_neox=True,
        eps=EPS,
        head_dim=HEAD_DIM,
        rope_dim=cos_sin_cache.shape[-1],
        round_norm_before_rope=True,
    )
    return pack_qkv_destination_major(q, k, v, world_size)


def fused_qknorm_rope_pack(
    qkv, q_weight, k_weight, cos_sin_cache, positions, num_heads, world_size
):
    q, k, v = _split_heads(qkv, num_heads)
    return fused_qknorm_rope_pack_qkv(
        q,
        k,
        v,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        world_size=world_size,
        is_neox=True,
        eps=EPS,
        head_dim=HEAD_DIM,
        rope_dim=cos_sin_cache.shape[-1],
    )


FN_MAP = {
    "chain": chain_qknorm_rope_pack,
    "fused": fused_qknorm_rope_pack,
}


# MiniMax-H3 T2VA 5s on 4xH200 runs rows=9456, heads=56, world_size=4.
@marker.parametrize("world_size", [1, 2, 4, 8], [4])
@marker.parametrize("num_heads", [56], [56])
@marker.parametrize("rows", [1024, 4096, 9456, 16384], [9456])
@marker.benchmark("impl", ["chain", "fused"])
def benchmark(world_size: int, num_heads: int, rows: int, impl: str):
    qkv = create_random(rows, 3 * num_heads * HEAD_DIM)
    q_weight = create_random(HEAD_DIM)
    k_weight = create_random(HEAD_DIM)
    positions = torch.arange(rows, device="cuda", dtype=torch.int64)
    cos_sin_cache = _cos_sin_cache(ROPE_DIM, rows)

    # One q/k/v read plus the packed write; the chain additionally rewrites q
    # and k in place before the pack reads them again.
    head_bytes = rows * num_heads * HEAD_DIM * qkv.element_size()
    footprint = 6 * head_bytes
    if impl == "chain":
        footprint += 4 * head_bytes
    return marker.do_bench(
        FN_MAP[impl],
        input_args=(
            qkv,
            q_weight,
            k_weight,
            cos_sin_cache,
            positions,
            num_heads,
            world_size,
        ),
        graph_clone_args=(0,),
        memory_args=None,
        memory_output=None,
        extra_memory_footprint=footprint,
    )


if __name__ == "__main__":
    benchmark.run()
