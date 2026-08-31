"""Fused fp32 residual add + RMSNorm + autocast cast (MiniMax-H3 VAE ViT).

Production decode-tile site shape: residual [1797, 2048] fp32, branch fp16,
71 sites per tile. The eager baseline is the three-kernel aten chain the
fused kernel replaces bit-for-bit (mixed-dtype add, fp32 rms_norm, dtype
copy).
"""

import torch
import torch.nn as nn

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.diffusion import fused_residual_rmsnorm_cast_
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=10, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)


def eager_triple(residual, branch, weight, *, eps, out_dtype):
    y = residual + branch
    return nn.functional.rms_norm(y, (y.shape[-1],), weight, eps).to(out_dtype)


FN_MAP = {
    "eager": eager_triple,
    "jit": fused_residual_rmsnorm_cast_,
}


@marker.parametrize("rows,hidden", [(1797, 2048), (4096, 4096)], [(1797, 2048)])
@marker.benchmark("impl", ["jit", "eager"])
def benchmark(rows: int, hidden: int, impl: str):
    residual = torch.randn(rows, hidden, device="cuda", dtype=torch.float32)
    branch = torch.randn(rows, hidden, device="cuda", dtype=torch.float16)
    weight = torch.randn(hidden, device="cuda", dtype=torch.float32)
    return marker.do_bench(
        FN_MAP[impl],
        input_args=(residual, branch, weight),
        input_kwargs={"eps": 1e-5, "out_dtype": torch.float16},
        # residual is read and rewritten in place by the jit path; branch is
        # read. Clone both per graph iteration.
        graph_clone_args=(0, 1),
    )


if __name__ == "__main__":
    benchmark.run()
