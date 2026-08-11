"""Per-step vs hoisted AdaLN projection for the MiniMax-H3 DiT.

``MiniMaxH3AdalnProj`` is a pure weight stream: at the production shape it
reads a [2688, 96768] BF16 weight (520MB) to produce the 1-3 modulation rows a
denoise step needs. Running it once per block per step pays that read 49 times;
projecting every step's rows up front pays it once. Both modes here are the
tp_size=1 GEMM of ``MiniMaxH3AdalnProj.project_local`` for a single block, so
the reported latency is that block's whole-request AdaLN cost -- multiply by
num_layers (50) for the per-rank total.
"""

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.multimodal_gen.configs.models.dits.minimax_h3 import (
    MiniMaxH3DiTArchConfig,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=30, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

_ARCH = MiniMaxH3DiTArchConfig()


def _per_step(x_step: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, steps):
    for _ in range(steps):
        torch.nn.functional.linear(x_step, weight, bias)


def _hoisted(x_request: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    torch.nn.functional.linear(x_request, weight, bias)


@marker.parametrize("steps,m_per_step", [(49, 2), (49, 4), (25, 2)], [(49, 2)])
@marker.benchmark("mode", ["per_step", "hoisted"], unit="us")
def benchmark(steps: int, m_per_step: int, mode: str):
    device = torch.device("cuda")
    in_features = _ARCH.time_embed_dim
    out_features = _ARCH.adaln_out_features
    weight = torch.randn(
        out_features, in_features, device=device, dtype=torch.bfloat16
    ) * (in_features**-0.5)
    bias = torch.randn(out_features, device=device, dtype=torch.bfloat16)
    rows = m_per_step if mode == "per_step" else steps * m_per_step
    x = torch.randn(rows, in_features, device=device, dtype=torch.bfloat16)

    weight_passes = steps if mode == "per_step" else 1
    args = (x, weight, bias, steps) if mode == "per_step" else (x, weight, bias)
    return marker.do_bench(
        _per_step if mode == "per_step" else _hoisted,
        input_args=args,
        # 520MB of weights cannot live in L2, and cloning them per rotation
        # would not fit either -- the naive loop already flushes L2 per iter.
        use_cuda_graph=False,
        warmup_iters=3,
        replay_iters=20,
        memory_args=None,
        memory_output=None,
        extra_memory_footprint=weight_passes * (weight.nbytes + bias.nbytes),
    )


if __name__ == "__main__":
    benchmark.run()
