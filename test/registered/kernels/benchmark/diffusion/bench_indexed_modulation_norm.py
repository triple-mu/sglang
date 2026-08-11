"""Fused indexed AdaLN modulation + RMSNorm vs. the eager kernel chain.

The reported GB/s counts only the *useful* traffic (the [T, H] tensors the
fused kernel has to touch), so the eager columns show the cost of the extra
intermediate passes as a lower effective bandwidth.
"""

import torch
from torch import nn

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.diffusion.indexed_modulation_norm import (
    indexed_gate_norm_scale_shift,
    indexed_norm_scale_shift,
)
from sglang.kernels.ops.diffusion.triton.indexed_modulation import (
    indexed_gate_bf16_,
    indexed_scale_shift_bf16_,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=30, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

_EPS = 1e-6
_NUM_PARAMS = 3


def eager_norm_scale_shift(x, weight, scale, shift, indices, out):
    h = nn.functional.rms_norm(x, (x.shape[-1],), weight, _EPS)
    return indexed_scale_shift_bf16_(h, shift, scale, indices)


def fused_norm_scale_shift(x, weight, scale, shift, indices, out):
    return indexed_norm_scale_shift(x, weight, scale, shift, indices, _EPS)


def eager_gate_norm_scale_shift(
    residual, update, gate, weight, scale, shift, indices, out
):
    res_out = indexed_gate_bf16_(residual, gate, update, indices)
    h = nn.functional.rms_norm(res_out, (res_out.shape[-1],), weight, _EPS)
    return indexed_scale_shift_bf16_(h, shift, scale, indices)


def fused_gate_norm_scale_shift(
    residual, update, gate, weight, scale, shift, indices, out
):
    return indexed_gate_norm_scale_shift(
        residual, update, gate, weight, scale, shift, indices, _EPS
    )


NORM_FN_MAP = {
    "eager_norm_ss": eager_norm_scale_shift,
    "fused_norm_ss": fused_norm_scale_shift,
}
GATE_FN_MAP = {
    "eager_gate_norm_ss": eager_gate_norm_scale_shift,
    "fused_gate_norm_ss": fused_gate_norm_scale_shift,
}


def _make_case(rows: int, hidden: int):
    packed = (
        torch.randn(_NUM_PARAMS, 6 * hidden, device="cuda", dtype=torch.bfloat16) * 0.5
    )
    _, _, gate, shift, scale, _ = packed.chunk(6, dim=-1)
    weight = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)
    indices = torch.randint(0, _NUM_PARAMS, (rows,), device="cuda", dtype=torch.int64)
    residual = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    update = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(residual)
    return residual, update, gate, shift, scale, weight, indices, out


# MiniMax-H3 T2VA 5s runs [9456, 5376] per rank at Ulysses degree 4.
@marker.parametrize("rows,hidden", [(9456, 5376), (2364, 5376)], [(2364, 5376)])
@marker.benchmark("impl", ["eager_norm_ss", "fused_norm_ss"])
def benchmark_norm_scale_shift(rows: int, hidden: int, impl: str):
    residual, _, _, shift, scale, weight, indices, out = _make_case(rows, hidden)
    return marker.do_bench(
        NORM_FN_MAP[impl],
        input_args=(residual, weight, scale, shift, indices, out),
        graph_clone_args=(0,),
        # Useful traffic: read x, write y.
        memory_args=(residual,),
        memory_output=(out,),
    )


@marker.parametrize("rows,hidden", [(9456, 5376), (2364, 5376)], [(2364, 5376)])
@marker.benchmark("impl", ["eager_gate_norm_ss", "fused_gate_norm_ss"])
def benchmark_gate_norm_scale_shift(rows: int, hidden: int, impl: str):
    residual, update, gate, shift, scale, weight, indices, out = _make_case(
        rows, hidden
    )
    return marker.do_bench(
        GATE_FN_MAP[impl],
        input_args=(residual, update, gate, weight, scale, shift, indices, out),
        graph_clone_args=(0, 1),
        # Useful traffic: read residual + update, write residual + y.
        memory_args=(residual, update),
        memory_output=(residual, out),
    )


if __name__ == "__main__":
    benchmark_norm_scale_shift.run()
    benchmark_gate_norm_scale_shift.run()
