"""Single-GPU correctness tests for the fused single-tensor Norm+RoPE JIT kernel.

Reference replicates Wan's math in fp32: RMSNorm (eps inside rsqrt, per-channel fp32
weight), then RoPE over the full head_dim, interleaved (GPT-J) or NeoX. The kernel keeps
fp32 throughout and rounds once at the store, so the reference composes in fp32 and
rounds once at the end; agreement is then ~1 dtype ULP (rtol-based gate). cos/sin are
cos/sin of random angles (rotation is norm-preserving, no error amplification).
"""

import itertools
import sys

import pytest
import torch

from sglang.kernels.jit.utils import get_ci_test_range
from sglang.kernels.ops.diffusion.norm_rope import (
    can_use_fused_inplace_norm_rope,
    fused_inplace_norm_rope,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=40, stage="base-b-kernel-unit", runner_config="1-gpu-large")
# Nightly sets SGLANG_JIT_KERNEL_RUN_FULL_TESTS=1 to expand get_ci_test_range sweeps.
register_cuda_ci(est_time=120, suite="nightly-kernel-1-gpu", nightly=True)

DEVICE = "cuda"
EPS = 1e-6
ATOL = {torch.float16: 2e-3, torch.bfloat16: 1e-2}
RTOL = {torch.float16: 2e-3, torch.bfloat16: 1.6e-2}

DTYPES = [torch.float16, torch.bfloat16]
HEADS_LIST = get_ci_test_range([8, 40], [8, 40])
# d=512 with n=40 puts the cross-head staging at 80KB dynamic smem, exercising the
# above-48KB opt-in on capable devices (skipped below the device cap).
HEAD_DIM_LIST = get_ci_test_range([64, 128, 256, 512], [64, 128, 512])
CROSS_HEAD_LIST = [False, True]
INTERLEAVED_LIST = [True, False]


def _smem_cap() -> int:
    props = torch.cuda.get_device_properties(0)
    return getattr(props, "shared_memory_per_block_optin", 48 * 1024) - 1024


def rms_ref_f32(x, w, cross_head, eps):  # returns fp32
    b, s, n, d = x.shape
    xf = x.float()
    if not cross_head:
        var = xf.pow(2).mean(dim=-1, keepdim=True)  # over d
        return xf * torch.rsqrt(var + eps) * w.view(1, 1, 1, d)
    xf2 = xf.reshape(b, s, n * d)  # cross_head: reduce over n*d
    var = xf2.pow(2).mean(dim=-1, keepdim=True)
    return (xf2 * torch.rsqrt(var + eps) * w.view(1, 1, n * d)).reshape(b, s, n, d)


def rope_ref_f32(xf, cos, sin, interleaved):  # fp32 in, fp32 out
    b, s, n, d = xf.shape
    c = cos.view(1, s, 1, d // 2)
    sn = sin.view(1, s, 1, d // 2)
    if interleaved:
        x1, x2 = xf[..., 0::2], xf[..., 1::2]
        o1, o2 = x1 * c - x2 * sn, x1 * sn + x2 * c
        return torch.stack([o1, o2], dim=-1).flatten(-2)
    half = d // 2
    x1, x2 = xf[..., :half], xf[..., half:]
    o1, o2 = x1 * c - x2 * sn, x2 * c + x1 * sn
    return torch.cat([o1, o2], dim=-1)


def _make_inputs(dtype, n, d, cross_head):
    torch.manual_seed(0)
    b, s = 2, 16
    x = torch.randn(b, s, n, d, device=DEVICE, dtype=dtype)
    theta = torch.randn(s, d // 2, device=DEVICE, dtype=torch.float32)
    cos, sin = theta.cos().contiguous(), theta.sin().contiguous()
    weight = torch.randn(n * d if cross_head else d, device=DEVICE, dtype=torch.float32)
    return x, cos, sin, weight


@pytest.mark.parametrize(
    "dtype,n,d,cross_head,interleaved",
    list(
        itertools.product(
            DTYPES, HEADS_LIST, HEAD_DIM_LIST, CROSS_HEAD_LIST, INTERLEAVED_LIST
        )
    ),
)
def test_norm_rope(dtype, n, d, cross_head, interleaved):
    if cross_head and n * d * 4 > _smem_cap():
        pytest.skip(f"cross_head needs {n * d * 4} B smem > device cap {_smem_cap()} B")
    assert can_use_fused_inplace_norm_rope(d, n, cross_head, interleaved, dtype)
    x, cos, sin, weight = _make_inputs(dtype, n, d, cross_head)
    x0 = x.clone()
    ref = rope_ref_f32(
        rms_ref_f32(x, weight, cross_head, EPS), cos, sin, interleaved
    ).to(dtype)
    out = fused_inplace_norm_rope(
        x, weight, cos, sin, cross_head=cross_head, interleaved=interleaved, eps=EPS
    )
    assert out is None and not torch.equal(x, x0), "op must mutate x in place"
    md = (x.float() - ref.float()).abs().max().item()
    assert torch.allclose(
        x.float(), ref.float(), atol=ATOL[dtype], rtol=RTOL[dtype]
    ), f"maxdiff={md:.4e}"


def test_can_use_gate_negatives():
    assert not can_use_fused_inplace_norm_rope(96, 8, False, True, torch.bfloat16)
    assert not can_use_fused_inplace_norm_rope(2048, 8, False, True, torch.bfloat16)
    assert not can_use_fused_inplace_norm_rope(128, 8, False, True, torch.float32)
    huge_heads = _smem_cap() // (128 * 4) + 1
    assert not can_use_fused_inplace_norm_rope(
        128, huge_heads, True, True, torch.bfloat16
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
