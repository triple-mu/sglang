"""NVFP4 producers of the MiniMax-H3 VAE decoder: fp32 residual + RMSNorm and
fp32 silu(gate + b) * (up + b), each ending in packed E2M1 codes with
128x4-swizzled E4M3 block scales for ``flashinfer.mm_fp4``.

The torch reference quantizer (block-16 amax, E4M3 scale ``amax * gs / 6``,
round-to-nearest E2M1, TensorRT-LLM scale swizzle, even element in the low
nibble) is pinned to ``flashinfer.fp4_quantize`` byte for byte on fp16 rows;
the producers are then held to it on the fp32 values, and every producer output
is fed to ``mm_fp4`` against an fp32 matmul so a layout mistake fails loudly.
"""

import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.diffusion import (
    can_use_minimax_h3_vae_residual_rmsnorm_fp4,
    can_use_silu_mul_quant_fp4,
    minimax_h3_vae_residual_rmsnorm_fp4,
    silu_mul_quant_fp4,
)
from sglang.kernels.ops.diffusion.common.platform import is_cuda_sm_at_least
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=40, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

pytestmark = pytest.mark.skipif(
    not is_cuda_sm_at_least((10, 0)),
    reason="MiniMax-H3 VAE NVFP4 producers are gated to SM100+",
)

WIDTH = 2048
FFN_WIDTH = 8192
EPS = 1e-5
# Spans three 128-row scale tiles, the last one partial.
ROWS = 300
GEMM_N = 512
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
E2M1_MIDPOINTS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])


@pytest.fixture(autouse=True)
def inference():
    torch.manual_seed(20260922)
    with torch.inference_mode():
        yield


def record_error(record_property, actual, expected, prefix):
    difference = (actual.float() - expected.float()).abs()
    record_property(f"{prefix}_max_abs", difference.max().item())
    record_property(f"{prefix}_rms_abs", difference.square().mean().sqrt().item())


def calibrated_scale(values, margin=1.5):
    """``448 * 6 / (amax * margin)``: the deployed input scale with its headroom."""
    amax = values.float().abs().amax().item() * margin
    return torch.tensor([448.0 * 6.0 / amax], dtype=torch.float32, device=values.device)


def swizzled_offsets(rows, num_blocks, device):
    """Flat index of (row, block) in the 128x4-swizzled ``[rows_pad, num_blocks]`` scales."""
    m = torch.arange(rows, device=device)[:, None]
    kb = torch.arange(num_blocks, device=device)[None, :]
    return (
        ((m // 128) * (num_blocks // 4) + kb // 4) * 512
        + (m % 32) * 16
        + (m % 128 // 32) * 4
        + kb % 4
    )


def unswizzle(sf, rows):
    return sf.reshape(-1)[swizzled_offsets(rows, sf.shape[-1], sf.device)]


def torch_nvfp4(reference, gs):
    """Signed E2M1 values ``[M, K]`` and row-major E4M3 scale bytes ``[M, K / 16]`` of fp32 rows."""
    rows, width = reference.shape
    blocks = reference.reshape(rows, width // 16, 16)
    amax = blocks.abs().amax(dim=-1)
    sf = (amax * (gs / 6.0)).clamp(max=448.0).to(torch.float8_e4m3fn)
    multiplier = torch.where(sf.float() == 0, torch.zeros_like(amax), gs / sf.float())
    scaled = blocks * multiplier[..., None]
    magnitude = scaled.abs().clamp(max=6.0)
    midpoints = E2M1_MIDPOINTS.to(reference.device)
    lower = torch.bucketize(magnitude, midpoints, right=False)
    upper = torch.bucketize(magnitude, midpoints, right=True)
    # Exact midpoints round to the even code.
    index = torch.where((lower != upper) & (lower % 2 == 1), upper, lower)
    values = E2M1.to(reference.device)[index] * torch.sign(scaled)
    return values.reshape(rows, width), sf.view(torch.uint8)


def decode_codes(codes):
    """Packed E2M1 bytes ``[M, K / 2]`` to signed values ``[M, K]``; even element in the low nibble."""
    nibbles = torch.stack([codes & 0xF, codes >> 4], dim=-1).reshape(codes.shape[0], -1)
    magnitude = E2M1.to(codes.device)[(nibbles & 7).long()]
    return torch.where(nibbles & 8 != 0, -magnitude, magnitude)


def check_nvfp4(codes, sf, reference, gs, record_property, prefix):
    rows, width = reference.shape
    assert codes.shape == (rows, width // 2) and codes.dtype == torch.uint8
    assert sf.shape == ((rows + 127) // 128 * 128, width // 16) and sf.dtype == torch.uint8
    expected_values, expected_sf = torch_nvfp4(reference, gs)
    actual_sf = unswizzle(sf, rows)
    actual_values = decode_codes(codes)
    sf_mismatch = (actual_sf != expected_sf).float().mean().item()
    code_mismatch = (actual_values != expected_values).float().mean().item()
    record_property(f"{prefix}_scale_mismatch_fraction", sf_mismatch)
    record_property(f"{prefix}_code_mismatch_fraction", code_mismatch)
    # A different fp32 evaluation order can move a value across an E2M1
    # midpoint or a block amax across an E4M3 midpoint; require >= 99.9% identity.
    assert sf_mismatch <= 1e-3
    assert code_mismatch <= 1e-3
    dequantized = (
        actual_values
        * actual_sf.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=-1)
        / gs
    )
    error = (dequantized - reference).abs()
    # The widest E2M1 half-step is one code unit, i.e. sf / gs; E4M3 rounding
    # of the scale widens it by at most 1/16.
    block_amax = reference.reshape(rows, -1, 16).abs().amax(-1).repeat_interleave(16, dim=-1)
    assert torch.all(error <= block_amax / 6 * (1 + 1 / 16) + 1e-6).item()
    record_error(record_property, dequantized, reference, f"{prefix}_dequantized")


def gemm_check(codes, sf, reference, gs, record_property, prefix):
    """``mm_fp4`` on the producer output against the fp32 matmul.

    NVFP4 x NVFP4 on Gaussian data lands near 0.13 relative L2; a wrong code or
    scale layout is uncorrelated with the reference and exceeds 0.5.
    """
    from flashinfer import fp4_quantize, mm_fp4

    weight = torch.randn(GEMM_N, reference.shape[1], device="cuda", dtype=torch.float16)
    weight_gs = calibrated_scale(weight, margin=1.0)
    weight_codes, weight_sf = fp4_quantize(weight, weight_gs)
    alpha = (1.0 / (gs * weight_gs)).contiguous()
    out = mm_fp4(
        codes,
        weight_codes.t(),
        sf.view(torch.float8_e4m3fn),
        weight_sf.view(torch.float8_e4m3fn).t(),
        alpha,
        torch.float16,
        backend="cudnn",
    )
    expected = reference @ weight.float().t()
    relative = ((out.float() - expected).norm() / expected.norm()).item()
    record_property(f"{prefix}_gemm_relative_l2", relative)
    assert relative < 0.3


def test_reference_quantizer_matches_flashinfer(record_property):
    """The torch quantizer and swizzle the producer checks rely on reproduce
    ``flashinfer.fp4_quantize`` (its approximate reciprocals leave rare one-bin
    differences), including the scale layout of a partial 128-row tile."""
    from flashinfer import fp4_quantize

    x = torch.randn(ROWS, WIDTH, device="cuda", dtype=torch.float16)
    x[0].zero_()
    gs = calibrated_scale(x, margin=1.0)
    codes, sf = fp4_quantize(x, gs)
    check_nvfp4(codes, sf.view(torch.uint8), x.float(), gs, record_property, "flashinfer")


@pytest.mark.parametrize("projected_dtype", [torch.float16, torch.float32])
def test_residual_rmsnorm_fp4(projected_dtype, record_property):
    x = torch.randn(ROWS, WIDTH, device="cuda")
    projected = torch.randn(ROWS, WIDTH, device="cuda", dtype=projected_dtype)
    x[0].zero_()
    projected[0].zero_()
    layer_scale = torch.randn(WIDTH, device="cuda") * 0.3
    weight = torch.randn(WIDTH, device="cuda")
    expected_residual = x + projected.float() * layer_scale
    reference = F.rms_norm(expected_residual, (WIDTH,), weight, eps=EPS)
    gs = calibrated_scale(reference)
    assert can_use_minimax_h3_vae_residual_rmsnorm_fp4(
        x, projected, layer_scale, weight, eps=EPS, global_scale=gs.item()
    )
    residual, codes, sf = minimax_h3_vae_residual_rmsnorm_fp4(
        x, projected, layer_scale, weight, eps=EPS, global_scale=gs.item()
    )
    assert residual.shape == x.shape and residual.dtype == torch.float32
    torch.testing.assert_close(residual, expected_residual, atol=1e-6, rtol=2e-6)
    check_nvfp4(codes, sf, reference, gs, record_property, "rmsnorm")
    gemm_check(codes, sf, reference, gs, record_property, "rmsnorm")


def test_residual_rmsnorm_fp4_flattens_leading_dims(record_property):
    x = torch.randn(2, 7, WIDTH, device="cuda")
    projected = torch.randn(2, 7, WIDTH, device="cuda", dtype=torch.float16)
    layer_scale = torch.randn(WIDTH, device="cuda") * 0.3
    weight = torch.randn(WIDTH, device="cuda")
    reference = F.rms_norm(x + projected.float() * layer_scale, (WIDTH,), weight, eps=EPS)
    gs = calibrated_scale(reference)
    residual, codes, sf = minimax_h3_vae_residual_rmsnorm_fp4(
        x, projected, layer_scale, weight, eps=EPS, global_scale=gs.item()
    )
    assert residual.shape == x.shape
    check_nvfp4(codes, sf, reference.reshape(14, WIDTH), gs, record_property, "rmsnorm_3d")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows", [1, ROWS])
def test_silu_mul_quant_fp4(dtype, rows, record_property):
    x = torch.randn(rows, 2 * FFN_WIDTH, device="cuda", dtype=dtype)
    gate_bias, up_bias = torch.randn(2, FFN_WIDTH, device="cuda") * 0.5
    gate, up = x.float().chunk(2, dim=-1)
    reference = F.silu(gate + gate_bias) * (up + up_bias)
    gs = calibrated_scale(reference)
    assert can_use_silu_mul_quant_fp4(x, gate_bias, up_bias, global_scale=gs.item())
    codes, sf = silu_mul_quant_fp4(
        x, global_scale=gs.item(), gate_bias=gate_bias, up_bias=up_bias
    )
    check_nvfp4(codes, sf, reference, gs, record_property, "silu")
    if rows > 1:
        gemm_check(codes, sf, reference, gs, record_property, "silu")


def test_reject_unsupported_inputs():
    x = torch.randn(2, 7, WIDTH, device="cuda")
    projected = torch.randn_like(x, dtype=torch.float16)
    vector = torch.randn(WIDTH, device="cuda")
    ok = dict(eps=EPS, global_scale=100.0)
    assert can_use_minimax_h3_vae_residual_rmsnorm_fp4(x, projected, vector, vector, **ok)
    assert not can_use_minimax_h3_vae_residual_rmsnorm_fp4(
        x, projected, vector, vector, eps=EPS, global_scale=0.0
    )
    assert not can_use_minimax_h3_vae_residual_rmsnorm_fp4(
        x, projected, vector, vector.half(), **ok
    )
    with pytest.raises(RuntimeError, match="global scale"):
        minimax_h3_vae_residual_rmsnorm_fp4(
            x, projected, vector, vector, eps=EPS, global_scale=-1.0
        )
    ffn = torch.randn(3, 2 * FFN_WIDTH, device="cuda", dtype=torch.float16)
    gate_bias, up_bias = torch.randn(2, FFN_WIDTH, device="cuda")
    assert can_use_silu_mul_quant_fp4(ffn, gate_bias, up_bias, global_scale=100.0)
    assert not can_use_silu_mul_quant_fp4(ffn, gate_bias.half(), up_bias, global_scale=100.0)
    assert not can_use_silu_mul_quant_fp4(
        ffn, gate_bias[: FFN_WIDTH // 2], up_bias, global_scale=100.0
    )
    assert not can_use_silu_mul_quant_fp4(ffn.float(), gate_bias, up_bias, global_scale=100.0)
    with pytest.raises(RuntimeError):
        silu_mul_quant_fp4(ffn, global_scale=100.0, gate_bias=gate_bias.half(), up_bias=up_bias)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
