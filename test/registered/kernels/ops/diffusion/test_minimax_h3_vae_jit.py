# SPDX-License-Identifier: Apache-2.0
"""SM100 numerical coverage for the opt-in MiniMax-H3 VAE CUDA/C++ kernels.

References use public PyTorch operators and the released tile/window recipe.
These are operator/dataflow checks, not a substitute for video quality tests.
"""

import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.diffusion import (
    can_use_minimax_h3_vae_group_norm_silu,
    can_use_minimax_h3_vae_qk_rope,
    can_use_minimax_h3_vae_silu_quant,
    minimax_h3_vae_assemble,
    minimax_h3_vae_denorm,
    minimax_h3_vae_group_norm_silu,
    minimax_h3_vae_norm_quant,
    minimax_h3_vae_qk_rope,
    minimax_h3_vae_silu_quant,
    minimax_h3_vae_temporal_write,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.hip is not None
    or torch.cuda.get_device_capability() != (10, 0),
    reason="MiniMax-H3 VAE JIT kernels are validated on SM100 only",
)


@pytest.fixture(autouse=True)
def inference():
    torch.manual_seed(20260919)
    with torch.inference_mode():
        yield


def group_norm_reference(x, groups, weight, bias, eps, isolated):
    if isolated and x.ndim == 5:
        b, c, t, h, w = x.shape
        merged = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        normalized = F.group_norm(merged, groups, weight, bias, eps)
        normalized = normalized.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
    else:
        normalized = F.group_norm(x, groups, weight, bias, eps)
    return F.silu(normalized)


def record_error(record_property, actual, expected, prefix="output"):
    difference = (actual.float() - expected.float()).abs()
    finite = difference[torch.isfinite(difference)]
    if finite.numel():
        record_property(f"{prefix}_max_abs", finite.max().item())
        record_property(f"{prefix}_rms_abs", finite.square().mean().sqrt().item())


# Encoder widths/spatial resolutions from ch=128, ch_mult=[1,2,2,4,4,8].
# B=7 exercises the measured seven tiles/rank keyframe batch; temporal cases
# cover video use as well. Synthetic odd dimensions exercise reduction tails.
GN_CASES = [
    pytest.param((1, 128, 1, 256, 256), 32, True, False, id="encoder-128x256"),
    pytest.param((7, 128, 1, 256, 256), 32, True, False, id="encoder-seven-tiles"),
    pytest.param((1, 256, 9, 64, 64), 32, True, False, id="encoder-temporal"),
    pytest.param((7, 512, 1, 16, 16), 32, True, False, id="encoder-count-4096"),
    pytest.param((7, 1024, 1, 16, 16), 32, True, False, id="encoder-count-8192"),
    pytest.param((2, 12, 3, 17, 61), 3, True, False, id="large-partial-tail"),
    pytest.param((2, 12, 3, 17, 61), 3, False, False, id="full-gn-partial-tail"),
    pytest.param((2, 32, 3, 7, 11), 8, True, False, id="small-odd-group"),
    pytest.param((2, 32, 3, 7, 11), 8, False, False, id="full-small-group"),
    pytest.param((2, 32, 3, 9, 17), 8, True, True, id="strided-isolated"),
    pytest.param((2, 32, 3, 9, 17), 8, False, True, id="strided-full"),
    pytest.param((2, 12, 3, 17, 61), 3, True, True, id="strided-large-tail"),
    pytest.param((2, 64, 7, 9), 32, True, False, id="four-dimensional"),
]


@pytest.mark.parametrize("shape,groups,isolated,strided", GN_CASES)
def test_group_norm_silu(shape, groups, isolated, strided, record_property):
    if strided:
        b, c, t, h, w = shape
        # Channel/time permutation plus a spatial slice and nonzero offset.
        x = torch.randn(b, t, c, h + 2, 2 * w + 1, device="cuda")
        x = x.permute(0, 2, 1, 3, 4)[..., 1 : h + 1, 1::2]
        assert tuple(x.shape) == shape and not x.is_contiguous()
    else:
        x = torch.randn(shape, device="cuda")
    weight = torch.randn(shape[1], device="cuda")
    bias = torch.randn_like(weight)
    expected = group_norm_reference(x, groups, weight, bias, 1e-6, isolated)
    actual = minimax_h3_vae_group_norm_silu(
        x, groups, weight, bias, 1e-6, time_isolated=isolated
    )
    assert actual.shape == x.shape and actual.dtype == torch.float32
    assert actual.is_contiguous() and actual.data_ptr() != x.data_ptr()
    record_error(record_property, actual, expected)
    # FP32 reductions use a different tree; no FP16 tolerance is used for GN.
    torch.testing.assert_close(actual, expected, atol=4e-5, rtol=3e-5)


# Power-of-two widths cover the float4 path at 4096/8192 elements and its
# large-group split at 16384; odd widths retain general-stride coverage.
@pytest.mark.parametrize("width", [31, 1031, 1024, 2048, 4096])
@pytest.mark.parametrize("kind", ["constant", "low_variance", "nan"])
@pytest.mark.parametrize("isolated", [False, True])
def test_group_norm_moments(width, kind, isolated, record_property):
    x = torch.randn(2, 8, 3, 1, width, device="cuda")
    if kind == "constant":
        x.fill_(3.25)
    elif kind == "low_variance":
        x.mul_(1e-3).add_(4.0)
    else:
        x[0, 0, 1, 0, 0] = float("nan")
    weight = torch.linspace(0.3, 1.3, 8, device="cuda")
    bias = torch.linspace(-0.7, 0.5, 8, device="cuda")
    expected = group_norm_reference(x, 2, weight, bias, 1e-6, isolated)
    actual = minimax_h3_vae_group_norm_silu(
        x, 2, weight, bias, 1e-6, time_isolated=isolated
    )
    record_error(record_property, actual, expected)
    # At mean=4 and sigma=1e-3, one FP32 mean ULP is amplified by rsqrt(var+eps).
    # Keep a separate bound for this deliberately ill-conditioned case.
    atol = 3e-3 if kind == "low_variance" else 5e-4
    torch.testing.assert_close(actual, expected, atol=atol, rtol=1e-3, equal_nan=True)
    if kind == "constant":
        analytic = F.silu(bias)[None, :, None, None, None].expand_as(actual)
        torch.testing.assert_close(actual, analytic, atol=1e-6, rtol=1e-6)


def test_group_norm_does_not_mix_frames():
    x = torch.randn(2, 32, 3, 9, 13, device="cuda")
    weight, bias = torch.ones(32, device="cuda"), torch.zeros(32, device="cuda")
    first = minimax_h3_vae_group_norm_silu(x, 8, weight, bias, 1e-6)
    changed = x.clone()
    changed[:, :, -1].mul_(19).add_(100)
    second = minimax_h3_vae_group_norm_silu(changed, 8, weight, bias, 1e-6)
    torch.testing.assert_close(first[:, :, :-1], second[:, :, :-1], atol=0, rtol=0)


def qk_reference(qkv, cache, eps_q, eps_k):
    b, s, heads, _ = qkv.shape
    cos, sin = cache.float().reshape(b, s, 1, 48).chunk(2, dim=-1)
    outputs = []
    for value, eps in zip(qkv[..., :128].chunk(2, dim=-1), (eps_q, eps_k)):
        normalized = F.rms_norm(value.float(), (64,), eps=eps)
        left, right = normalized[..., :48].chunk(2, dim=-1)
        outputs.append(
            torch.cat(
                (
                    left * cos - right * sin,
                    right * cos + left * sin,
                    normalized[..., 48:],
                ),
                dim=-1,
            ).to(qkv.dtype)
        )
    return tuple(outputs)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1, 7, 3, 192), (2, 257, 32, 192)])
def test_joint_qk_rmsnorm_rope(dtype, shape, record_property):
    qkv = torch.randn(shape, device="cuda", dtype=dtype)
    qkv[..., :64].mul_(0.3)
    qkv[..., 64:128].mul_(7)
    angles = torch.randn(shape[0] * shape[1], 24, device="cuda")
    cache = torch.cat((angles.cos(), angles.sin()), dim=-1).to(dtype)
    expected = qk_reference(qkv, cache, 1e-5, 3e-3)
    actual = minimax_h3_vae_qk_rope(qkv, cache, 1e-5, 3e-3)
    atol, rtol = (3e-3, 2e-3) if dtype == torch.float16 else (2e-2, 8e-3)
    for name, result, reference in zip(("q", "k"), actual, expected):
        record_error(record_property, result, reference, name)
        torch.testing.assert_close(result, reference, atol=atol, rtol=rtol)


def check_fp8(q, scales, reference, record_property):
    reference = reference.reshape(q.shape)
    expected_scale = reference.abs().amax(dim=-1, keepdim=True) / 448.0
    torch.testing.assert_close(scales, expected_scale, atol=1e-8, rtol=3e-6)
    expected_q = (
        (reference / expected_scale.clamp_min(torch.finfo(torch.float32).tiny))
        .clamp(-448, 448)
        .to(torch.float8_e4m3fn)
    )
    mismatch = (q.float() != expected_q.float()).float().mean().item()
    record_property("fp8_boundary_mismatch_fraction", mismatch)
    # A different reduction tree can move values across an E4M3 midpoint.
    # Require >99.9% identical bins as well as a pointwise half-ULP error bound.
    assert mismatch <= 1e-3
    dequantized = q.float() * scales
    error = (dequantized - reference).abs()
    bound = 0.063 * reference.abs() + scales / 1024 + 1e-6
    assert torch.all(error <= bound).item()
    record_error(record_property, dequantized, reference, "dequantized")
    zero_rows = expected_scale.squeeze(-1) == 0
    if zero_rows.any():
        assert torch.count_nonzero(q.float()[zero_rows]) == 0
        assert torch.count_nonzero(scales[zero_rows]) == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("projected_dtype", [None, torch.float16, torch.float32])
def test_residual_rmsnorm_quant(dtype, projected_dtype, record_property):
    x = torch.randn(2, 7, 2048, dtype=dtype, device="cuda")
    x[0, 0].zero_()
    weight = torch.randn(2048, device="cuda")
    projected = layer_scale = None
    expected_residual = x.float()
    if projected_dtype is not None:
        projected = torch.randn_like(x, dtype=projected_dtype)
        projected[0, 0].zero_()
        layer_scale = torch.randn(2048, device="cuda") * 0.3
        expected_residual = expected_residual + projected.float() * layer_scale
    residual, q, scales = minimax_h3_vae_norm_quant(
        x, weight, 1e-5, projected, layer_scale
    )
    torch.testing.assert_close(
        residual.float(), expected_residual, atol=1e-6, rtol=2e-6
    )
    reference = F.rms_norm(expected_residual, (2048,), weight, eps=1e-5)
    check_fp8(q, scales, reference, record_property)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("projected_dtype", [torch.float16, torch.float32])
def test_final_residual_layer_norm(dtype, projected_dtype, record_property):
    x = torch.randn(2, 7, 2048, device="cuda", dtype=dtype)
    projected = torch.randn_like(x, dtype=projected_dtype)
    scale = torch.randn(2048, device="cuda") * 0.3
    weight, bias = torch.randn(2, 2048, device="cuda")
    reference = F.layer_norm(x + projected.float() * scale, (2048,), weight, bias, 1e-5)
    result = minimax_h3_vae_norm_quant(
        x, weight, 1e-5, projected, scale, bias, final=True
    )
    record_error(record_property, result, reference)
    torch.testing.assert_close(result, reference, atol=4e-6, rtol=3e-5)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows", [1, 7, 257])
def test_silu_mul_quant(dtype, rows, record_property):
    value = torch.randn(rows, 16384, device="cuda", dtype=dtype)
    if rows > 1:
        value[0].zero_()
    gate, up = value.float().chunk(2, dim=-1)
    reference = F.silu(gate) * up
    q, scales = minimax_h3_vae_silu_quant(value)
    check_fp8(q, scales, reference, record_property)


def blend_reference(previous, current, extent, dim):
    extent = min(extent, previous.shape[dim], current.shape[dim])
    if extent == 0:
        return current
    shape = [1] * current.ndim
    shape[dim] = extent
    weight = (torch.arange(extent, device=current.device) / extent).reshape(shape)
    first = current.narrow(dim, 0, extent)
    tail = previous.narrow(dim, previous.shape[dim] - extent, extent)
    mixed = tail * (1 - weight) + first * weight
    return torch.cat(
        (mixed, current.narrow(dim, extent, current.shape[dim] - extent)), dim=dim
    )


def assemble_reference(rows, y_overlap, x_overlap):
    output_rows = []
    for i, row in enumerate(rows):
        output_tiles = []
        for j, raw_tile in enumerate(row):
            tile = raw_tile
            if i:
                tile = blend_reference(rows[i - 1][j], tile, y_overlap[i - 1], -2)
            if j:
                # The checkpoint recipe uses the raw left tile at corners.
                tile = blend_reference(row[j - 1], tile, x_overlap[j - 1], -1)
            if i < len(rows) - 1:
                tile = tile[..., : -y_overlap[i], :]
            if j < len(row) - 1:
                tile = tile[..., : -x_overlap[j]]
            output_tiles.append(tile)
        output_rows.append(torch.cat(output_tiles, dim=-1))
    return torch.cat(output_rows, dim=-2)


@pytest.mark.parametrize("strided", [False, True])
def test_spatial_assembly_corners(strided, record_property):
    shape = (2, 3, 5, 9, 24) if strided else (2, 3, 5, 8, 12)
    rows = [
        [torch.randn(shape, device="cuda") + 11 * i + 3 * j for j in range(3)]
        for i in range(3)
    ]
    if strided:
        rows = [[tile[..., 1::2] for tile in row] for row in rows]
    y_overlap, x_overlap = (3, 2), (4, 8)
    expected = assemble_reference(rows, y_overlap, x_overlap)
    actual = minimax_h3_vae_assemble(rows, y_overlap, x_overlap)
    record_error(record_property, actual, expected)
    torch.testing.assert_close(actual, expected, atol=4e-6, rtol=1e-6)


def test_one_tile_assembly():
    tile = torch.randn(1, 3, 2, 7, 11, device="cuda")
    actual = minimax_h3_vae_assemble([[tile]], [], [])
    torch.testing.assert_close(actual, tile, atol=0, rtol=0)
    assert actual.data_ptr() != tile.data_ptr()


@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("extent,count", [(0, 4), (2, 4), (7, 4), (2, 0)])
def test_temporal_blend_write(strided, extent, count, record_property):
    width = 24 if strided else 12
    part = torch.randn(2, 3, 5, 8, width, device="cuda")
    overlap = torch.randn(2, 3, 3, 8, width, device="cuda")
    storage = torch.full((2, 3, 12, 8, width), -111.0, device="cuda")
    out = storage
    if strided:
        part, overlap, out = part[..., 1::2], overlap[..., 1::2], storage[..., 1::2]
    expected = out.clone()
    expected[:, :, 3 : 3 + count] = blend_reference(overlap, part, extent, 2)[
        :, :, :count
    ]
    minimax_h3_vae_temporal_write(part, overlap, extent, out, 3, count)
    record_error(record_property, out, expected)
    torch.testing.assert_close(out, expected, atol=5e-7, rtol=1e-6)
    if strided:
        assert torch.all(storage[..., ::2] == -111).item()


def test_temporal_without_overlap_and_alias_guard():
    part = torch.randn(1, 3, 5, 7, 11, device="cuda")
    out = torch.full((1, 3, 8, 7, 11), -9.0, device="cuda")
    minimax_h3_vae_temporal_write(part, None, 4, out, 2, 5)
    torch.testing.assert_close(out[:, :, 2:7], part, atol=0, rtol=0)
    with pytest.raises(ValueError, match="alias"):
        minimax_h3_vae_temporal_write(out[:, :, :5], None, 0, out, 0, 5)


@pytest.mark.parametrize("strided", [False, True])
def test_denormalize_clamp(strided, record_property):
    value = torch.randn(2, 3, 5, 8, 24 if strided else 12, device="cuda")
    if strided:
        value = value[..., 1::2]
    value[0, 0, 0, 0, :3] = torch.tensor(
        [float("nan"), float("inf"), -float("inf")], device="cuda"
    )
    mean = (-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225)
    std = (1 / 0.229, 1 / 0.224, 1 / 0.225)
    expected = (
        (value - value.new_tensor(mean)[None, :, None, None, None])
        / value.new_tensor(std)[None, :, None, None, None]
    ).clamp(0, 1)
    actual = minimax_h3_vae_denorm(value, mean, std)
    record_error(record_property, actual, expected)
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=1e-6, equal_nan=True)


def test_reject_unsupported_inputs():
    x = torch.randn(1, 32, 3, 7, 11, device="cuda")
    weight, bias = torch.ones(32, device="cuda"), torch.zeros(32, device="cuda")
    assert not can_use_minimax_h3_vae_group_norm_silu(x.half(), 8, weight, bias, 1e-6)
    assert not can_use_minimax_h3_vae_group_norm_silu(x, 7, weight, bias, 1e-6)
    with pytest.raises(ValueError):
        minimax_h3_vae_group_norm_silu(x, 8, weight, bias, 0)
    qkv = torch.randn(2, 7, 32, 192, device="cuda", dtype=torch.float16)
    cache = torch.empty(14, 48, device="cuda", dtype=torch.float16)
    assert not can_use_minimax_h3_vae_qk_rope(qkv.transpose(0, 1), cache)
    # A contiguous slice can still be misaligned for the vectorized FP16 load.
    offset_ffn = torch.empty(16385, dtype=torch.float16, device="cuda")[1:].view(
        1, 16384
    )
    assert offset_ffn.is_contiguous() and offset_ffn.data_ptr() % 16 != 0
    assert not can_use_minimax_h3_vae_silu_quant(offset_ffn)
    with pytest.raises(ValueError):
        minimax_h3_vae_silu_quant(offset_ffn)
    with torch.enable_grad():
        assert not can_use_minimax_h3_vae_qk_rope(qkv, cache)


@pytest.mark.parametrize("backend", ["TORCH_SDPA", "TORCH_CUDNN_SDPA"])
def test_two_real_decoder_blocks(backend, monkeypatch, record_property):
    """Same FP8 weights through eager blocks versus cross-block JIT dataflow."""
    from types import SimpleNamespace

    from sglang.multimodal_gen.configs.models.vaes.minimax_h3_video import (
        MiniMaxH3VideoVAEConfig,
    )
    from sglang.multimodal_gen.runtime.layers.attention.selector import (
        global_force_attn_backend_context_manager,
    )
    from sglang.multimodal_gen.runtime.managers.forward_context import (
        set_forward_context,
    )
    from sglang.multimodal_gen.runtime.models.vaes.fast_path_gate import (
        register_vae_fast_path_gate,
        use_vae_fast_path,
    )
    from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae import (
        fused_decoder,
    )
    from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.fp8 import (
        MiniMaxH3FP8Linear,
    )
    from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.optimizations import (
        MiniMaxH3VAEOptimizationState,
    )
    from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.vae_vit import (
        ViT3DDecoder,
    )
    from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.vit_utils import (
        prepare_rotary_pos_emb,
    )
    from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
    from sglang.multimodal_gen.runtime.server_args import (
        server_args as server_args_module,
    )

    # Only server configuration is supplied by the fixture; every attention,
    # quantizer, GEMM, norm and activation below is the production implementation.
    args = SimpleNamespace(
        attention_backend=None,
        attention_backend_config=None,
        enable_attention_backend_autotune=False,
        enable_breakable_cuda_graph=False,
        enable_layerwise_nvtx_marker=False,
        kv_gather_degree=1,
        sp_split_auto=False,
    )
    monkeypatch.setattr(server_args_module, "_global_server_args", args)
    with global_force_attn_backend_context_manager(AttentionBackendEnum[backend]):
        decoder = (
            ViT3DDecoder(
                num_layers=2,
                heads=32,
                dim_head=64,
                in_channels=24,
                patch_size=1,
                patch_size_t=1,
                norm_type="rms_norm",
                qk_norm_type="rms_norm",
                qk_norm_affine=False,
                ffn_activation_fn="silu",
                ffn_use_gated=True,
                rope_dim_ratio=0.75,
            )
            .cuda()
            .eval()
        )
    decoder.prepare_autocast_linear_weights(torch.float16)
    for block in decoder.transformer_blocks:
        # Constructor scales are zero: leaving them untouched would hide all
        # attention/FFN errors behind the residual identity.
        block.scale1.uniform_(0.2, 0.8)
        block.scale2.uniform_(0.2, 0.8)
        block.norm1.weight.uniform_(0.8, 1.2)
        block.norm2.weight.uniform_(0.8, 1.2)
        for owner, name in (
            (block.attn, "to_qkv"),
            (block.attn, "to_out"),
            (block.ff, "w1"),
            (block.ff, "w2"),
        ):
            setattr(owner, name, MiniMaxH3FP8Linear(getattr(owner, name)))
    decoder._sgl_fp8_installed = True
    state = MiniMaxH3VAEOptimizationState(
        MiniMaxH3VideoVAEConfig(enable_optimizations=True)
    )
    for module in decoder.modules():
        module._sgl_h3_vae_optimization = state
    register_vae_fast_path_gate(decoder, state.gate)
    value = torch.randn(2, 65, 2048, device="cuda")
    token_ids = torch.randn(2, 65, 3, device="cuda")
    rope = prepare_rotary_pos_emb(
        decoder.pos_embed(token_ids), dtype=torch.float16, allow_batched_native=True
    )

    with set_forward_context(0, None), torch.autocast("cuda", dtype=torch.float16):
        with use_vae_fast_path(decoder, False):
            reference = value
            for block in decoder.transformer_blocks:
                reference = block(reference, rope)
            reference = decoder.norm_out(reference)
        with use_vae_fast_path(decoder, True):
            assert fused_decoder.can_use_fused_decoder(decoder, value, rope)
            pending = value
            for block in decoder.transformer_blocks:
                pending = fused_decoder.block_forward(block, pending, rope)
                assert isinstance(pending, fused_decoder.ResidualPending)
            actual = fused_decoder.finish(pending, decoder.norm_out)
    assert not state.gate.enabled
    record_error(record_property, actual, reference)
    relative_rms = (
        ((actual - reference).square().mean() / reference.square().mean()).sqrt().item()
    )
    cosine = F.cosine_similarity(actual.flatten(), reference.flatten(), dim=0).item()
    record_property("relative_rms", relative_rms)
    record_property("cosine", cosine)
    # Fused producers remove intermediate FP16 rounding before FP8. This is
    # a close-contract comparison with identical deployed FP8 weights.
    assert relative_rms < 0.01
    assert cosine > 0.99995
    torch.testing.assert_close(actual, reference, atol=0.075, rtol=0.02)


@pytest.mark.parametrize("inference_weights", [False, True])
def test_output_projection_rounding_and_cache(inference_weights, record_property):
    from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.fused_decoder import (
        output_projection,
    )

    with torch.inference_mode(inference_weights):
        module = torch.nn.Linear(2048, 3072, device="cuda", dtype=torch.float32).eval()
    assert module.weight.is_inference() == inference_weights
    value = torch.randn(2, 11, 2048, device="cuda")
    flags = (
        torch.backends.cuda.matmul.allow_fp16_accumulation,
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        torch.backends.cuda.matmul.allow_tf32,
    )
    if flags[0]:
        assert output_projection(module, value) is None
        return

    def check():
        # FP64 reference isolates accumulation from intentional input/weight
        # rounding; it also works on Torch versions without mm(out_dtype=...).
        reference = (
            value.half().double().reshape(-1, 2048) @ module.weight.half().double().t()
        ).float()
        reference = (reference + module.bias).reshape(2, 11, 3072)
        actual = output_projection(module, value)
        assert actual is not None and actual.dtype == torch.float32
        record_error(record_property, actual, reference)
        torch.testing.assert_close(actual, reference, atol=2e-5, rtol=1e-5)
        return actual

    first = check()
    cached_weight = module._sgl_half_projection[1]
    second = check()
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    if not inference_weights:
        assert module._sgl_half_projection[1] is cached_weight
    module.weight.add_(0.125)
    third = check()
    assert not torch.equal(first, third)
    assert module._sgl_half_projection[1] is not cached_weight
    assert flags == (
        torch.backends.cuda.matmul.allow_fp16_accumulation,
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        torch.backends.cuda.matmul.allow_tf32,
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
