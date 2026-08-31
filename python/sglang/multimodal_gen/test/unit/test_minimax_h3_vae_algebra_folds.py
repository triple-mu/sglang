# SPDX-License-Identifier: Apache-2.0
"""Old-path vs folded-path exactness for the MiniMax H3 VAE weight folds.

Each fold rewrites checkpoint values so the runtime modules skip the absorbed
op (LayerScale, 1x1x1 quant convs, pixel normalize/denormalize, norm_out
affine). These tests rebuild the pre-fold op chain from the raw checkpoint
values and assert the folded modules reproduce it.
"""

import copy

import pytest
import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.base_module import (
    TransformerBlock,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.conv import (
    BaseConv3d,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.klvae import (
    AutoencoderKLLegacy,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.processor import (
    get_denormalize_transform,
    get_norm_constants,
    get_normalize_transform,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.vae_vit import (
    _pack_tensors_3d,
    _unpack_tensors_3d,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.vit_utils import (
    create_token_ids,
    prepare_rotary_pos_emb,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.weight_folds import (
    CONV_IN_PIXEL_NORM_FOLDED_KEY,
    PROJ_OUT_PIXEL_DENORM_FOLDED_KEY,
    apply_minimax_h3_vae_weight_folds_,
    fold_layer_scales_,
    fold_norm_out_and_pixel_denorm_into_proj_out_,
    fold_pixel_norm_into_conv_in_,
    fold_post_quant_conv_into_x_embedder_,
    fold_quant_conv_into_conv_out_,
)

_MEAN, _STD = get_norm_constants("imagenet")


@pytest.fixture(autouse=True)
def _strict_fp32():
    """TF32 would put fp32 comparisons at 2^-11 rounding; pin true fp32."""
    if not torch.cuda.is_available():
        yield
        return
    previous_matmul = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_matmul
        torch.backends.cudnn.allow_tf32 = previous_cudnn


_BLOCK_KWARGS = dict(
    heads=2,
    dim_head=8,
    norm_type="rms_norm",
    norm_affine=True,
    qk_norm_type="rms_norm",
    qk_norm_affine=False,
    ffn_activation_fn="silu",
    ffn_use_gated=True,
    use_scale=True,
    bias=True,
    eps=1e-5,
)


def _log_uniform(shape, low, high, generator=None):
    exponent = torch.empty(shape).uniform_(
        torch.log10(torch.tensor(low)).item(),
        torch.log10(torch.tensor(high)).item(),
        generator=generator,
    )
    return torch.pow(10.0, exponent)


def _random_layer_scales(dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    # Real H3 checkpoint LayerScale spans ~9e-9..0.10; cover both extremes.
    scale1 = _log_uniform((dim,), 1e-8, 0.1)
    scale2 = _log_uniform((dim,), 1e-8, 0.1)
    return scale1, scale2


def _make_block(device: torch.device, dtype=torch.float32) -> TransformerBlock:
    block = TransformerBlock(**_BLOCK_KWARGS).to(device=device, dtype=dtype)
    # F.scaled_dot_product_attention runs for every dtype so old/new use the
    # same attention kernel; USPAttention's fp16 path needs a forward context.
    block.attn.attn = None
    for parameter in block.parameters():
        parameter.data.normal_(0.0, 0.2)
    return block


def _old_block_forward(block, x, scale1, scale2):
    """Pre-fold TransformerBlock math: scaled residual adds."""
    norm_hidden = block.norm1(x.float()).to(x.dtype)
    attn_out = block.attn(norm_hidden, None)
    x = x + attn_out * scale1
    norm_hidden = block.norm2(x.float()).to(x.dtype)
    ff_out = block.ff(norm_hidden)
    return x + ff_out * scale2


def _folded_block_from(block, scale1, scale2):
    state_dict = {
        f"decoder.transformer_blocks.0.{key}": value.detach().clone()
        for key, value in block.state_dict().items()
    }
    state_dict["decoder.transformer_blocks.0.scale1"] = scale1.clone()
    state_dict["decoder.transformer_blocks.0.scale2"] = scale2.clone()
    assert fold_layer_scales_(state_dict, prefix="") == 2
    folded = copy.deepcopy(block)
    folded.load_state_dict(
        {
            key[len("decoder.transformer_blocks.0.") :]: value
            for key, value in state_dict.items()
        }
    )
    assert torch.equal(folded.scale1, torch.ones_like(folded.scale1))
    return folded, state_dict


class TestLayerScaleFold:
    def test_fp32_matches_old_path(self):
        torch.manual_seed(0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        block = _make_block(device)
        scale1, scale2 = _random_layer_scales(16)
        scale1, scale2 = scale1.to(device), scale2.to(device)
        folded, _ = _folded_block_from(block, scale1, scale2)
        x = torch.randn(1, 32, 16, device=device)
        with torch.inference_mode():
            old = _old_block_forward(block, x, scale1, scale2)
            new = folded(x)
        # Folding moves one fp32 multiply across the GEMM; error is a few
        # fp32 roundings of the branch output.
        torch.testing.assert_close(new, old, rtol=2e-6, atol=1e-6)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA fp16")
    def test_fp16_storage_no_worse_than_unfolded(self):
        torch.manual_seed(1)
        device = torch.device("cuda")
        block = _make_block(device)
        scale1, scale2 = _random_layer_scales(16)
        scale1, scale2 = scale1.to(device), scale2.to(device)
        folded, _ = _folded_block_from(block, scale1, scale2)
        x = torch.randn(1, 32, 16, device=device)

        def _cast_linears(module, dtype):
            for linear in (
                module.attn.to_qkv,
                module.attn.to_out,
                module.ff.w1,
                module.ff.w2,
            ):
                linear.to(dtype=dtype)

        with torch.inference_mode():
            # fp32-storage old path is the numeric ground truth.
            reference = _old_block_forward(block, x, scale1, scale2)
            _cast_linears(block, torch.float16)
            _cast_linears(folded, torch.float16)
            with torch.autocast("cuda", dtype=torch.float16):
                old = _old_block_forward(block, x, scale1, scale2)
                new = folded(x)

        error_old = (old - reference).abs().max().item()
        error_new = (new - reference).abs().max().item()
        # fp16 stores round scale*W once instead of rounding W and scaling in
        # fp32; both paths sit at fp16-rounding distance from the fp32 result.
        assert error_new <= 2.0 * error_old + 1e-6, (error_new, error_old)
        torch.testing.assert_close(new, old, rtol=0.0, atol=8.0 * error_old + 1e-6)


class TestPostQuantConvFold:
    def test_matches_conv_then_linear(self):
        torch.manual_seed(2)
        conv = torch.nn.Conv3d(6, 6, 1)
        linear = torch.nn.Linear(6, 20)
        z = torch.randn(1, 6, 3, 4, 4)
        state_dict = {
            "post_quant_conv.weight": conv.weight.detach().clone(),
            "post_quant_conv.bias": conv.bias.detach().clone(),
            "decoder.x_embedder.weight": linear.weight.detach().clone(),
            "decoder.x_embedder.bias": linear.bias.detach().clone(),
        }
        assert fold_post_quant_conv_into_x_embedder_(state_dict, prefix="")
        with torch.inference_mode():
            old = linear(_pack_tensors_3d(conv(z), 1, 1))
            new = F.linear(
                _pack_tensors_3d(z, 1, 1),
                state_dict["decoder.x_embedder.weight"],
                state_dict["decoder.x_embedder.bias"],
            )
        torch.testing.assert_close(new, old, rtol=1e-5, atol=1e-6)
        identity = torch.eye(6).view(6, 6, 1, 1, 1)
        assert torch.equal(state_dict["post_quant_conv.weight"], identity)
        assert torch.equal(
            state_dict["post_quant_conv.bias"], torch.zeros_like(conv.bias)
        )


class TestQuantConvFold:
    def test_matches_conv_then_pointwise(self):
        torch.manual_seed(3)
        conv_out = torch.nn.Conv3d(10, 8, 3, padding=1)
        quant = torch.nn.Conv3d(8, 8, 1)
        x = torch.randn(1, 10, 4, 6, 6)
        state_dict = {
            "encoder.conv_out.weight": conv_out.weight.detach().clone(),
            "encoder.conv_out.bias": conv_out.bias.detach().clone(),
            "quant_conv.weight": quant.weight.detach().clone(),
            "quant_conv.bias": quant.bias.detach().clone(),
        }
        assert fold_quant_conv_into_conv_out_(state_dict, prefix="")
        with torch.inference_mode():
            old = quant(conv_out(x))
            new = F.conv3d(
                x,
                state_dict["encoder.conv_out.weight"],
                state_dict["encoder.conv_out.bias"],
                padding=1,
            )
        torch.testing.assert_close(new, old, rtol=1e-5, atol=1e-6)


class TestPixelNormFold:
    @pytest.mark.parametrize("frames", [1, 5])
    def test_matches_normalize_then_conv(self, frames):
        torch.manual_seed(4)
        conv = BaseConv3d(
            3,
            8,
            kernel_size=3,
            padding=1,
            padding_mode="reflect",
            padding_mode_t=None,
            causal=True,
        )
        state_dict = {
            "encoder.conv_in.weight": conv.weight.detach().clone(),
            "encoder.conv_in.bias": conv.bias.detach().clone(),
        }
        assert fold_pixel_norm_into_conv_in_(
            state_dict, prefix="", mean=_MEAN, std=_STD
        )
        assert bool(state_dict[CONV_IN_PIXEL_NORM_FOLDED_KEY])
        folded = BaseConv3d(
            3,
            8,
            kernel_size=3,
            padding=1,
            padding_mode="reflect",
            padding_mode_t=None,
            causal=True,
        )
        with torch.no_grad():
            folded.weight.copy_(state_dict["encoder.conv_in.weight"])
            folded.bias.copy_(state_dict["encoder.conv_in.bias"])
        # Constant temporal padding must inject the raw value that normalized
        # to zero; production sets this in AutoencoderKLLegacy.__init__.
        folded.temporal_pad_values = torch.tensor(_MEAN, dtype=torch.float32)

        x_raw = torch.rand(1, 3, frames, 12, 12)
        frames_2d = x_raw.transpose(1, 2).reshape(frames, 3, 12, 12)
        x_norm = (
            get_normalize_transform("imagenet")(frames_2d)
            .reshape(1, frames, 3, 12, 12)
            .transpose(1, 2)
            .contiguous()
        )
        with torch.inference_mode():
            old = conv(x_norm)
            new = folded(x_raw)
        torch.testing.assert_close(new, old, rtol=1e-4, atol=1e-5)


class TestProjOutFold:
    def test_matches_affine_norm_then_proj_then_denorm(self):
        torch.manual_seed(5)
        dim = 32
        patch_size, patch_size_t = 2, 2
        rows_per_channel = patch_size_t * patch_size * patch_size
        patch_dim = 3 * rows_per_channel
        latent_t, latent_h, latent_w = 2, 2, 2
        tokens = latent_t * latent_h * latent_w

        norm = torch.nn.LayerNorm(dim, elementwise_affine=True, eps=1e-5)
        proj = torch.nn.Linear(dim, patch_dim)
        with torch.no_grad():
            norm.weight.normal_(1.0, 0.2)
            norm.bias.normal_(0.0, 0.05)
        state_dict = {
            "decoder.norm_out.weight": norm.weight.detach().clone(),
            "decoder.norm_out.bias": norm.bias.detach().clone(),
            "decoder.proj_out.weight": proj.weight.detach().clone(),
            "decoder.proj_out.bias": proj.bias.detach().clone(),
        }
        assert fold_norm_out_and_pixel_denorm_into_proj_out_(
            state_dict,
            prefix="",
            mean=_MEAN,
            std=_STD,
            rows_per_channel=rows_per_channel,
        )
        assert "decoder.norm_out.weight" not in state_dict
        assert "decoder.norm_out.bias" not in state_dict
        assert bool(state_dict[PROJ_OUT_PIXEL_DENORM_FOLDED_KEY])

        hidden = torch.randn(1, tokens, dim)
        video_shape = (
            latent_t * patch_size_t,
            latent_h * patch_size,
            latent_w * patch_size,
        )
        with torch.inference_mode():
            old = _unpack_tensors_3d(
                proj(norm(hidden)), patch_size, patch_size_t, *video_shape
            )
            old_frames = old.transpose(1, 2).reshape(-1, 3, *video_shape[1:])
            old_pixels = (
                get_denormalize_transform("imagenet")(old_frames)
                .reshape(1, video_shape[0], 3, *video_shape[1:])
                .transpose(1, 2)
            )
            norm_free = F.layer_norm(hidden, (dim,), eps=1e-5)
            new = _unpack_tensors_3d(
                F.linear(
                    norm_free,
                    state_dict["decoder.proj_out.weight"],
                    state_dict["decoder.proj_out.bias"],
                ),
                patch_size,
                patch_size_t,
                *video_shape,
            )
        torch.testing.assert_close(new, old_pixels, rtol=1e-4, atol=1e-5)


def _synthetic_fold_state_dict() -> dict:
    torch.manual_seed(6)
    state_dict = {
        "decoder.transformer_blocks.0.scale1": _log_uniform((16,), 1e-8, 0.1),
        "decoder.transformer_blocks.0.scale2": _log_uniform((16,), 1e-8, 0.1),
        "decoder.transformer_blocks.0.attn.to_out.weight": torch.randn(16, 16),
        "decoder.transformer_blocks.0.attn.to_out.bias": torch.randn(16),
        "decoder.transformer_blocks.0.ff.w2.weight": torch.randn(16, 64),
        "decoder.transformer_blocks.0.ff.w2.bias": torch.randn(16),
        "post_quant_conv.weight": torch.randn(6, 6, 1, 1, 1),
        "post_quant_conv.bias": torch.randn(6),
        "decoder.x_embedder.weight": torch.randn(16, 6),
        "decoder.x_embedder.bias": torch.randn(16),
        "quant_conv.weight": torch.randn(8, 8, 1, 1, 1),
        "quant_conv.bias": torch.randn(8),
        "encoder.conv_out.weight": torch.randn(8, 10, 3, 3, 3),
        "encoder.conv_out.bias": torch.randn(8),
        "encoder.conv_in.weight": torch.randn(8, 3, 3, 3, 3),
        "encoder.conv_in.bias": torch.randn(8),
        "decoder.norm_out.weight": torch.randn(16).abs() + 0.5,
        "decoder.norm_out.bias": torch.randn(16) * 0.05,
        "decoder.proj_out.weight": torch.randn(24, 16),
        "decoder.proj_out.bias": torch.randn(24),
    }
    return state_dict


def _apply_all_folds(state_dict: dict) -> dict:
    apply_minimax_h3_vae_weight_folds_(
        state_dict,
        prefix="",
        pixel_mean=_MEAN,
        pixel_std=_STD,
        proj_rows_per_channel=8,
    )
    return state_dict


class TestFoldIdempotence:
    def test_second_application_is_identity(self):
        once = _apply_all_folds(_synthetic_fold_state_dict())
        twice = _apply_all_folds(copy.deepcopy(once))
        assert set(once) == set(twice)
        for key in once:
            assert torch.equal(once[key], twice[key]), key

    def test_markers_and_neutral_elements(self):
        folded = _apply_all_folds(_synthetic_fold_state_dict())
        assert bool(folded[CONV_IN_PIXEL_NORM_FOLDED_KEY])
        assert bool(folded[PROJ_OUT_PIXEL_DENORM_FOLDED_KEY])
        assert "decoder.norm_out.weight" not in folded
        assert "decoder.norm_out.bias" not in folded
        assert torch.equal(
            folded["decoder.transformer_blocks.0.scale1"], torch.ones(16)
        )
        assert torch.equal(
            folded["quant_conv.weight"], torch.eye(8).view(8, 8, 1, 1, 1)
        )
        assert torch.equal(
            folded["post_quant_conv.weight"], torch.eye(6).view(6, 6, 1, 1, 1)
        )

    def test_partial_store_mapping_passes_through(self):
        # The fp16 decode-dtype store carries only block linears; no fold may
        # touch them without their scale/source keys.
        store = {
            "decoder.transformer_blocks.0.attn.to_out.weight": torch.randn(
                16, 16, dtype=torch.float16
            ),
            "decoder.transformer_blocks.0.ff.w2.weight": torch.randn(
                16, 64, dtype=torch.float16
            ),
        }
        before = copy.deepcopy(store)
        _apply_all_folds(store)
        assert set(store) == set(before)
        for key in before:
            assert torch.equal(store[key], before[key])


def _small_legacy_vae(device: torch.device) -> AutoencoderKLLegacy:
    model = AutoencoderKLLegacy(
        in_channels=3,
        out_ch=3,
        ch=32,
        embed_dim=4,
        z_channels=4,
        use_3d_conv=True,
        zq_ch_encoder=None,
        zq_ch_decoder=None,
        num_res_blocks=1,
        ch_mult=[1, 1],
        space_down=[2, 2],
        space_up=[1, 2],
        time_down=[1, 2],
        time_up=None,
        padding_mode="reflect",
        padding_mode_t=None,
        use_t_isolated_gn=True,
        causal_encoder=True,
        causal_decoder=False,
        use_vit_decoder=True,
        vit_decoder_kwargs=dict(
            dim_head=8,
            heads=2,
            num_layers=2,
            norm_type="rms_norm",
            norm_affine=True,
            qk_norm_type="rms_norm",
            qk_norm_affine=False,
            ffn_activation_fn="silu",
            ffn_use_gated=True,
            rope_dim_ratio=0.75,
            rope_theta=100.0,
        ),
        shift_factor=0.0,
        scaling_factor=1.0,
        pixel_norm_type="imagenet",
        clip_length=4,
        token_drop=0,
    )
    return model.to(device).eval()


def _synthetic_checkpoint(model: AutoencoderKLLegacy) -> dict:
    # Checkpoints load from CPU mappings; folds run on one device.
    state_dict = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    dim = 16
    for block_index in range(2):
        prefix = f"decoder.transformer_blocks.{block_index}."
        state_dict[prefix + "scale1"] = _log_uniform((dim,), 1e-4, 0.1)
        state_dict[prefix + "scale2"] = _log_uniform((dim,), 1e-4, 0.1)
    state_dict["decoder.norm_out.weight"] = torch.randn(dim).abs() + 0.5
    state_dict["decoder.norm_out.bias"] = torch.randn(dim) * 0.05
    return state_dict


def _copy_raw_checkpoint_(model: AutoencoderKLLegacy, checkpoint: dict) -> None:
    """Load checkpoint values while bypassing the fold hook."""
    target = model.state_dict()
    with torch.no_grad():
        for key, value in checkpoint.items():
            if key in target:
                target[key].copy_(value)


def _old_decoder_forward(decoder, checkpoint, z2, device):
    """Pre-fold ViT decoder math driven by raw checkpoint values."""
    batch = z2.shape[0]
    latent_size = tuple(z2.shape[2:])
    hidden = _pack_tensors_3d(z2, 1, 1)
    hidden = F.linear(
        hidden,
        checkpoint["decoder.x_embedder.weight"].to(device),
        checkpoint["decoder.x_embedder.bias"].to(device),
    )
    num_patches = hidden.shape[1]
    tokens = [hidden, decoder.register_tokens.expand(batch, -1, -1)]
    tokens.append(torch.zeros_like(hidden[:, 0:1, :]))
    hidden = torch.cat(tokens, dim=1)

    num_suffix = 1 + decoder.num_register_tokens
    img_ids = create_token_ids(latent_size, device, z2.dtype).expand(batch, -1, -1)
    suffix_ids = torch.zeros((batch, num_suffix, 3), device=device, dtype=img_ids.dtype)
    img_ids = torch.cat([img_ids, suffix_ids], dim=1)
    rotary = prepare_rotary_pos_emb(decoder.pos_embed(img_ids), dtype=hidden.dtype)

    for index, block in enumerate(decoder.transformer_blocks):
        prefix = f"decoder.transformer_blocks.{index}."
        scale1 = checkpoint[prefix + "scale1"].to(device)
        scale2 = checkpoint[prefix + "scale2"].to(device)
        norm_hidden = block.norm1(hidden.float()).to(hidden.dtype)
        hidden = hidden + block.attn(norm_hidden, rotary) * scale1
        norm_hidden = block.norm2(hidden.float()).to(hidden.dtype)
        hidden = hidden + block.ff(norm_hidden) * scale2

    hidden = F.layer_norm(
        hidden,
        (hidden.shape[-1],),
        checkpoint["decoder.norm_out.weight"].to(device),
        checkpoint["decoder.norm_out.bias"].to(device),
        eps=1e-5,
    )
    output = F.linear(
        hidden,
        checkpoint["decoder.proj_out.weight"].to(device),
        checkpoint["decoder.proj_out.bias"].to(device),
    )
    output = output[:, :num_patches, :]
    patch_size = decoder.config.patch_size
    patch_size_t = decoder.config.patch_size_t
    return _unpack_tensors_3d(
        output,
        patch_size,
        patch_size_t,
        latent_size[0] * patch_size_t,
        latent_size[1] * patch_size,
        latent_size[2] * patch_size,
    )


class TestLegacyVAELoadIntegration:
    @pytest.fixture()
    def loaded_pair(self):
        torch.manual_seed(7)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_new = _small_legacy_vae(device)
        checkpoint = _synthetic_checkpoint(model_new)
        model_new.load_state_dict(
            {key: value.clone() for key, value in checkpoint.items()}
        )
        model_old = _small_legacy_vae(device)
        _copy_raw_checkpoint_(model_old, checkpoint)
        # The old reference consumes normalized pixels, so its conv_in pads
        # the normalized zero.
        model_old.encoder.conv_in.temporal_pad_values = None
        return model_new, model_old, checkpoint, device

    def test_markers_and_neutralized_weights(self, loaded_pair):
        model_new, _, _, _ = loaded_pair
        assert bool(model_new.conv_in_pixel_norm_folded)
        assert bool(model_new.proj_out_pixel_denorm_folded)
        eye = torch.eye(8, device=model_new.quant_conv.weight.device)
        assert torch.equal(model_new.quant_conv.weight, eye.view(8, 8, 1, 1, 1))
        for block in model_new.decoder.transformer_blocks:
            assert torch.equal(block.scale1, torch.ones_like(block.scale1))
            assert torch.equal(block.scale2, torch.ones_like(block.scale2))

    def test_encode_matches_old_chain(self, loaded_pair):
        model_new, model_old, checkpoint, device = loaded_pair
        torch.manual_seed(8)
        x_raw = torch.rand(1, 3, 5, 16, 16, device=device)
        frames = x_raw.transpose(1, 2).reshape(5, 3, 16, 16)
        x_norm = (
            get_normalize_transform("imagenet")(frames)
            .reshape(1, 5, 3, 16, 16)
            .transpose(1, 2)
            .contiguous()
        )
        with torch.inference_mode():
            old = F.conv3d(
                model_old.encoder(x_norm),
                checkpoint["quant_conv.weight"].to(device),
                checkpoint["quant_conv.bias"].to(device),
            )
            new = model_new.encode(x_raw)
        torch.testing.assert_close(new, old, rtol=1e-4, atol=1e-5)

    def test_decode_matches_old_chain(self, loaded_pair):
        model_new, model_old, checkpoint, device = loaded_pair
        torch.manual_seed(9)
        z = torch.randn(1, 4, 2, 4, 4, device=device)
        with torch.inference_mode():
            z2 = F.conv3d(
                z,
                checkpoint["post_quant_conv.weight"].to(device),
                checkpoint["post_quant_conv.bias"].to(device),
            )
            old = _old_decoder_forward(model_old.decoder, checkpoint, z2, device)
            frames = old.transpose(1, 2).reshape(-1, 3, 16, 16)
            old_pixels = (
                get_denormalize_transform("imagenet")(frames)
                .reshape(1, old.shape[2], 3, 16, 16)
                .transpose(1, 2)
            )
            new = model_new.decode(z)
        torch.testing.assert_close(new, old_pixels, rtol=1e-4, atol=1e-5)

    def test_reload_same_checkpoint_is_stable(self, loaded_pair):
        model_new, _, checkpoint, _ = loaded_pair
        first = {
            key: value.detach().clone() for key, value in model_new.state_dict().items()
        }
        model_new.load_state_dict(
            {key: value.clone() for key, value in checkpoint.items()}
        )
        second = model_new.state_dict()
        assert set(first) == set(second)
        for key in first:
            assert torch.equal(first[key], second[key]), key

    def test_unfolded_weights_fail_loud(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _small_legacy_vae(device)
        with pytest.raises(RuntimeError, match="not folded"):
            model.encode(torch.rand(1, 3, 5, 16, 16, device=device))
        with pytest.raises(RuntimeError, match="not folded"):
            model.decode(torch.randn(1, 4, 2, 4, 4, device=device))

    def test_reload_folded_dump_is_stable(self, loaded_pair):
        model_new, _, _, _ = loaded_pair
        dump = {
            key: value.detach().clone() for key, value in model_new.state_dict().items()
        }
        model_new.load_state_dict(dump)
        for key, value in model_new.state_dict().items():
            assert torch.equal(value, dump[key]), key
