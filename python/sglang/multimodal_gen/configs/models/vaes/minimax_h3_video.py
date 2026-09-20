# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.vaes.base import VAEArchConfig, VAEConfig
from sglang.multimodal_gen.configs.models.vaes.minimax_h3_contract import (
    validate_minimax_h3_vae_latent_stats,
)


@dataclass
class MiniMaxH3VideoVAEArchConfig(VAEArchConfig):
    latent_channels: int = 24
    latents_mean: list[float] | None = None
    latents_std: list[float] | None = None
    temporal_compression_ratio: int = 4
    spatial_compression_ratio: int = 16
    vae_clip_length: int = 17
    vae_token_drop: int = 3
    vae_encoder_tiling: int = 1
    vae_decoder_tiling: int = 1
    vae_parallel_tiling: int = 1
    vae_tile_size: int = 256
    vae_tile_overlap_min: int = 64
    vae_chunk_dim: int = -1


@dataclass
class MiniMaxH3VideoVAEConfig(VAEConfig):
    arch_config: MiniMaxH3VideoVAEArchConfig = field(
        default_factory=MiniMaxH3VideoVAEArchConfig
    )
    load_encoder: bool = True
    load_decoder: bool = True
    use_tiling: bool = True
    use_parallel_tiling: bool = True
    # The released checkpoint's quality contract uses overlapping latent
    # tiles. Parallel tiling distributes whole tiles without changing that
    # recipe. Spatial-shard decode is rejected because validation found output
    # mismatches on H3.
    parallel_decode_mode: str = "tiled"
    # Batch/fusion paths are additionally scoped by request quality. Online
    # FP8 is an independent deployment choice for the decoder block linears.
    enable_optimizations: bool = False
    encoder_tile_batch_size: int = 8
    decoder_tile_batch_size: int = 64
    decoder_window_batch_size: int = 0
    decoder_quantization: str | None = None
    decoder_output_projection_precision: str = "fp32"

    def validate_optimization_options(self) -> None:
        if type(self.enable_optimizations) is not bool:
            raise ValueError("enable_optimizations must be a boolean")
        for name in ("encoder_tile_batch_size", "decoder_tile_batch_size"):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= 64:
                raise ValueError(f"{name} must be an integer in [1, 64]")
        value = self.decoder_window_batch_size
        if type(value) is not int or not 0 <= value <= 64:
            raise ValueError(
                "decoder_window_batch_size must be in [0, 64]; 0 groups all equal-shaped windows"
            )
        if self.decoder_quantization not in (None, "fp8"):
            raise ValueError("MiniMax-H3 decoder_quantization must be None or 'fp8'")
        if self.decoder_output_projection_precision not in ("fp32", "fp16"):
            raise ValueError("decoder_output_projection_precision must be fp32 or fp16")

    def resolved_parallel_decode_mode(self) -> str:
        if self.parallel_decode_mode == "auto":
            return "tiled"
        if self.parallel_decode_mode in ("spatial", "spatial_shard"):
            raise ValueError(
                "MiniMax H3 rejects spatial-shard VAE decode because it failed "
                "the released quality contract; use tiled"
            )
        if self.parallel_decode_mode == "tiled":
            return "tiled"
        if self.parallel_decode_mode == "patch":
            raise ValueError("MiniMax H3 does not support patch VAE decode; use tiled")
        raise ValueError(
            f"unsupported MiniMax H3 VAE parallel decode mode "
            f"{self.parallel_decode_mode!r}"
        )

    def update_model_arch(self, source_model_dict: dict) -> None:
        # Native-Diffusers AutoencoderKLMiniMaxH3 config field names.
        aliases = {
            "clip_length": "vae_clip_length",
            "token_drop": "vae_token_drop",
        }
        model_dict = {
            aliases.get(key, key): value for key, value in source_model_dict.items()
        }
        super().update_model_arch(model_dict)

    def post_init(self) -> None:
        self.validate_optimization_options()
        self.resolved_parallel_decode_mode()
        validate_minimax_h3_vae_latent_stats(
            self.arch_config,
            component_name="video_vae",
            expected_channels=24,
        )


__all__ = ["MiniMaxH3VideoVAEArchConfig", "MiniMaxH3VideoVAEConfig"]
