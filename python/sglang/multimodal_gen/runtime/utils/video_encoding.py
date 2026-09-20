# SPDX-License-Identifier: Apache-2.0
"""Shared libx264 settings for direct CUDA saves and imageio fallbacks."""

from __future__ import annotations

from dataclasses import dataclass

X264_PRESETS = frozenset(
    {
        "ultrafast",
        "superfast",
        "veryfast",
        "faster",
        "fast",
        "medium",
        "slow",
        "slower",
        "veryslow",
        "placebo",
    }
)
DEFAULT_X264_PRESET = "fast"


@dataclass(frozen=True)
class X264EncodingOptions:
    """Optional deployment overrides, independent of model quality tiers.

    Explicit CRF overrides request output_compression; omitted CRF and
    threads retain the existing backend behavior.
    """

    preset: str = DEFAULT_X264_PRESET
    crf: int | None = None
    threads: int | None = None

    def __post_init__(self) -> None:
        if self.preset not in X264_PRESETS:
            raise ValueError(
                "SGLANG_DIFFUSION_VIDEO_ENCODING_PRESET must be one of "
                f"{sorted(X264_PRESETS)}, got {self.preset!r}"
            )
        if self.crf is not None and (
            type(self.crf) is not int or not 0 <= self.crf <= 51
        ):
            raise ValueError("SGLANG_DIFFUSION_VIDEO_ENCODING_CRF must be in [0, 51]")
        if self.threads is not None and (
            type(self.threads) is not int or not 1 <= self.threads <= 128
        ):
            raise ValueError(
                "SGLANG_DIFFUSION_VIDEO_ENCODING_THREADS must be in [1, 128]"
            )

    def output_params(self) -> list[str]:
        """Arguments appended by imageio after its standard encoder options."""
        params = ["-preset", self.preset]
        if self.crf is not None:
            params.extend(("-crf", str(self.crf)))
        if self.threads is not None:
            params.extend(("-threads", str(self.threads)))
        return params

    def imageio_quality(self, quality: float) -> float | None:
        # Avoid competing CRFs from imageio and an explicit override.
        return quality if self.crf is None else None


def get_x264_encoding_options() -> X264EncodingOptions:
    from sglang.multimodal_gen import envs

    return X264EncodingOptions(
        preset=envs.SGLANG_DIFFUSION_VIDEO_ENCODING_PRESET,
        crf=envs.SGLANG_DIFFUSION_VIDEO_ENCODING_CRF,
        threads=envs.SGLANG_DIFFUSION_VIDEO_ENCODING_THREADS,
    )
