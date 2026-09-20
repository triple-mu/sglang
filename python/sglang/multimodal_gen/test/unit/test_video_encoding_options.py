# SPDX-License-Identifier: Apache-2.0
"""Deployment encoder controls must reach every save path consistently."""

from unittest.mock import Mock

import numpy as np
import pytest
import torch

import sglang.multimodal_gen.runtime.entrypoints.utils as output_utils
from sglang.multimodal_gen.configs.sample.sampling_params import DataType
from sglang.multimodal_gen.runtime.utils.video_encoding import (
    X264EncodingOptions,
    get_x264_encoding_options,
)

PREFIX = "SGLANG_DIFFUSION_VIDEO_ENCODING_"


@pytest.fixture(autouse=True)
def reset_encoding_environment(monkeypatch):
    for suffix in ("PRESET", "CRF", "THREADS"):
        monkeypatch.delenv(PREFIX + suffix, raising=False)


def _configure(monkeypatch, *, preset="veryfast", crf=19, threads=2):
    for suffix, value in (("PRESET", preset), ("CRF", crf), ("THREADS", threads)):
        monkeypatch.setenv(PREFIX + suffix, str(value))


def test_default_settings_preserve_existing_encoder_behavior():
    options = get_x264_encoding_options()
    assert options == X264EncodingOptions(preset="fast", crf=None, threads=None)
    assert options.output_params() == ["-preset", "fast"]
    assert options.imageio_quality(5) == 5


@pytest.mark.parametrize("crf", [0, 19, 51])
@pytest.mark.parametrize("threads", [1, 128])
def test_explicit_crf_replaces_imageio_quality(monkeypatch, crf, threads):
    _configure(monkeypatch, crf=crf, threads=threads)
    options = get_x264_encoding_options()
    assert options.output_params() == [
        "-preset",
        "veryfast",
        "-crf",
        str(crf),
        "-threads",
        str(threads),
    ]
    assert options.imageio_quality(5) is None


@pytest.mark.parametrize(
    ("suffix", "value"),
    [
        ("PRESET", "invalid"),
        ("PRESET", "FAST"),
        ("PRESET", ""),
        ("CRF", "-1"),
        ("CRF", "52"),
        ("CRF", "1.5"),
        ("CRF", ""),
        ("THREADS", "0"),
        ("THREADS", "129"),
        ("THREADS", "1.5"),
    ],
)
def test_invalid_settings_fail_before_encoding(monkeypatch, suffix, value, tmp_path):
    monkeypatch.setenv(PREFIX + suffix, value)
    encode = Mock()
    monkeypatch.setattr(output_utils.imageio, "mimsave", encode)
    materialized = output_utils.MaterializedOutput(
        sample=None, frames=[np.zeros((16, 16, 3), np.uint8)], audio=None, fps=8
    )
    with pytest.raises(ValueError):
        output_utils.save_materialized_output(
            materialized, DataType.VIDEO, str(tmp_path / "clip.mp4")
        )
    encode.assert_not_called()


@pytest.mark.parametrize("with_audio", [False, True])
@pytest.mark.parametrize("fail_single_pass", [False, True])
def test_imageio_and_audio_fallback_keep_identical_settings(
    tmp_path, monkeypatch, with_audio, fail_single_pass
):
    _configure(monkeypatch)
    calls = []

    def encode(_path, _frames, **kwargs):
        calls.append(kwargs)
        if fail_single_pass and "audio_path" in kwargs:
            raise RuntimeError("single-pass audio unsupported")

    monkeypatch.setattr(output_utils.imageio, "mimsave", encode)
    monkeypatch.setattr(output_utils.scipy_wavfile, "write", Mock())
    mux = Mock()
    monkeypatch.setattr(output_utils, "_maybe_mux_audio_into_mp4", mux)
    materialized = output_utils.MaterializedOutput(
        sample=None,
        frames=[np.zeros((16, 16, 3), np.uint8)],
        audio=np.zeros(8000, np.float32) if with_audio else None,
        fps=8,
    )
    output_utils.save_materialized_output(
        materialized,
        DataType.VIDEO,
        str(tmp_path / "clip.mp4"),
        output_compression=90,
        audio_sample_rate=8000,
    )
    assert len(calls) == (2 if with_audio and fail_single_pass else 1)
    for kwargs in calls:
        assert kwargs["quality"] is None
        assert kwargs["output_params"] == [
            "-preset",
            "veryfast",
            "-crf",
            "19",
            "-threads",
            "2",
        ]
    assert mux.call_count == (0 if with_audio and not fail_single_pass else 1)


class _CudaMetadataTensor(torch.Tensor):
    """CPU storage with CUDA metadata; stop before any device operation."""

    @property
    def device(self):
        return torch.device("cuda:0")


def _video(height=64):
    return torch.zeros((3, 8, height, 64)).as_subclass(_CudaMetadataTensor)


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("output_compression", [None, 90])
def test_direct_cuda_command_uses_same_settings(
    tmp_path, monkeypatch, explicit, output_compression
):
    if explicit:
        _configure(monkeypatch)
    monkeypatch.setattr(output_utils, "_resolve_ffmpeg_exe", lambda: "ffmpeg")
    monkeypatch.setattr(output_utils, "_x264_auto_thread_count", lambda _h: 7)
    launch = Mock(side_effect=RuntimeError("stop before CUDA buffer allocation"))
    monkeypatch.setattr(output_utils.subprocess, "Popen", launch)
    assert not output_utils._try_save_cuda_video_direct(
        save_file_path=str(tmp_path / "clip.mp4"),
        sample=_video(),
        fps=8,
        audio_sample_rate=None,
        output_compression=output_compression,
    )
    command = launch.call_args.args[0]
    assert command[command.index("-preset") + 1] == ("veryfast" if explicit else "fast")
    expected_crf = 19 if explicit else (25 if output_compression is None else 5)
    assert command[command.index("-crf") + 1] == str(expected_crf)
    assert command[command.index("-threads") + 1] == ("2" if explicit else "7")


def test_explicit_crf_does_not_use_output_compression_validation(tmp_path, monkeypatch):
    _configure(monkeypatch, crf=0)
    monkeypatch.setattr(output_utils, "_resolve_ffmpeg_exe", lambda: "ffmpeg")
    launch = Mock(side_effect=RuntimeError("stop before CUDA buffer allocation"))
    monkeypatch.setattr(output_utils.subprocess, "Popen", launch)
    output_utils._try_save_cuda_video_direct(
        save_file_path=str(tmp_path / "clip.mp4"),
        sample=_video(),
        fps=8,
        audio_sample_rate=None,
        output_compression=0,
    )
    command = launch.call_args.args[0]
    assert command[command.index("-crf") + 1] == "0"


@pytest.mark.parametrize(("available_cpus", "expected"), [(3, None), (4, [True, True])])
def test_parallel_budget_accounts_for_explicit_threads(
    monkeypatch, available_cpus, expected
):
    _configure(monkeypatch, threads=2)
    monkeypatch.setattr(
        output_utils.torch.cuda, "mem_get_info", lambda _d: (2**30, 2**30)
    )
    monkeypatch.setattr(
        output_utils.os, "sched_getaffinity", lambda _p: set(range(available_cpus))
    )
    save = Mock(return_value=True)
    monkeypatch.setattr(output_utils, "_try_save_cuda_video_direct", save)
    result = output_utils._try_save_cuda_videos_direct(
        [_video(), _video()],
        ["first.mp4", "second.mp4"],
        fps=8,
        audio_sample_rate=None,
        output_compression=None,
    )
    assert result == expected
    assert save.call_count == (0 if expected is None else 2)
