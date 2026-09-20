# SPDX-License-Identifier: Apache-2.0
"""Check encoder settings and playable output using the real FFmpeg backend.

libx264 records resolved options inside the MP4, so these tests detect options
that are accepted by a wrapper but never reach the encoder.
"""

import subprocess

import imageio_ffmpeg
import numpy as np
import pytest
import torch

from sglang.multimodal_gen.configs.sample.sampling_params import DataType
from sglang.multimodal_gen.runtime.entrypoints.utils import (
    X264_PRESET,
    _resolve_ffmpeg_exe,
    save_outputs,
)

FPS = 8
FRAMES = 8
SIZE = 64
PREFIX = "SGLANG_DIFFUSION_VIDEO_ENCODING_"
# Derive expectations from the same FFmpeg build instead of hard-coding the
# individual x264 options associated with each preset.
PRESET_DERIVED_KEYS = ("subme", "ref", "rc_lookahead", "me", "trellis")


@pytest.fixture(autouse=True)
def encoding_environment(monkeypatch):
    for suffix in ("PRESET", "CRF", "THREADS"):
        monkeypatch.delenv(PREFIX + suffix, raising=False)
    try:
        executable = _resolve_ffmpeg_exe()
    except RuntimeError:
        pytest.skip("needs FFmpeg with libx264")
    monkeypatch.setenv("IMAGEIO_FFMPEG_EXE", executable)
    return executable


def _x264_options(path) -> dict[str, str]:
    """The `options:` line libx264 embeds in its own output."""
    blob = path.read_bytes()
    start = blob.find(b"x264 - core")
    assert start >= 0, "libx264 did not stamp its settings into the file"
    text = blob[start : blob.find(b"\x00", start)].decode("utf-8", "replace")
    _, _, options = text.partition("options: ")
    return dict(kv.split("=", 1) for kv in options.split() if "=" in kv)


def _reference_encode(tmp_path, frames, preset, executable, *, crf=25, threads=None):
    out = tmp_path / f"ref_{preset}.mp4"
    command = [
        executable,
        "-v",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{SIZE}x{SIZE}",
        "-r",
        str(FPS),
        "-i",
        "pipe:0",
        "-vcodec",
        "libx264",
        "-preset",
        preset,
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(crf),
    ]
    if threads is not None:
        command += ["-threads", str(threads)]
    command.append(str(out))
    subprocess.run(command, input=frames.tobytes(), check=True)
    return out


def _decode_frames(path):
    reader = imageio_ffmpeg.read_frames(str(path))
    try:
        metadata = next(reader)
        frames = list(reader)
    finally:
        reader.close()
    assert metadata["codec"] == "h264"
    assert metadata["size"] == (SIZE, SIZE)
    assert metadata["fps"] == FPS
    assert len(frames) == FRAMES
    assert all(len(frame) == SIZE * SIZE * 3 for frame in frames)
    return frames


@pytest.mark.parametrize("with_audio", [False, True])
@pytest.mark.parametrize(
    ("preset", "crf", "threads"),
    [(None, None, None), ("medium", 19, 2), ("ultrafast", 0, 1), ("fast", 51, 1)],
)
def test_saved_video_carries_configured_options_and_decodes(
    tmp_path, monkeypatch, encoding_environment, with_audio, preset, crf, threads
):
    for suffix, value in (("PRESET", preset), ("CRF", crf), ("THREADS", threads)):
        if value is not None:
            monkeypatch.setenv(PREFIX + suffix, str(value))
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 256, (FRAMES, SIZE, SIZE, 3), dtype=np.uint8)
    video = torch.from_numpy(frames).permute(3, 0, 1, 2).float() / 255.0
    # One second of non-silent audio verifies that the single-pass AAC stream
    # remains usable when deployment encoder overrides are active.
    audio = (0.1 * np.sin(2 * np.pi * 440 * np.arange(24000) / 24000)).astype(
        np.float32
    )
    sample = (video, audio) if with_audio else video
    saved = tmp_path / "clip.mp4"
    paths = save_outputs(
        [sample],
        DataType.VIDEO,
        FPS,
        True,
        lambda _idx: str(saved),
        audio_sample_rate=24000,
    )
    assert paths == [str(saved)] and saved.exists()
    selected_preset = preset or X264_PRESET
    reference = _reference_encode(
        tmp_path,
        frames,
        selected_preset,
        encoding_environment,
        crf=25 if crf is None else crf,
        threads=threads,
    )
    got = _x264_options(saved)
    expected = _x264_options(reference)
    for key in (*PRESET_DERIVED_KEYS, "crf", "qp", "threads"):
        assert got.get(key) == expected.get(key), f"{key} differs from the reference"
    assert _decode_frames(saved) == _decode_frames(reference)
    if preset is None:
        default = _x264_options(
            _reference_encode(tmp_path, frames, "medium", encoding_environment)
        )
        assert any(got.get(key) != default.get(key) for key in PRESET_DERIVED_KEYS)
    if with_audio:
        decoded_audio = subprocess.run(
            [
                encoding_environment,
                "-v",
                "error",
                "-i",
                str(saved),
                "-map",
                "0:a:0",
                "-f",
                "f32le",
                "-acodec",
                "pcm_f32le",
                "pipe:1",
            ],
            capture_output=True,
            check=True,
        ).stdout
        samples = np.frombuffer(decoded_audio, dtype=np.float32)
        assert samples.size >= audio.size
        assert np.isfinite(samples).all()
        assert np.sqrt(np.mean(samples * samples)) > 0.01
