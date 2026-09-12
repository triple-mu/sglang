# SPDX-License-Identifier: Apache-2.0
"""Request/serve-argument derivation of the serving-mode nsys capture script."""

import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / ".claude/skills/sglang-diffusion-benchmark-profile/scripts/nsys_serving_capture.py"
)


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("nsys_serving_capture", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config(tmp_path: Path, conditions) -> dict:
    return {
        "model_path": "/models/MiniMax-H3",
        "model_variant": "ref2va",
        "num_gpus": 8,
        "tp_size": 2,
        "enable_torch_compile": False,
        "output_path": "/somewhere/else",
        "quality": "lossless",
        "prompt": "a prompt",
        "task": "ref2va",
        "conditions": conditions,
        "target": {"short_edge": 768, "aspect_ratio": "auto", "duration_seconds": 5.0},
        "num_inference_steps": 50,
        "seed": 3101,
        "save_output": True,
        "return_file_paths_only": True,
        "output_file_name": "ref2va.mp4",
    }


def test_relative_condition_media_becomes_file_uris(script, tmp_path):
    """fl2va and ref2va configs name their media relative to the config file;
    the server needs absolute URIs, and remote ones must pass untouched."""
    (tmp_path / "ref.mp4").write_bytes(b"x")
    config = _config(
        tmp_path,
        [
            {"type": "video", "uri": "ref.mp4", "role": "reference"},
            {"type": "audio", "uri": "https://example.com/a.mp3", "role": "reference"},
        ],
    )
    resolved = script.resolve_conditions(config, tmp_path)
    assert resolved["conditions"][0]["uri"] == (tmp_path / "ref.mp4").resolve().as_uri()
    assert resolved["conditions"][1]["uri"] == "https://example.com/a.mp3"
    with pytest.raises(FileNotFoundError):
        script.resolve_conditions(
            _config(tmp_path, [{"type": "image", "uri": "missing.png"}]), tmp_path
        )


def test_request_payload_keeps_request_fields_only(script, tmp_path):
    payload = script.request_payload(
        _config(tmp_path, []), output_dir=tmp_path, perf_dump=tmp_path / "perf.json"
    )
    assert payload["task"] == "ref2va" and payload["quality"] == "lossless"
    assert payload["seed"] == 3101 and payload["conditions"] == []
    for server_or_generate_field in (
        "model_path",
        "num_gpus",
        "save_output",
        "output_file_name",
    ):
        assert server_or_generate_field not in payload
    assert payload["output_path"] == str(tmp_path)
    assert payload["perf_dump_path"] == str(tmp_path / "perf.json")
    json.dumps(payload)


def test_serve_args_come_from_config_unless_passed_explicitly(script, tmp_path):
    """The config written for eight GPUs must serve on four when --num-gpus is
    given after `--`; booleans use the StoreBoolean true/false spelling."""
    derived = script.serve_args_from_config(
        _config(tmp_path, []), ["--num-gpus", "4", "--tp-size", "1"]
    )
    pairs = dict(zip(derived[::2], derived[1::2]))
    assert pairs["--model-variant"] == "ref2va"
    assert pairs["--model-path"] == "/models/MiniMax-H3"
    assert pairs["--enable-torch-compile"] == "false"
    assert "--num-gpus" not in pairs and "--tp-size" not in pairs
    assert "--output-path" not in pairs and "--quality" not in pairs
