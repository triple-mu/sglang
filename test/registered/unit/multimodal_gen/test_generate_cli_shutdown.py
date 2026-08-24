from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sglang.multimodal_gen.runtime.entrypoints.cli import generate as generate_cli


class _FakeSamplingParams:
    @classmethod
    def get_cli_args(cls, _args):
        return {"prompt": "test prompt"}


def _args():
    return SimpleNamespace(
        config="",
        diffusers_kwargs=None,
        output_file_path=None,
        perf_dump_path=None,
    )


def _setup(monkeypatch, generator):
    server_args = SimpleNamespace(
        load_diffusion_decoder=False,
        model_path="/model",
    )
    monkeypatch.setattr(
        generate_cli.ServerArgs,
        "from_cli_args",
        Mock(return_value=server_args),
    )
    monkeypatch.setattr(
        generate_cli,
        "_resolve_cli_sampling_params_cls",
        Mock(return_value=_FakeSamplingParams),
    )
    monkeypatch.setattr(
        generate_cli.DiffGenerator,
        "from_pretrained",
        Mock(return_value=generator),
    )
    dump = Mock()
    monkeypatch.setattr(generate_cli, "maybe_dump_performance", dump)
    return dump


def test_generate_cmd_shuts_down_after_success(monkeypatch):
    generator = Mock()
    generator.generate.return_value = object()
    dump = _setup(monkeypatch, generator)

    generate_cli.generate_cmd(_args())

    generator.generate.assert_called_once()
    dump.assert_called_once()
    generator.shutdown.assert_called_once_with()


def test_generate_cmd_shuts_down_when_generate_raises(monkeypatch):
    generator = Mock()
    generator.generate.side_effect = RuntimeError("generate failed")
    dump = _setup(monkeypatch, generator)

    with pytest.raises(RuntimeError, match="generate failed"):
        generate_cli.generate_cmd(_args())

    dump.assert_not_called()
    generator.shutdown.assert_called_once_with()


def test_generate_cmd_shuts_down_when_perf_dump_raises(monkeypatch):
    generator = Mock()
    generator.generate.return_value = object()
    dump = _setup(monkeypatch, generator)
    dump.side_effect = RuntimeError("dump failed")

    with pytest.raises(RuntimeError, match="dump failed"):
        generate_cli.generate_cmd(_args())

    generator.shutdown.assert_called_once_with()


def test_generate_cmd_preserves_primary_error_when_shutdown_also_raises(monkeypatch):
    generator = Mock()
    generator.generate.side_effect = ValueError("primary failure")
    generator.shutdown.side_effect = RuntimeError("shutdown failure")
    dump = _setup(monkeypatch, generator)
    log_exception = Mock()
    monkeypatch.setattr(generate_cli.logger, "exception", log_exception)

    with pytest.raises(ValueError, match="primary failure"):
        generate_cli.generate_cmd(_args())

    dump.assert_not_called()
    generator.shutdown.assert_called_once_with()
    log_exception.assert_called_once_with(
        "Generator shutdown failed while handling an exception"
    )


def test_generate_cmd_propagates_shutdown_error_after_success(monkeypatch):
    generator = Mock()
    generator.generate.return_value = object()
    generator.shutdown.side_effect = RuntimeError("shutdown failure")
    dump = _setup(monkeypatch, generator)

    with pytest.raises(RuntimeError, match="shutdown failure"):
        generate_cli.generate_cmd(_args())

    dump.assert_called_once()
    generator.shutdown.assert_called_once_with()
