"""Unit tests for the rocmhub CLI commands."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from rocmhub.cli.main import main
from rocmhub.core.errors import ModelNotFoundError
from rocmhub.core.types import ModelSpec


def test_cli_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "rocmhub 0.1.0" in (captured.out + captured.err)


def test_cli_help(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main([])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "usage: rocmhub" in captured.err


def test_cli_env_not_implemented(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["env"])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "NOT_IMPLEMENTED" in captured.err
    assert "Phase 3" in captured.err


def test_cli_run_not_implemented(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["run", "--model", "Qwen/Qwen2.5-0.5B-Instruct"])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "NOT_IMPLEMENTED" in captured.err
    assert "Phase 5-8" in captured.err


def test_cli_inspect_human_readable(capsys: pytest.CaptureFixture[str]) -> None:
    mock_spec = ModelSpec(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        requested_revision="main",
        commit_sha="7ae557604adf67be50417f59c2c2f167def9a775",
        architecture="Qwen2ForCausalLM",
        parameter_count=494_032_768,
        context_length=32768,
        default_dtype="bfloat16",
        weights_format="safetensors",
    )
    with patch("rocmhub.cli.main.ModelInspector.inspect", return_value=mock_spec):
        exit_code = main(["inspect", "Qwen/Qwen2.5-0.5B-Instruct"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Model:                 Qwen/Qwen2.5-0.5B-Instruct" in captured.out
    assert "Resolved commit SHA:   7ae557604adf67be50417f59c2c2f167def9a775" in captured.out
    assert "Architecture:          Qwen2ForCausalLM" in captured.out
    assert "494,032,768" in captured.out
    assert "safetensors" in captured.out


def test_cli_inspect_json(capsys: pytest.CaptureFixture[str]) -> None:
    mock_spec = ModelSpec(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        requested_revision="main",
        commit_sha="7ae557604adf67be50417f59c2c2f167def9a775",
        architecture="Qwen2ForCausalLM",
        parameter_count=494_032_768,
        context_length=32768,
        default_dtype="bfloat16",
        weights_format="safetensors",
    )
    with patch("rocmhub.cli.main.ModelInspector.inspect", return_value=mock_spec):
        exit_code = main(["inspect", "Qwen/Qwen2.5-0.5B-Instruct", "--json"])

    assert exit_code == 0
    captured = capsys.readouterr()
    # Output must be pure parseable JSON matching ModelSpec
    loaded_spec = ModelSpec.model_validate_json(captured.out.strip())
    assert loaded_spec.model_id == "Qwen/Qwen2.5-0.5B-Instruct"
    assert loaded_spec.commit_sha == "7ae557604adf67be50417f59c2c2f167def9a775"
    assert loaded_spec.parameter_count == 494_032_768


def test_cli_inspect_error_handling(capsys: pytest.CaptureFixture[str]) -> None:
    with patch(
        "rocmhub.cli.main.ModelInspector.inspect",
        side_effect=ModelNotFoundError("Repository 'bad/model' does not exist."),
    ):
        exit_code = main(["inspect", "bad/model"])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Error: [MODEL_NOT_FOUND]" in captured.err
