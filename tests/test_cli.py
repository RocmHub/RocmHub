"""Unit tests for the rocmhub CLI commands."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from rocmhub.cli.main import main
from rocmhub.core.errors import ModelNotFoundError
from rocmhub.core.types import DetectionReport, EnvironmentSpec, HardwareSpec, ModelSpec


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


def test_cli_env_live(capsys: pytest.CaptureFixture[str]) -> None:
    """Test live rocmhub env on current host without mocking."""
    exit_code = main(["env"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "System:" in captured.out
    assert "Detected GPUs:" in captured.out


def test_cli_env_json_live(capsys: pytest.CaptureFixture[str]) -> None:
    """Test live rocmhub env --json produces parseable DetectionReport."""
    exit_code = main(["env", "--json"])
    assert exit_code == 0
    captured = capsys.readouterr()
    report = DetectionReport.model_validate_json(captured.out.strip())
    assert report.schema_version == "1.0.0"
    assert report.environment.os is not None


def test_cli_env_with_mocked_gpu(capsys: pytest.CaptureFixture[str]) -> None:
    """Test human-readable output formatting when AMD GPUs are detected."""
    mock_env = EnvironmentSpec(
        os="Linux 6.8.0-40-generic",
        kernel="6.8.0-40-generic",
        architecture="x86_64",
        python_version="3.11.9",
        rocm_version="6.2.0",
        hip_version="6.2.41133",
        torch_version="2.4.0+rocm6.2",
        torch_hip_available=True,
        env_vars={"HSA_OVERRIDE_GFX_VERSION": "11.0.0"},
    )
    mock_gpu = HardwareSpec(
        gpu_present=True,
        gpu_vendor="AMD",
        device_id=0,
        device_name="AMD Radeon RX 7900 XTX",
        family="Radeon",
        gfx_target="gfx1100",
        vram_total_mb=24576,
        compute_units=96,
    )
    mock_report = DetectionReport(
        environment=mock_env,
        gpus=[mock_gpu],
        provenance={"gpu[0]": "rocminfo"},
        warnings=[],
    )

    with patch("rocmhub.cli.main.SystemObserver.observe", return_value=mock_report):
        exit_code = main(["env"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "System:          Linux 6.8.0-40-generic" in captured.out
    assert "ROCm:            6.2.0" in captured.out
    assert "Active ROCm Environment Variables:" in captured.out
    assert "HSA_OVERRIDE_GFX_VERSION = 11.0.0" in captured.out
    assert "Detected GPUs: 1" in captured.out
    assert "GPU 0:" in captured.out
    assert "Vendor:          AMD" in captured.out
    assert "Device:          AMD Radeon RX 7900 XTX" in captured.out
    assert "gfx target:      gfx1100" in captured.out
    assert "VRAM:            24,576 MB" in captured.out
    assert "Compute Units:   96" in captured.out


def test_cli_run_missing_model_id(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["run"])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "model_id must be provided" in captured.err


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
