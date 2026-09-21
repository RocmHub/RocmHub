"""Unit tests for the rocmhub CLI skeleton."""

import pytest

from rocmhub.cli.main import main


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


def test_cli_inspect_not_implemented(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["inspect", "Qwen/Qwen2.5-0.5B-Instruct"])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "NOT_IMPLEMENTED" in captured.err
    assert "Phase 4" in captured.err


def test_cli_run_not_implemented(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["run", "--model", "Qwen/Qwen2.5-0.5B-Instruct"])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "NOT_IMPLEMENTED" in captured.err
    assert "Phase 5-8" in captured.err
