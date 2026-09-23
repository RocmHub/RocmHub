"""Tests for ROCmDoctor diagnostic engine and CLI integration."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from rocmhub.cli.main import main
from rocmhub.core.types import DetectionReport, EnvironmentSpec, HardwareSpec
from rocmhub.doctor import DoctorReport, DoctorVerdict, ROCmDoctor


class TestROCmDoctor:
    """Unit tests for ROCmDoctor logic and check rules."""

    def test_doctor_on_current_host(self) -> None:
        doctor = ROCmDoctor()
        report = doctor.run_diagnostics()

        assert isinstance(report, DoctorReport)
        assert report.verdict in (
            DoctorVerdict.READY,
            DoctorVerdict.WARNING,
            DoctorVerdict.NO_ACCELERATOR,
        )
        assert len(report.checks) >= 4
        check_names = {c.name for c in report.checks}
        assert "os_platform" in check_names
        assert "amd_hardware" in check_names
        assert "kernel_kfd" in check_names
        assert "rocm_stack" in check_names
        assert "pytorch_hip" in check_names

    def test_doctor_ready_on_linux_amd_gpu(self) -> None:
        mock_env = EnvironmentSpec(
            os="Linux 6.8.0-generic",
            kernel="6.8.0-generic",
            architecture="x86_64",
            python_version="3.11.0",
            rocm_version="6.2.0",
            hip_version="6.2.0",
            torch_version="2.4.0+rocm6.2",
            torch_hip_available=True,
            env_vars={"ROCM_PATH": "/opt/rocm"},
        )
        mock_gpu = HardwareSpec(
            device_id=0,
            gpu_vendor="AMD",
            device_name="Radeon RX 7900 XTX",
            gfx_target="gfx1100",
            vram_total_mb=24576,
        )
        mock_detection = DetectionReport(
            schema_version="1.0.0",
            environment=mock_env,
            gpus=[mock_gpu],
        )

        mock_observer = MagicMock()
        mock_observer.observe.return_value = mock_detection

        with patch("platform.system", return_value="Linux"), \
             patch("platform.release", return_value="6.8.0-generic"), \
             patch("platform.machine", return_value="x86_64"), \
             patch("pathlib.Path.exists", return_value=True), \
             patch("os.access", return_value=True), \
             patch("shutil.which", return_value="/opt/rocm/bin/rocm-smi"), \
             patch.dict("sys.modules", {"torch": MagicMock(
                 __version__="2.4.0+rocm6.2",
                 version=MagicMock(hip="6.2.0"),
                 cuda=MagicMock(is_available=lambda: True, device_count=lambda: 1)
             )}):
            doctor = ROCmDoctor(observer=mock_observer)
            report = doctor.run_diagnostics()

            assert report.verdict == DoctorVerdict.READY
            assert "ready" in report.summary.lower()

    def test_doctor_no_accelerator_on_mac(self) -> None:
        mock_env = EnvironmentSpec(
            os="Darwin 24.0.0",
            kernel="24.0.0",
            architecture="arm64",
            python_version="3.9.6",
            rocm_version=None,
            hip_version=None,
            torch_version="not_installed",
            torch_hip_available=False,
            env_vars={},
        )
        mock_detection = DetectionReport(
            schema_version="1.0.0",
            environment=mock_env,
            gpus=[],
        )
        mock_observer = MagicMock()
        mock_observer.observe.return_value = mock_detection

        with patch("platform.system", return_value="Darwin"):
            doctor = ROCmDoctor(observer=mock_observer)
            report = doctor.run_diagnostics()

            assert report.verdict == DoctorVerdict.NO_ACCELERATOR
            assert "diagnostic mode" in report.summary.lower()

    def test_cli_doctor_command(self, capsys: pytest.CaptureFixture[str]) -> None:
        code = main(["doctor"])
        out, err = capsys.readouterr()

        assert "ROCmHub Doctor Diagnostics" in out
        assert "Verdict:" in out
        assert "Diagnostic Checks:" in out
        # On Mac without AMD GPU, exit code is 2 (NO_ACCELERATOR)
        assert code in (0, 2)

    def test_cli_doctor_json_command(self, capsys: pytest.CaptureFixture[str]) -> None:
        code = main(["doctor", "--json"])
        out, err = capsys.readouterr()

        parsed = json.loads(out)
        assert "verdict" in parsed
        assert "summary" in parsed
        assert "checks" in parsed
        assert "environment" in parsed
        assert code in (0, 2)
