"""Tests verifying that amd_validated=True and EXECUTED status cannot be assigned falsely."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rocmhub.cli.main import main
from rocmhub.core.types import DetectionReport, EnvironmentSpec, ExecutionStatus, HardwareSpec
from rocmhub.forge.executor import ForgeExecutor


@pytest.fixture
def dummy_build_dir(tmp_path: Path) -> Path:
    bdir = tmp_path / "test_build"
    bdir.mkdir()
    launcher = bdir / "run_inference.py"
    launcher.write_text("#!/usr/bin/env python3\nprint('hello')\n")
    launcher.chmod(0o755)

    manifest = bdir / "build_manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": "1.0.0",
        "build_id": "bld_test",
        "plan_id": "plan_test",
        "model_id": "test/model",
        "revision": "a" * 40,
        "target_gpu": "gfx1100",
        "precision": "fp16",
        "recipe_id": "pytorch_transformers_hip",
        "recipe_version": "1.0.0",
        "runtime": "pytorch_transformers_hip",
        "status": "PREPARED",
        "build_dir": str(bdir),
        "weights_path": "/fake/weights",
        "steps": [],
        "created_at": "2026-09-23T00:00:00Z",
        "amd_validated": False,
        "secret_scan_clean": True,
        "artifacts": {},
    }))
    return bdir


@pytest.fixture
def amd_observer() -> MagicMock:
    obs = MagicMock()
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
    obs.observe.return_value = DetectionReport(
        schema_version="1.0.0",
        environment=mock_env,
        gpus=[mock_gpu],
    )
    return obs


class TestExecutionSafetyGuards:
    """Verifies strict separation of facts and prevents false amd_validated assertions."""

    def test_cpu_fallback_never_sets_amd_validated(
        self, dummy_build_dir: Path, amd_observer: MagicMock
    ) -> None:
        """When launcher exits 0 on CPU fallback, amd_validated must be False and status PREPARED."""
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = json.dumps({
            "status": "SUCCESS",
            "model_id": "test/model",
            "revision": "a" * 40,
            "device": "cpu",
            "device_fallback": True,
            "precision": "fp16",
            "prompt": "Hello",
            "generated_text": "Hello world from CPU.",
            "tokens_generated": 10,
            "tokens_per_second": 15.0,
            "is_hip": False,
            "hip_version": None,
            "pytorch_version": "2.4.0",
            "amd_gpu_used": False,
            "gpu_device_name": None,
        })
        mock_proc.stderr = ""

        executor = ForgeExecutor(observer=amd_observer)
        with patch("subprocess.run", return_value=mock_proc):
            res = executor.execute_build(dummy_build_dir, device="cuda")

            assert res.status == ExecutionStatus.SUCCESS
            assert res.process_success is True
            assert res.inference_executed is True
            assert res.hip_runtime_used is False
            assert res.amd_gpu_used is False
            assert res.amd_validated is False

            # Check that manifest was NOT upgraded to EXECUTED
            manifest_data = json.loads((dummy_build_dir / "build_manifest.json").read_text())
            assert manifest_data["status"] == "PREPARED"
            assert manifest_data["amd_validated"] is False

    def test_nvidia_cuda_execution_never_sets_amd_validated(
        self, dummy_build_dir: Path, amd_observer: MagicMock
    ) -> None:
        """When execution occurs on NVIDIA CUDA, amd_validated must be False."""
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = json.dumps({
            "status": "SUCCESS",
            "model_id": "test/model",
            "revision": "a" * 40,
            "device": "cuda:0",
            "device_fallback": False,
            "precision": "fp16",
            "prompt": "Hello",
            "generated_text": "Hello from NVIDIA CUDA.",
            "tokens_generated": 12,
            "tokens_per_second": 40.0,
            "is_hip": False,  # CUDA build, NOT HIP
            "hip_version": None,
            "pytorch_version": "2.4.0+cu121",
            "amd_gpu_used": False,
            "gpu_device_name": "NVIDIA GeForce RTX 4090",
        })
        mock_proc.stderr = ""

        executor = ForgeExecutor(observer=amd_observer)
        with patch("subprocess.run", return_value=mock_proc):
            res = executor.execute_build(dummy_build_dir, device="cuda")

            assert res.status == ExecutionStatus.SUCCESS
            assert res.hip_runtime_used is False
            assert res.amd_gpu_used is False
            assert res.amd_validated is False

            manifest_data = json.loads((dummy_build_dir / "build_manifest.json").read_text())
            assert manifest_data["status"] == "PREPARED"
            assert manifest_data["amd_validated"] is False

    def test_output_corruption_fails_validation_gate(
        self, dummy_build_dir: Path, amd_observer: MagicMock
    ) -> None:
        """Numeric corruption in output must fail validation check, not automatically PASS."""
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = json.dumps({
            "status": "SUCCESS",
            "model_id": "test/model",
            "revision": "a" * 40,
            "device": "cuda:0",
            "device_fallback": False,
            "precision": "fp16",
            "prompt": "Hello",
            "generated_text": "Output: NaN NaN inf -inf overflow",
            "tokens_generated": 15,
            "tokens_per_second": 50.0,
            "is_hip": True,
            "hip_version": "6.2.0",
            "pytorch_version": "2.4.0+rocm6.2",
            "amd_gpu_used": True,
            "gpu_device_name": "AMD Radeon RX 7900 XTX",
        })
        mock_proc.stderr = ""

        executor = ForgeExecutor(observer=amd_observer)
        with patch("subprocess.run", return_value=mock_proc):
            res = executor.execute_build(dummy_build_dir, device="cuda")

            assert res.status == ExecutionStatus.SUCCESS
            assert res.validation_passed is False
            assert len(res.validation_failures) > 0
            assert any("corruption" in f.lower() for f in res.validation_failures)

    def test_zero_tokens_generated_fails_inference_executed(
        self, dummy_build_dir: Path, amd_observer: MagicMock
    ) -> None:
        """Zero tokens generated means inference_executed is False."""
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = json.dumps({
            "status": "SUCCESS",
            "model_id": "test/model",
            "revision": "a" * 40,
            "device": "cuda:0",
            "device_fallback": False,
            "precision": "fp16",
            "prompt": "Hello",
            "generated_text": "",
            "tokens_generated": 0,
            "tokens_per_second": 0.0,
            "is_hip": True,
            "hip_version": "6.2.0",
            "pytorch_version": "2.4.0+rocm6.2",
            "amd_gpu_used": True,
            "gpu_device_name": "AMD Radeon RX 7900 XTX",
        })
        mock_proc.stderr = ""

        executor = ForgeExecutor(observer=amd_observer)
        with patch("subprocess.run", return_value=mock_proc):
            res = executor.execute_build(dummy_build_dir, device="cuda")

            assert res.inference_executed is False
            assert res.amd_validated is False

    def test_genuine_amd_hip_execution_upgrades_manifest(
        self, dummy_build_dir: Path, amd_observer: MagicMock
    ) -> None:
        """When all 5 facts are verified, amd_validated is True and status is EXECUTED."""
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = json.dumps({
            "status": "SUCCESS",
            "model_id": "test/model",
            "revision": "a" * 40,
            "device": "cuda:0",
            "device_fallback": False,
            "precision": "fp16",
            "prompt": "Hello",
            "generated_text": "Open-source AI acceleration enables scalable innovation.",
            "tokens_generated": 20,
            "tokens_per_second": 85.5,
            "is_hip": True,
            "hip_version": "6.2.0",
            "pytorch_version": "2.4.0+rocm6.2",
            "amd_gpu_used": True,
            "gpu_device_name": "AMD Radeon RX 7900 XTX",
        })
        mock_proc.stderr = ""

        executor = ForgeExecutor(observer=amd_observer)
        with patch("subprocess.run", return_value=mock_proc):
            res = executor.execute_build(dummy_build_dir, device="cuda")

            assert res.status == ExecutionStatus.SUCCESS
            assert res.process_success is True
            assert res.inference_executed is True
            assert res.hip_runtime_used is True
            assert res.amd_gpu_used is True
            assert res.validation_passed is True
            assert res.amd_validated is True

            # Verify build_manifest.json upgraded to EXECUTED
            manifest_data = json.loads((dummy_build_dir / "build_manifest.json").read_text())
            assert manifest_data["status"] == "EXECUTED"
            assert manifest_data["amd_validated"] is True
            assert manifest_data["hip_version"] == "6.2.0"
            assert manifest_data["torch_version"] == "2.4.0+rocm6.2"

    def test_dry_run_never_executes_subprocess(self, dummy_build_dir: Path) -> None:
        """Dry-run mode skips subprocess entirely and sets dry_run=True, amd_validated=False."""
        executor = ForgeExecutor()
        with patch("subprocess.run") as mock_subproc:
            res = executor.execute_build(dummy_build_dir, dry_run=True)

            mock_subproc.assert_not_called()
            assert res.status == ExecutionStatus.SKIPPED
            assert res.dry_run is True
            assert res.amd_validated is False
            assert res.inference_executed is False

    def test_cli_forge_execute_dry_run(self, dummy_build_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """CLI rocmhub forge execute <dir> --dry-run outputs clean dry-run report."""
        with patch("subprocess.run") as mock_subproc:
            code = main(["forge", "execute", str(dummy_build_dir), "--dry-run"])
            mock_subproc.assert_not_called()
            out, err = capsys.readouterr()

            assert code == 0
            assert "Dry Run:                   yes" in out
            assert "AMD Hardware Validated:    no" in out
            assert "Execution Status:          SKIPPED" in out
