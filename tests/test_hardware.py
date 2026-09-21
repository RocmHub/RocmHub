"""Unit tests for SystemHardwareDetector, EnvironmentDetector, and SystemObserver."""

from __future__ import annotations

import json
import os
from typing import Any, List, Optional
from unittest.mock import MagicMock, patch

from rocmhub.core.types import DetectionReport
from rocmhub.hardware.detector import SystemHardwareDetector, SystemObserver
from rocmhub.hardware.environment import EnvironmentDetector

ROCMINFO_RX7900XTX_SAMPLE = """
ROCk module is loaded
=====================
HSA System Attributes
=====================
Runtime Version:         1.14
System Timestamp:        123456789
*******
Agent 1
*******
  Name:                    AMD Ryzen 9 7950X 16-Core Processor
  Marketing Name:          AMD Ryzen 9 7950X 16-Core Processor
  Device Type:             CPU
*******
Agent 2
*******
  Name:                    gfx1100
  Marketing Name:          AMD Radeon RX 7900 XTX
  Vendor Name:             AMD
  Compute Unit:            96
  Pool Info:
    Pool 1
      Segment:                 GLOBAL; FAST
      Size:                    25149440(0x17fc000) KB
      Allocatable:             TRUE
  ISA Info:
    ISA 1
      Name:                    amdgcn-amd-amdhsa--gfx1100
"""

ROCMINFO_INSTINCT_MI300X_SAMPLE = """
*******
Agent 1
*******
  Name:                    AMD EPYC 9654
  Device Type:             CPU
*******
Agent 2
*******
  Name:                    gfx942
  Marketing Name:          AMD Instinct MI300X
  Vendor Name:             AMD
  Compute Unit:            304
  Pool Info:
    Pool 1
      Segment:                 GLOBAL; FAST
      Size:                    201326592(0xc000000) KB
      Allocatable:             TRUE
  ISA Info:
    ISA 1
      Name:                    amdgcn-amd-amdhsa--gfx942
"""

ROCMINFO_UNKNOWN_GFX_SAMPLE = """
*******
Agent 1
*******
  Name:                    gfx1205_custom
  Marketing Name:          AMD Experimental AI Accelerator
  Vendor Name:             AMD
  Compute Unit:            160
  Pool Info:
    Pool 1
      Segment:                 GLOBAL; FAST
      Size:                    33554432(0x2000000) KB
  ISA Info:
    ISA 1
      Name:                    amdgcn-amd-amdhsa--gfx1205_custom
  Device Type:             GPU
"""

ROCMINFO_MULTI_GPU_SAMPLE = """
*******
Agent 1
*******
  Name:                    AMD EPYC
  Device Type:             CPU
*******
Agent 2
*******
  Name:                    gfx942
  Marketing Name:          AMD Instinct MI300X
  Device Type:             GPU
  Compute Unit:            304
  Pool Info:
    Pool 1
      Segment:                 GLOBAL; FAST
      Size:                    201326592(0xc000000) KB
  ISA Info:
    ISA 1
      Name:                    amdgcn-amd-amdhsa--gfx942
*******
Agent 3
*******
  Name:                    gfx942
  Marketing Name:          AMD Instinct MI300X
  Device Type:             GPU
  Compute Unit:            304
  Pool Info:
    Pool 1
      Segment:                 GLOBAL; FAST
      Size:                    201326592(0xc000000) KB
  ISA Info:
    ISA 1
      Name:                    amdgcn-amd-amdhsa--gfx942
"""


class TestHardwareDetector:
    """Unit tests for GPU discovery across multiple sources and fallback tiers."""

    def test_macos_or_no_gpu_diagnostic_report(self) -> None:
        """On a machine with no AMD GPU tools, detector returns an empty list without crashing."""

        def mock_runner(cmd: List[str], timeout: float) -> Optional[str]:
            return None

        detector = SystemHardwareDetector(cmd_runner=mock_runner, sysfs_root="/nonexistent/sys")
        gpus, prov, warnings = detector.detect_gpus()

        assert gpus == []
        assert len(warnings) > 0

    def test_mocked_rx7900xtx_via_rocminfo_parser(self) -> None:
        """General parser extracts RX 7900 XTX attributes without hardcoding."""

        def mock_runner(cmd: List[str], timeout: float) -> Optional[str]:
            if "rocminfo" in cmd[0]:
                return ROCMINFO_RX7900XTX_SAMPLE
            return None

        detector = SystemHardwareDetector(cmd_runner=mock_runner)
        gpus, prov, warnings = detector.detect_gpus()

        assert len(gpus) == 1
        gpu = gpus[0]
        assert gpu.gpu_present is True
        assert gpu.gpu_vendor == "AMD"
        assert gpu.device_name == "AMD Radeon RX 7900 XTX"
        assert gpu.family == "Radeon"
        assert gpu.gfx_target == "gfx1100"
        assert gpu.compute_units == 96
        assert gpu.vram_total_mb == 24560
        assert "rocminfo" in prov["gpu[0]"]

    def test_mocked_instinct_mi300x_via_rocminfo(self) -> None:
        """General parser correctly extracts Instinct MI300X specifications."""

        def mock_runner(cmd: List[str], timeout: float) -> Optional[str]:
            if "rocminfo" in cmd[0]:
                return ROCMINFO_INSTINCT_MI300X_SAMPLE
            return None

        detector = SystemHardwareDetector(cmd_runner=mock_runner)
        gpus, prov, warnings = detector.detect_gpus()

        assert len(gpus) == 1
        gpu = gpus[0]
        assert gpu.gpu_vendor == "AMD"
        assert gpu.device_name == "AMD Instinct MI300X"
        assert gpu.family == "Instinct"
        assert gpu.gfx_target == "gfx942"
        assert gpu.compute_units == 304
        assert gpu.vram_total_mb == 196608

    def test_unknown_gfx_target_parsed_dynamically(self) -> None:
        """Unlisted/future gfx target is parsed as an open string without error."""

        def mock_runner(cmd: List[str], timeout: float) -> Optional[str]:
            if "rocminfo" in cmd[0]:
                return ROCMINFO_UNKNOWN_GFX_SAMPLE
            return None

        detector = SystemHardwareDetector(cmd_runner=mock_runner)
        gpus, prov, warnings = detector.detect_gpus()

        assert len(gpus) == 1
        gpu = gpus[0]
        assert gpu.gfx_target == "gfx1205_custom"
        assert gpu.compute_units == 160
        assert gpu.vram_total_mb == 32768
        assert gpu.family is None  # Not guessed

    def test_multi_gpu_discovery(self) -> None:
        """Detector discovers multiple AMD GPUs and assigns unique device IDs."""

        def mock_runner(cmd: List[str], timeout: float) -> Optional[str]:
            if "rocminfo" in cmd[0]:
                return ROCMINFO_MULTI_GPU_SAMPLE
            return None

        detector = SystemHardwareDetector(cmd_runner=mock_runner)
        gpus, prov, warnings = detector.detect_gpus()

        assert len(gpus) == 2
        assert gpus[0].device_id == 0
        assert gpus[1].device_id == 1
        assert gpus[0].device_name == "AMD Instinct MI300X"
        assert gpus[1].device_name == "AMD Instinct MI300X"

    def test_amd_smi_tier_fallback_when_rocminfo_absent(self) -> None:
        """When rocminfo is absent, detector falls back to amd-smi."""
        amd_smi_json = json.dumps(
            [
                {
                    "gpu": 0,
                    "asic": {
                        "market_name": "AMD Radeon RX 7900 XTX",
                        "target_graphics_version": "gfx1100",
                    },
                    "vram": {"vram_size": 24576},
                }
            ]
        )

        def mock_runner(cmd: List[str], timeout: float) -> Optional[str]:
            if "rocminfo" in cmd[0]:
                return None
            if "amd-smi" in cmd[0]:
                return amd_smi_json
            return None

        detector = SystemHardwareDetector(cmd_runner=mock_runner)
        gpus, prov, warnings = detector.detect_gpus()

        assert len(gpus) == 1
        gpu = gpus[0]
        assert gpu.device_name == "AMD Radeon RX 7900 XTX"
        assert gpu.gfx_target == "gfx1100"
        assert gpu.vram_total_mb == 24576
        assert "amd-smi" in prov["gpu[0]"]

    def test_rocm_smi_tier_fallback_when_amd_smi_absent(self) -> None:
        """When rocminfo and amd-smi are absent, falls back to rocm-smi."""
        rocm_smi_json = json.dumps(
            {
                "card0": {
                    "Card Series": "AMD Radeon RX 7900 XTX",
                    "VRAM Total Memory (B)": "25752535040",
                }
            }
        )

        def mock_runner(cmd: List[str], timeout: float) -> Optional[str]:
            if "rocminfo" in cmd[0] or "amd-smi" in cmd[0]:
                return None
            if "rocm-smi" in cmd[0]:
                return rocm_smi_json
            return None

        detector = SystemHardwareDetector(cmd_runner=mock_runner)
        gpus, prov, warnings = detector.detect_gpus()

        assert len(gpus) == 1
        assert gpus[0].device_name == "AMD Radeon RX 7900 XTX"
        assert gpus[0].vram_total_mb == 24560
        assert "rocm-smi" in prov["gpu[0]"]

    def test_malformed_tool_output_triggers_warning_and_fallback(self) -> None:
        """Malformed output from a tool logs a warning and tries the next tier."""
        rocm_smi_json = json.dumps(
            {
                "card0": {
                    "Card Series": "AMD Radeon RX 6800",
                    "VRAM Total Memory (B)": "17179869184",
                }
            }
        )

        def mock_runner(cmd: List[str], timeout: float) -> Optional[str]:
            if "rocminfo" in cmd[0]:
                return "CORRUPTED RANDOM TEXT NON-GPU"
            if "amd-smi" in cmd[0]:
                return "{NOT_VALID_JSON"
            if "rocm-smi" in cmd[0]:
                return rocm_smi_json
            return None

        detector = SystemHardwareDetector(cmd_runner=mock_runner)
        gpus, prov, warnings = detector.detect_gpus()

        assert len(gpus) == 1
        assert gpus[0].device_name == "AMD Radeon RX 6800"
        assert any("Failed to parse amd-smi JSON" in w for w in warnings)

    def test_sysfs_kfd_fallback(self, tmp_path: Any) -> None:
        """When CLI tools are absent, sysfs KFD topology is parsed."""
        # Setup mock /sys/class/kfd/kfd/topology/nodes/node1/properties
        kfd_dir = tmp_path / "kfd" / "kfd" / "topology" / "nodes" / "node1"
        kfd_dir.mkdir(parents=True)
        props_file = kfd_dir / "properties"
        props_file.write_text(
            "name gfx1100\nsimd_count 384\nvram_size_kb 25149440\n",
            encoding="utf-8",
        )

        def mock_runner(cmd: List[str], timeout: float) -> Optional[str]:
            return None

        detector = SystemHardwareDetector(cmd_runner=mock_runner, sysfs_root=str(tmp_path))
        gpus, prov, warnings = detector.detect_gpus()

        assert len(gpus) == 1
        assert gpus[0].gfx_target == "gfx1100"
        assert gpus[0].compute_units == 96
        assert gpus[0].vram_total_mb == 24560


class TestEnvironmentDetector:
    """Unit tests for EnvironmentDetector, PyTorch detection, and env var whitelist."""

    def test_environment_without_rocm(self, tmp_path: Any) -> None:
        detector = EnvironmentDetector(rocm_path=str(tmp_path / "opt_rocm"))
        spec, prov, warnings = detector.detect()

        assert spec.rocm_version is None
        assert spec.hip_version is None
        assert spec.torch_hip_available is False
        assert spec.python_version is not None
        assert spec.os is not None

    def test_rocm_version_from_filesystem(self, tmp_path: Any) -> None:
        rocm_dir = tmp_path / "rocm"
        info_dir = rocm_dir / ".info"
        info_dir.mkdir(parents=True)
        (info_dir / "version").write_text("6.2.0-12345\n")

        detector = EnvironmentDetector(rocm_path=str(rocm_dir))
        spec, prov, warnings = detector.detect()

        assert spec.rocm_version == "6.2.0"
        assert "rocm_version" in prov

    def test_pytorch_rocm_build_mock(self) -> None:
        mock_torch = MagicMock()
        mock_torch.__version__ = "2.4.0+rocm6.2"
        mock_torch.version.hip = "6.2.41133"
        mock_torch.cuda.is_available.return_value = True

        with patch.dict("sys.modules", {"torch": mock_torch}):
            detector = EnvironmentDetector()
            spec, prov, warnings = detector.detect()

            assert spec.torch_version == "2.4.0+rocm6.2"
            assert spec.torch_hip_available is True
            assert spec.hip_version == "6.2.41133"

    def test_pytorch_cpu_only_mock(self) -> None:
        mock_torch = MagicMock()
        mock_torch.__version__ = "2.4.0"
        mock_torch.version.hip = None
        mock_torch.cuda.is_available.return_value = False

        with patch.dict("sys.modules", {"torch": mock_torch}):
            detector = EnvironmentDetector()
            spec, prov, warnings = detector.detect()

            assert spec.torch_version == "2.4.0"
            assert spec.torch_hip_available is False
            assert spec.hip_version is None

    def test_environment_variable_whitelist_and_secret_isolation(self) -> None:
        """Ensure whitelisted ROCm env vars are retained and all secrets are filtered out."""
        test_env = {
            "HSA_OVERRIDE_GFX_VERSION": "11.0.0",
            "ROCR_VISIBLE_DEVICES": "0,1",
            "HF_TOKEN": "hf_super_secret_token_12345",
            "GITHUB_TOKEN": "ghp_secret_access_key",
            "AWS_SECRET_ACCESS_KEY": "classified",
            "DATABASE_URL": "postgres://user:pass@localhost:5432/db",
            "PATH": "/usr/bin:/bin",
        }

        with patch.dict(os.environ, test_env, clear=True):
            detector = EnvironmentDetector()
            spec, prov, warnings = detector.detect()

            # Whitelisted variables present
            assert spec.env_vars.get("HSA_OVERRIDE_GFX_VERSION") == "11.0.0"
            assert spec.env_vars.get("ROCR_VISIBLE_DEVICES") == "0,1"

            # Secret variables strictly excluded
            assert "HF_TOKEN" not in spec.env_vars
            assert "GITHUB_TOKEN" not in spec.env_vars
            assert "AWS_SECRET_ACCESS_KEY" not in spec.env_vars
            assert "DATABASE_URL" not in spec.env_vars
            assert "PATH" not in spec.env_vars


class TestSystemObserver:
    """Tests for unified SystemObserver."""

    def test_system_observer_assembles_report(self) -> None:
        observer = SystemObserver()
        report: DetectionReport = observer.observe()

        assert report.schema_version == "1.0.0"
        assert report.environment is not None
        assert isinstance(report.gpus, list)
        assert isinstance(report.provenance, dict)
        assert isinstance(report.warnings, list)
