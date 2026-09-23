"""ROCmHub Doctor: Environment, Hardware, and ROCm Stack Diagnostics."""

from __future__ import annotations

import os
import platform
import shutil
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from rocmhub.hardware.detector import SystemObserver


class CheckStatus(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    INFO = "INFO"
    SKIPPED = "SKIPPED"


class DoctorVerdict(str, Enum):
    READY = "READY"
    WARNING = "WARNING"
    NO_ACCELERATOR = "NO_ACCELERATOR"


class DoctorCheck(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(..., description="Short name of the check.")
    status: CheckStatus = Field(..., description="Check outcome status.")
    message: str = Field(..., description="Human-readable explanation of check outcome.")
    details: Dict[str, Any] = Field(default_factory=dict, description="Structured diagnostics.")


class DoctorReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    verdict: DoctorVerdict = Field(..., description="Overall diagnostic verdict.")
    summary: str = Field(..., description="High-level diagnostic summary.")
    checks: List[DoctorCheck] = Field(default_factory=list, description="Detailed list of check results.")
    environment: Dict[str, Any] = Field(default_factory=dict, description="Host environment metadata.")
    gpus: List[Dict[str, Any]] = Field(default_factory=list, description="Detected GPU devices.")


class ROCmDoctor:
    """Diagnoses host environment, AMD hardware, kernel drivers, and PyTorch/HIP stack."""

    def __init__(self, observer: Optional[SystemObserver] = None) -> None:
        self._observer = observer or SystemObserver()

    def run_diagnostics(self) -> DoctorReport:
        checks: List[DoctorCheck] = []
        detection_report = self._observer.observe()
        env_spec = detection_report.environment
        hw_report = detection_report

        # 1. OS & Platform Check
        system = platform.system()
        kernel = platform.release()
        arch = platform.machine()
        if system == "Linux":
            checks.append(
                DoctorCheck(
                    name="os_platform",
                    status=CheckStatus.PASS,
                    message=f"Linux host detected ({kernel}, {arch}). Native ROCm supported.",
                    details={"system": system, "kernel": kernel, "arch": arch},
                )
            )
        else:
            checks.append(
                DoctorCheck(
                    name="os_platform",
                    status=CheckStatus.INFO,
                    message=(
                        f"{system} host detected ({kernel}, {arch}). "
                        "ROCmHub diagnostic & validation mode active. Native ROCm execution requires Linux."
                    ),
                    details={"system": system, "kernel": kernel, "arch": arch},
                )
            )

        # 2. AMD Hardware Detection
        amd_gpus = [g for g in hw_report.gpus if g.gpu_vendor and g.gpu_vendor.lower() == "amd"]
        if amd_gpus:
            gpu_names = ", ".join(f"{g.device_name} ({g.gfx_target})" for g in amd_gpus)
            checks.append(
                DoctorCheck(
                    name="amd_hardware",
                    status=CheckStatus.PASS,
                    message=f"Detected {len(amd_gpus)} AMD GPU(s): {gpu_names}.",
                    details={
                        "count": len(amd_gpus),
                        "devices": [g.model_dump() for g in amd_gpus],
                    },
                )
            )
        else:
            checks.append(
                DoctorCheck(
                    name="amd_hardware",
                    status=CheckStatus.WARN,
                    message="No AMD GPU detected on host. Real GPU acceleration unavailable.",
                    details={"total_gpus_detected": len(hw_report.gpus)},
                )
            )

        # 3. Kernel Drivers & Device Permissions (Linux only)
        if system == "Linux":
            kfd_path = Path("/dev/kfd")
            if kfd_path.exists():
                is_rw = os.access(kfd_path, os.R_OK | os.W_OK)
                checks.append(
                    DoctorCheck(
                        name="kernel_kfd",
                        status=CheckStatus.PASS if is_rw else CheckStatus.WARN,
                        message=(
                            "Kernel Fusion Driver (/dev/kfd) is accessible (read/write)."
                            if is_rw
                            else "Kernel Fusion Driver (/dev/kfd) exists but lacks read/write permissions."
                        ),
                        details={"path": str(kfd_path), "readable_writable": is_rw},
                    )
                )
            else:
                checks.append(
                    DoctorCheck(
                        name="kernel_kfd",
                        status=CheckStatus.WARN,
                        message="Kernel Fusion Driver (/dev/kfd) not found. ROCm kernel driver (amdgpu) may not be loaded.",
                        details={"path": str(kfd_path), "exists": False},
                    )
                )

            # Check video / render group membership
            try:
                import grp

                user_groups = [grp.getgrgid(g).gr_name for g in os.getgroups()]
                has_render_or_video = any(grp_name in user_groups for grp_name in ("render", "video"))
                checks.append(
                    DoctorCheck(
                        name="user_groups",
                        status=CheckStatus.PASS if has_render_or_video else CheckStatus.WARN,
                        message=(
                            f"Current user is member of GPU access groups: {user_groups}."
                            if has_render_or_video
                            else f"User is not in 'render' or 'video' groups (current: {user_groups}). May encounter permission errors."
                        ),
                        details={"groups": user_groups, "has_render_or_video": has_render_or_video},
                    )
                )
            except Exception:
                checks.append(
                    DoctorCheck(
                        name="user_groups",
                        status=CheckStatus.INFO,
                        message="Could not inspect user group memberships.",
                    )
                )
        else:
            checks.append(
                DoctorCheck(
                    name="kernel_kfd",
                    status=CheckStatus.SKIPPED,
                    message="KFD driver check skipped on non-Linux host.",
                )
            )

        # 4. ROCm Stack & Binaries
        rocm_path = env_spec.env_vars.get("ROCM_PATH") or os.environ.get("ROCM_PATH", "/opt/rocm")
        rocm_dir = Path(rocm_path)
        rocm_smi_bin = shutil.which("rocm-smi") or (str(rocm_dir / "bin" / "rocm-smi") if (rocm_dir / "bin" / "rocm-smi").exists() else None)
        hipcc_bin = shutil.which("hipcc") or (str(rocm_dir / "bin" / "hipcc") if (rocm_dir / "bin" / "hipcc").exists() else None)

        if rocm_smi_bin or env_spec.rocm_version:
            checks.append(
                DoctorCheck(
                    name="rocm_stack",
                    status=CheckStatus.PASS,
                    message=f"ROCm stack detected (version: {env_spec.rocm_version or 'detected'}, path: {rocm_path}).",
                    details={
                        "rocm_version": env_spec.rocm_version,
                        "rocm_path": rocm_path,
                        "rocm_smi": rocm_smi_bin,
                        "hipcc": hipcc_bin,
                    },
                )
            )
        else:
            checks.append(
                DoctorCheck(
                    name="rocm_stack",
                    status=CheckStatus.WARN,
                    message="ROCm userspace stack not found (rocm-smi / /opt/rocm absent).",
                    details={"rocm_path": rocm_path, "checked_paths": ["/opt/rocm"]},
                )
            )

        # 5. PyTorch HIP Runtime Check
        torch_installed = False
        torch_version: Optional[str] = None
        hip_version: Optional[str] = None
        cuda_avail = False
        cuda_device_count = 0

        try:
            import torch  # type: ignore[import-untyped]

            torch_installed = True
            torch_version = str(getattr(torch, "__version__", None))
            hip_version = getattr(torch.version, "hip", None)
            cuda_avail = bool(torch.cuda.is_available())
            cuda_device_count = torch.cuda.device_count() if cuda_avail else 0
        except ImportError:
            pass

        if not torch_installed:
            checks.append(
                DoctorCheck(
                    name="pytorch_hip",
                    status=CheckStatus.WARN,
                    message="PyTorch is not installed in the current environment.",
                )
            )
        elif hip_version is not None and cuda_avail:
            checks.append(
                DoctorCheck(
                    name="pytorch_hip",
                    status=CheckStatus.PASS,
                    message=f"PyTorch {torch_version} built with HIP {hip_version} is functional ({cuda_device_count} device(s) ready).",
                    details={
                        "torch_version": torch_version,
                        "hip_version": hip_version,
                        "cuda_available": cuda_avail,
                        "device_count": cuda_device_count,
                    },
                )
            )
        elif hip_version is not None and not cuda_avail:
            checks.append(
                DoctorCheck(
                    name="pytorch_hip",
                    status=CheckStatus.WARN,
                    message=f"PyTorch {torch_version} is built with HIP {hip_version}, but torch.cuda.is_available() returned False.",
                    details={"torch_version": torch_version, "hip_version": hip_version},
                )
            )
        else:
            checks.append(
                DoctorCheck(
                    name="pytorch_hip",
                    status=CheckStatus.INFO,
                    message=f"PyTorch {torch_version} is installed without HIP support (CPU / CUDA build).",
                    details={"torch_version": torch_version, "hip_version": None},
                )
            )

        # 6. Overall Verdict Computation
        has_amd = len(amd_gpus) > 0
        has_rocm = env_spec.rocm_version is not None or rocm_smi_bin is not None
        has_torch_hip = hip_version is not None and cuda_avail

        if system == "Linux" and has_amd and has_rocm and has_torch_hip:
            verdict = DoctorVerdict.READY
            summary = "AMD ROCm environment is fully configured and ready for live accelerated inference."
        elif not has_amd:
            verdict = DoctorVerdict.NO_ACCELERATOR
            summary = f"No AMD ROCm GPU accelerator detected on this host ({system} {arch}). Diagnostic mode active."
        else:
            verdict = DoctorVerdict.WARNING
            summary = "AMD hardware detected, but ROCm driver, permissions, or PyTorch HIP stack requires configuration."

        return DoctorReport(
            verdict=verdict,
            summary=summary,
            checks=checks,
            environment={
                "os": env_spec.os,
                "kernel": env_spec.kernel,
                "architecture": env_spec.architecture,
                "python_version": env_spec.python_version,
                "torch_version": torch_version or env_spec.torch_version,
                "rocm_version": env_spec.rocm_version,
                "hip_version": hip_version or env_spec.hip_version,
            },
            gpus=[g.model_dump() for g in hw_report.gpus],
        )
