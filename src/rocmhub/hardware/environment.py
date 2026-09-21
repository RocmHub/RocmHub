"""Environment observation: collects host OS, kernel, python, torch, and ROCm runtime details."""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from typing import Dict, Optional, Tuple

from rocmhub.core.types import EnvironmentSpec

ROCM_ENV_WHITELIST = (
    "HSA_OVERRIDE_GFX_VERSION",
    "ROCR_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "PYTORCH_ROCM_ARCH",
    "AMD_SERIALIZE_KERNEL",
    "HIP_FORCE_DEV_KERNARG",
    "GPU_MAX_HW_QUEUES",
    "AMD_LOG_LEVEL",
    "HIP_PATH",
    "ROCM_PATH",
)


def _safe_read_file(path: str) -> Optional[str]:
    """Read file content safely without root or external shell."""
    try:
        if os.path.isfile(path) and os.access(path, os.R_OK):
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read().strip()
                return content if content else None
    except OSError:
        pass
    return None


def _run_tool(cmd: list[str], timeout_sec: float = 3.0) -> Optional[str]:
    """Execute a system binary safely with explicit timeout, no shell, and fixed locale."""
    binary = cmd[0]
    if not shutil.which(binary) and not os.path.isfile(binary):
        return None
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_sec,
            env={**os.environ, "LC_ALL": "C", "LANG": "C"},
        )
        if proc.returncode == 0 and proc.stdout:
            return proc.stdout.strip()
    except (subprocess.TimeoutExpired, OSError):
        pass
    return None


class EnvironmentDetector:
    """Discovers host operating system, runtime, and ROCm environment."""

    def __init__(self, rocm_path: str = "/opt/rocm") -> None:
        self.rocm_path = rocm_path

    def _detect_torch(self) -> Tuple[str, bool, Optional[str], Optional[str]]:
        """Probe PyTorch for version, HIP availability, and HIP runtime version."""
        try:
            import torch  # type: ignore

            torch_ver = str(getattr(torch, "__version__", "unknown"))
            hip_ver = getattr(getattr(torch, "version", None), "hip", None)
            hip_available = False

            if hasattr(torch, "cuda") and torch.cuda.is_available():
                if hip_ver is not None:
                    hip_available = True

            return torch_ver, hip_available, hip_ver, "torch.version.hip"
        except ImportError:
            return "not_installed", False, None, None

    def _detect_rocm_version(self) -> Tuple[Optional[str], Optional[str]]:
        """Probe ROCm package version from standard paths and commands."""
        # 1. /opt/rocm/.info/version
        info_ver_path = os.path.join(self.rocm_path, ".info", "version")
        ver = _safe_read_file(info_ver_path)
        if ver:
            # Often formatted as e.g. "6.2.0-12345"
            match = re.search(r"(\d+\.\d+\.\d+)", ver)
            return match.group(1) if match else ver, info_ver_path

        # 2. /opt/rocm/share/rocm/version
        share_ver_path = os.path.join(self.rocm_path, "share", "rocm", "version")
        ver = _safe_read_file(share_ver_path)
        if ver:
            match = re.search(r"(\d+\.\d+\.\d+)", ver)
            return match.group(1) if match else ver, share_ver_path

        # 3. rocm-smi --version
        smi_out = _run_tool(["rocm-smi", "--version"])
        if smi_out:
            match = re.search(r"version[:\s]+(\d+\.\d+\.\d+)", smi_out, re.IGNORECASE)
            if match:
                return match.group(1), "rocm-smi --version"

        # 4. /opt/rocm/bin/rocminfo or rocminfo in PATH
        rocminfo_bin = shutil.which("rocminfo") or os.path.join(self.rocm_path, "bin", "rocminfo")
        rocminfo_out = _run_tool([rocminfo_bin])
        if rocminfo_out:
            match = re.search(r"Runtime Version:\s*(\d+\.\d+[\.\d]*)", rocminfo_out)
            if match:
                return match.group(1), f"{rocminfo_bin}"

        return None, None

    def _detect_hip_version(self, torch_hip_ver: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
        """Probe HIP version from torch, hipconfig, or headers."""
        if torch_hip_ver:
            return torch_hip_ver, "torch.version.hip"

        # Try hipconfig --version
        hipconfig_bin = shutil.which("hipconfig") or os.path.join(self.rocm_path, "bin", "hipconfig")
        hip_out = _run_tool([hipconfig_bin, "--version"])
        if hip_out:
            match = re.search(r"(\d+\.\d+[\.\d]*)", hip_out)
            if match:
                return match.group(1), f"{hipconfig_bin} --version"

        # Try /opt/rocm/include/hip/hip_version.h
        header_path = os.path.join(self.rocm_path, "include", "hip", "hip_version.h")
        header_content = _safe_read_file(header_path)
        if header_content:
            major = re.search(r"#define\s+HIP_VERSION_MAJOR\s+(\d+)", header_content)
            minor = re.search(r"#define\s+HIP_VERSION_MINOR\s+(\d+)", header_content)
            patch = re.search(r"#define\s+HIP_VERSION_PATCH\s+(\d+)", header_content)
            if major and minor:
                p = patch.group(1) if patch else "0"
                return f"{major.group(1)}.{minor.group(1)}.{p}", header_path

        return None, None

    def _collect_whitelisted_env_vars(self) -> Dict[str, str]:
        """Collect only whitelisted ROCm/HIP environment variables to prevent secret leakage."""
        collected: Dict[str, str] = {}
        for var in ROCM_ENV_WHITELIST:
            val = os.environ.get(var)
            if val is not None:
                collected[var] = val
        return collected

    def detect(self) -> Tuple[EnvironmentSpec, Dict[str, str], list[str]]:
        """Observe host software environment.

        Returns:
            (EnvironmentSpec, provenance_dict, warnings)
        """
        provenance: Dict[str, str] = {}
        warnings: list[str] = []

        # System and kernel
        system = platform.system()
        kernel_release = platform.release()
        arch = platform.machine()
        os_str = f"{system} {kernel_release}".strip()
        provenance["os"] = "platform.system + platform.release"
        provenance["kernel"] = "platform.release"
        provenance["architecture"] = "platform.machine"

        # Python version
        py_ver = platform.python_version()
        provenance["python_version"] = "platform.python_version"

        # PyTorch details
        torch_ver, torch_hip_avail, torch_hip_ver, torch_prov = self._detect_torch()
        provenance["torch_version"] = "python_import"
        if torch_prov:
            provenance["torch_hip_available"] = torch_prov

        # ROCm version
        rocm_ver, rocm_prov = self._detect_rocm_version()
        if rocm_prov:
            provenance["rocm_version"] = rocm_prov

        # HIP version
        hip_ver, hip_prov = self._detect_hip_version(torch_hip_ver)
        if hip_prov:
            provenance["hip_version"] = hip_prov

        # Filtered environment variables (Strict whitelist)
        env_vars = self._collect_whitelisted_env_vars()
        if env_vars:
            provenance["env_vars"] = "whitelisted_os.environ"

        spec = EnvironmentSpec(
            os=os_str,
            kernel=kernel_release,
            architecture=arch,
            python_version=py_ver,
            rocm_version=rocm_ver,
            hip_version=hip_ver,
            torch_version=torch_ver,
            torch_hip_available=torch_hip_avail,
            env_vars=env_vars,
        )

        return spec, provenance, warnings
