"""AMD and host GPU hardware discovery without compatibility policy decisions."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from typing import Callable, Dict, List, Optional, Tuple

from rocmhub.core.types import DetectionReport, HardwareSpec
from rocmhub.hardware.base import HardwareDetector
from rocmhub.hardware.environment import EnvironmentDetector


def _run_tool(cmd: List[str], timeout_sec: float = 3.0) -> Optional[str]:
    """Execute command safely with explicit timeout, no shell, and fixed locale."""
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


def _detect_family_from_name(device_name: Optional[str]) -> Optional[str]:
    """Determine architectural family (Radeon / Instinct) strictly without guessing."""
    if not device_name:
        return None
    lower = device_name.lower()
    if "radeon" in lower:
        return "Radeon"
    if "instinct" in lower:
        return "Instinct"
    return None


class SystemHardwareDetector(HardwareDetector):
    """Observes available host GPUs using multi-tier fallback discovery.

    Priority Order:
    1. PyTorch ROCm/HIP runtime (if torch with HIP is functional)
    2. rocminfo CLI
    3. amd-smi CLI
    4. rocm-smi CLI
    5. Linux sysfs / KFD / DRM topology metadata
    """

    def __init__(
        self,
        cmd_runner: Callable[[List[str], float], Optional[str]] = _run_tool,
        sysfs_root: str = "/sys/class",
    ) -> None:
        self._cmd_runner = cmd_runner
        self._sysfs_root = sysfs_root

    def _detect_via_torch(self) -> Tuple[List[HardwareSpec], Dict[str, str]]:
        """Tier 1: Discover GPUs via PyTorch HIP runtime."""
        specs: List[HardwareSpec] = []
        prov: Dict[str, str] = {}
        try:
            import torch  # type: ignore

            if not hasattr(torch, "cuda") or not torch.cuda.is_available():
                return [], {}

            hip_ver = getattr(getattr(torch, "version", None), "hip", None)
            if not hip_ver:
                return [], {}

            device_count = torch.cuda.device_count()
            for i in range(device_count):
                device_name = torch.cuda.get_device_name(i)
                props = torch.cuda.get_device_properties(i)

                vram_bytes = getattr(props, "total_memory", 0)
                vram_mb = vram_bytes // (1024 * 1024) if vram_bytes > 0 else None
                cu_count = getattr(props, "multi_processor_count", None)

                gcn_arch = getattr(props, "gcnArchName", "")
                gfx_match = re.search(r"(gfx[0-9a-zA-Z]+)", gcn_arch)
                gfx_target = gfx_match.group(1) if gfx_match else None

                family = _detect_family_from_name(device_name)

                spec = HardwareSpec(
                    gpu_present=True,
                    gpu_vendor="AMD",
                    device_id=i,
                    device_name=device_name,
                    family=family,
                    gfx_target=gfx_target,
                    vram_total_mb=vram_mb,
                    compute_units=cu_count,
                )
                specs.append(spec)
                prov[f"gpu[{i}]"] = "torch.cuda (ROCm HIP runtime)"

        except Exception:
            # PyTorch discovery failed, proceed to next source
            return [], {}

        return specs, prov

    def _parse_rocminfo_output(self, output: str) -> List[HardwareSpec]:
        """Parse raw rocminfo stdout into HardwareSpec list."""
        specs: List[HardwareSpec] = []
        # Split by agent blocks
        agent_blocks = re.split(r"\*+\s*\n\s*Agent\s+\d+\s*\n\s*\*+", output)

        device_index = 0
        for block in agent_blocks:
            # Must be a GPU agent (indicated by Device Type: GPU, Name: gfx..., or amdgcn ISA)
            is_gpu = (
                re.search(r"Device Type:\s*GPU", block, re.IGNORECASE) is not None
                or re.search(r"Name:\s*gfx", block, re.IGNORECASE) is not None
                or re.search(r"amdgcn-amd-amdhsa", block, re.IGNORECASE) is not None
            )
            # Explicitly reject CPU agents
            is_cpu = re.search(r"Device Type:\s*CPU", block, re.IGNORECASE) is not None
            if not is_gpu or is_cpu:
                continue

            # Device marketing name
            mkt_match = re.search(r"Marketing Name:\s*(.+)", block)
            device_name = mkt_match.group(1).strip() if mkt_match else None

            # Target architecture (Name or ISA name)
            gfx_target: Optional[str] = None
            isa_match = re.search(r"amdgcn-amd-amdhsa--(gfx[0-9a-zA-Z_]+)", block)
            if isa_match:
                gfx_target = isa_match.group(1).strip()
            else:
                name_match = re.search(r"Name:\s*(gfx[0-9a-zA-Z_]+)", block)
                if name_match:
                    gfx_target = name_match.group(1).strip()

            # Compute units
            cu_match = re.search(r"Compute Unit:\s*(\d+)", block)
            compute_units = int(cu_match.group(1)) if cu_match else None

            # VRAM total from Global Fast pool
            vram_total_mb: Optional[int] = None
            vram_match = re.search(
                r"Segment:\s*GLOBAL(?:;\s*FAST)?.*?Size:\s*(\d+)\(0x[0-9a-fA-F]+\)\s*KB",
                block,
                re.DOTALL,
            )
            if vram_match:
                vram_kb = int(vram_match.group(1))
                vram_total_mb = round(vram_kb / 1024)

            family = _detect_family_from_name(device_name)

            spec = HardwareSpec(
                gpu_present=True,
                gpu_vendor="AMD",
                device_id=device_index,
                device_name=device_name,
                family=family,
                gfx_target=gfx_target,
                vram_total_mb=vram_total_mb,
                compute_units=compute_units,
            )
            specs.append(spec)
            device_index += 1

        return specs

    def _detect_via_rocminfo(self) -> Tuple[List[HardwareSpec], Dict[str, str], Optional[str]]:
        """Tier 2: Discover GPUs via rocminfo CLI."""
        rocminfo_bin = shutil.which("rocminfo") or "/opt/rocm/bin/rocminfo"
        out = self._cmd_runner([rocminfo_bin], 4.0)
        if not out:
            return [], {}, "rocminfo tool not found or returned no output"

        specs = self._parse_rocminfo_output(out)
        if not specs:
            return [], {}, "rocminfo ran but no GPU agents were found"

        prov = {f"gpu[{i}]": f"{rocminfo_bin}" for i in range(len(specs))}
        return specs, prov, None

    def _detect_via_amd_smi(self) -> Tuple[List[HardwareSpec], Dict[str, str], Optional[str]]:
        """Tier 3: Discover GPUs via amd-smi CLI."""
        amd_smi_bin = shutil.which("amd-smi") or "/opt/rocm/bin/amd-smi"
        out = self._cmd_runner([amd_smi_bin, "static", "--json"], 4.0)
        if not out:
            # Try plain list
            out = self._cmd_runner([amd_smi_bin, "list", "--json"], 4.0)

        if not out:
            return [], {}, "amd-smi tool not found or returned no output"

        try:
            data = json.loads(out)
            if not isinstance(data, list):
                data = [data]

            specs: List[HardwareSpec] = []
            for item in data:
                asic = item.get("asic", {})
                vram = item.get("vram", {})

                dev_name = asic.get("market_name") or item.get("device_name")
                gfx_target = asic.get("target_graphics_version") or item.get("gfx_target")
                vram_mb = vram.get("vram_size") or item.get("vram_total_mb")
                gpu_id = item.get("gpu", len(specs))

                spec = HardwareSpec(
                    gpu_present=True,
                    gpu_vendor="AMD",
                    device_id=gpu_id,
                    device_name=dev_name,
                    family=_detect_family_from_name(dev_name),
                    gfx_target=str(gfx_target) if gfx_target else None,
                    vram_total_mb=int(vram_mb) if vram_mb else None,
                    compute_units=None,
                )
                specs.append(spec)

            if specs:
                prov = {f"gpu[{i}]": f"{amd_smi_bin}" for i in range(len(specs))}
                return specs, prov, None
            return [], {}, "amd-smi output JSON contained no devices"
        except Exception as exc:
            return [], {}, f"Failed to parse amd-smi JSON output: {exc}"

    def _detect_via_rocm_smi(self) -> Tuple[List[HardwareSpec], Dict[str, str], Optional[str]]:
        """Tier 4: Discover GPUs via rocm-smi CLI."""
        rocm_smi_bin = shutil.which("rocm-smi") or "/opt/rocm/bin/rocm-smi"
        out = self._cmd_runner([rocm_smi_bin, "--showproductname", "--showmeminfo", "vram", "--json"], 4.0)
        if not out:
            return [], {}, "rocm-smi tool not found or returned no output"

        try:
            data = json.loads(out)
            specs: List[HardwareSpec] = []

            # rocm-smi JSON is keyed by card0, card1, etc.
            idx = 0
            for card_key, card_info in sorted(data.items()):
                if not isinstance(card_info, dict):
                    continue
                dev_name = card_info.get("Card Series") or card_info.get("Card model")
                vram_bytes_str = card_info.get("VRAM Total Memory (B)")
                vram_mb = round(int(vram_bytes_str) / (1024 * 1024)) if vram_bytes_str and vram_bytes_str.isdigit() else None

                spec = HardwareSpec(
                    gpu_present=True,
                    gpu_vendor="AMD",
                    device_id=idx,
                    device_name=dev_name,
                    family=_detect_family_from_name(dev_name),
                    gfx_target=None,
                    vram_total_mb=vram_mb,
                    compute_units=None,
                )
                specs.append(spec)
                idx += 1

            if specs:
                prov = {f"gpu[{i}]": f"{rocm_smi_bin}" for i in range(len(specs))}
                return specs, prov, None
            return [], {}, "rocm-smi output contained no card entries"
        except Exception as exc:
            return [], {}, f"Failed to parse rocm-smi JSON output: {exc}"

    def _detect_via_sysfs(self) -> Tuple[List[HardwareSpec], Dict[str, str]]:
        """Tier 5: Fallback scan of Linux /sys/class/kfd and /sys/class/drm."""
        specs: List[HardwareSpec] = []
        prov: Dict[str, str] = {}

        # 1. Check KFD topology
        kfd_nodes_path = os.path.join(self._sysfs_root, "kfd", "kfd", "topology", "nodes")
        if os.path.isdir(kfd_nodes_path):
            try:
                for node_dir in sorted(os.listdir(kfd_nodes_path)):
                    props_path = os.path.join(kfd_nodes_path, node_dir, "properties")
                    if os.path.isfile(props_path) and os.access(props_path, os.R_OK):
                        with open(props_path, "r", encoding="utf-8", errors="replace") as f:
                            content = f.read()

                        # Check GPU node
                        name_match = re.search(r"name\s+(gfx[0-9a-zA-Z]+)", content)
                        simd_match = re.search(r"simd_count\s+(\d+)", content)
                        vram_match = re.search(r"vram_size_kb\s+(\d+)", content)

                        if name_match:
                            gfx_target = name_match.group(1)
                            cu = int(simd_match.group(1)) // 4 if simd_match else None
                            vram_mb = int(vram_match.group(1)) // 1024 if vram_match else None

                            spec = HardwareSpec(
                                gpu_present=True,
                                gpu_vendor="AMD",
                                device_id=len(specs),
                                device_name=f"AMD GPU ({gfx_target})",
                                family=_detect_family_from_name(gfx_target),
                                gfx_target=gfx_target,
                                vram_total_mb=vram_mb,
                                compute_units=cu,
                            )
                            specs.append(spec)
                            prov[f"gpu[{len(specs)-1}]"] = props_path

                if specs:
                    return specs, prov
            except OSError:
                pass

        # 2. Check DRM devices (/sys/class/drm/card*/device)
        drm_path = os.path.join(self._sysfs_root, "drm")
        if os.path.isdir(drm_path):
            try:
                card_dirs = [d for d in os.listdir(drm_path) if re.match(r"^card\d+$", d)]
                for card in sorted(card_dirs):
                    vendor_path = os.path.join(drm_path, card, "device", "vendor")
                    if os.path.isfile(vendor_path):
                        with open(vendor_path, "r", encoding="utf-8", errors="replace") as vf:
                            vendor_id = vf.read().strip()
                        # AMD vendor ID is 0x1002
                        if vendor_id.lower() == "0x1002":
                            spec = HardwareSpec(
                                gpu_present=True,
                                gpu_vendor="AMD",
                                device_id=len(specs),
                                device_name="AMD GPU (DRM)",
                                family=None,
                                gfx_target=None,
                                vram_total_mb=None,
                                compute_units=None,
                            )
                            specs.append(spec)
                            prov[f"gpu[{len(specs)-1}]"] = vendor_path
            except OSError:
                pass

        return specs, prov

    def detect_gpus(self) -> Tuple[List[HardwareSpec], Dict[str, str], List[str]]:
        """Observe available host GPUs using ordered fallbacks.

        Returns:
            (gpus, provenance, warnings)
        """
        all_warnings: List[str] = []

        # Tier 1: PyTorch HIP runtime
        torch_specs, torch_prov = self._detect_via_torch()
        if torch_specs:
            return torch_specs, torch_prov, all_warnings

        # Tier 2: rocminfo
        rocminfo_specs, rocminfo_prov, rocminfo_warn = self._detect_via_rocminfo()
        if rocminfo_warn:
            all_warnings.append(rocminfo_warn)
        if rocminfo_specs:
            return rocminfo_specs, rocminfo_prov, all_warnings

        # Tier 3: amd-smi
        amd_specs, amd_prov, amd_warn = self._detect_via_amd_smi()
        if amd_warn:
            all_warnings.append(amd_warn)
        if amd_specs:
            return amd_specs, amd_prov, all_warnings

        # Tier 4: rocm-smi
        rocm_specs, rocm_prov, rocm_warn = self._detect_via_rocm_smi()
        if rocm_warn:
            all_warnings.append(rocm_warn)
        if rocm_specs:
            return rocm_specs, rocm_prov, all_warnings

        # Tier 5: sysfs / KFD / DRM
        sysfs_specs, sysfs_prov = self._detect_via_sysfs()
        if sysfs_specs:
            return sysfs_specs, sysfs_prov, all_warnings

        # No AMD GPU found (normal diagnostic outcome on macOS, CPU, or non-AMD host)
        return [], {}, all_warnings


class SystemObserver:
    """Combines EnvironmentDetector and HardwareDetector to produce a complete DetectionReport."""

    def __init__(
        self,
        hardware_detector: Optional[HardwareDetector] = None,
        environment_detector: Optional[EnvironmentDetector] = None,
    ) -> None:
        self.hw_detector = hardware_detector or SystemHardwareDetector()
        self.env_detector = environment_detector or EnvironmentDetector()

    def observe(self) -> DetectionReport:
        """Produce unified DetectionReport representing host state."""
        env_spec, env_prov, env_warn = self.env_detector.detect()
        gpus, hw_prov, hw_warn = self.hw_detector.detect_gpus()

        combined_prov: Dict[str, str] = {**env_prov, **hw_prov}
        combined_warn: List[str] = env_warn + hw_warn

        return DetectionReport(
            environment=env_spec,
            gpus=gpus,
            provenance=combined_prov,
            warnings=combined_warn,
        )
