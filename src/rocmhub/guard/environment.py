"""Environment fingerprinting and drift detection for Benchmark Guard."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple

from rocmhub.core.types import DetectionReport, EnvironmentFingerprint, EnvironmentSpec, HardwareSpec


def extract_environment_fingerprint(
    report: Optional[DetectionReport] = None,
    *,
    environment: Optional[EnvironmentSpec] = None,
    gpus: Optional[List[HardwareSpec]] = None,
) -> EnvironmentFingerprint:
    """Extract a canonical, deterministic environment fingerprint.

    Accepts either a full DetectionReport, or explicitly environment and gpus.
    Excludes ephemeral fields (PIDs, timestamps, temp paths) and normalizes GPU lists.
    """
    if report is not None:
        env = report.environment
        raw_gpus = report.gpus
    elif environment is not None:
        env = environment
        raw_gpus = gpus or []
    else:
        raise ValueError("Either report or environment must be provided to extract fingerprint")

    # Normalize GPU list deterministically
    sorted_gpus: List[HardwareSpec] = sorted(
        raw_gpus,
        key=lambda g: (g.device_id if g.device_id is not None else -1, g.bus_id or "", g.device_name or ""),
    )

    normalized_gpus = [
        {
            "device_id": g.device_id,
            "gpu_present": g.gpu_present,
            "gpu_vendor": g.gpu_vendor,
            "device_name": g.device_name,
            "family": g.family,
            "gfx_target": g.gfx_target,
            "vram_total_mb": g.vram_total_mb,
            "compute_units": g.compute_units,
            "bus_id": g.bus_id,
        }
        for g in sorted_gpus
    ]

    # Environment attributes excluding any non-reproducible host state
    attributes: Dict[str, Any] = {
        "os": env.os,
        "kernel": env.kernel,
        "architecture": env.architecture,
        "python_version": env.python_version,
        "rocm_version": env.rocm_version,
        "hip_version": env.hip_version,
        "torch_version": env.torch_version,
        "torch_hip_available": env.torch_hip_available,
        "env_vars": dict(sorted(env.env_vars.items())),
        "gpus": normalized_gpus,
    }

    # Canonical JSON representation
    canonical_json = json.dumps(attributes, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    fingerprint_hash = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    return EnvironmentFingerprint(
        fingerprint_hash=fingerprint_hash,
        attributes=attributes,
    )


def compare_fingerprints(
    expected: EnvironmentFingerprint,
    observed: EnvironmentFingerprint,
) -> Tuple[bool, Dict[str, Any]]:
    """Compare expected and observed environment fingerprints.

    Returns:
        Tuple of (is_match, diff_dict). If matched, diff_dict is empty.
    """
    if expected.fingerprint_hash == observed.fingerprint_hash:
        return True, {}

    diff: Dict[str, Any] = {}
    exp_attrs = expected.attributes
    obs_attrs = observed.attributes

    all_keys = set(exp_attrs.keys()).union(set(obs_attrs.keys()))
    for k in sorted(all_keys):
        val_exp = exp_attrs.get(k)
        val_obs = obs_attrs.get(k)
        if val_exp != val_obs:
            diff[k] = {
                "expected": val_exp,
                "observed": val_obs,
            }

    return False, diff
