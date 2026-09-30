"""Health check route for ROCmHub API."""

from __future__ import annotations

import platform
import sys
from typing import Any, Dict

from fastapi import APIRouter, Request

from rocmhub.hardware.detector import SystemObserver

router = APIRouter(tags=["System"])


@router.get("/health")
async def health_check(request: Request) -> Dict[str, Any]:
    """Health status, platform environment, and hardware summary."""
    manager = request.app.state.job_manager

    observer = SystemObserver()
    detection_report = observer.observe()
    gpus = detection_report.gpus
    env = detection_report.environment

    rocm_gpus = [
        {
            "device_name": g.device_name,
            "vendor": g.gpu_vendor,
            "gfx_target": g.gfx_target,
            "vram_total_mb": g.vram_total_mb,
            "gpu_present": g.gpu_present,
        }
        for g in gpus
        if g.gpu_vendor and g.gpu_vendor.lower() == "amd"
    ]
    rocm_available = len(rocm_gpus) > 0 and (env.rocm_version is not None or sys.platform != "darwin")

    warnings = []
    if sys.platform == "darwin":
        warnings.append(
            "Running on macOS (Apple Silicon / Darwin). AMD ROCm GPU acceleration is not available on this host. "
            "Model preparation operates in CONFIG_ONLY mode without synthetic GPU metrics."
        )
    elif not rocm_available:
        warnings.append(
            "No AMD ROCm GPU detected on this host. Model preparation operates in CONFIG_ONLY mode."
        )

    return {
        "status": "healthy",
        "version": "0.1.0",
        "rocm_available": rocm_available,
        "host_platform": {
            "os": env.os,
            "arch": platform.machine(),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "is_apple_silicon": sys.platform == "darwin",
        },
        "system": {
            "os": env.os,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "rocm_version": env.rocm_version,
            "torch_version": env.torch_version,
            "gpus_detected": len(rocm_gpus),
            "gpus": rocm_gpus,
        },
        "orchestrator": {
            "queue_size": manager._work_queue.qsize() if hasattr(manager, "_work_queue") else 0,
            "active_directory_lock_count": len(manager._active_directory_locks)
            if hasattr(manager, "_active_directory_locks")
            else 0,
        },
        "warnings": warnings,
    }
