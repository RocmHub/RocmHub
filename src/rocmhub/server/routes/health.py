"""Health check route for ROCmHub API."""

from __future__ import annotations

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

    gpu_info = []
    for g in gpus:
        gpu_info.append({
            "device_name": g.device_name,
            "vendor": g.gpu_vendor,
            "gfx_target": g.gfx_target,
            "vram_total_mb": g.vram_total_mb,
            "gpu_present": g.gpu_present,
        })

    return {
        "status": "healthy",
        "version": "0.1.0",
        "system": {
            "os": env.os,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "rocm_version": env.rocm_version,
            "torch_version": env.torch_version,
            "gpus_detected": len(gpus),
            "gpus": gpu_info,
        },
        "orchestrator": {
            "queue_size": manager._work_queue.qsize() if hasattr(manager, "_work_queue") else 0,
            "active_directory_locks": list(manager._active_directory_locks) if hasattr(manager, "_active_directory_locks") else [],
        },
    }
