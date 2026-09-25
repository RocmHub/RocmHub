"""Opt-in public Qwen snapshot E2E; never downloads during the default suite."""

from __future__ import annotations

import os
import threading
from pathlib import Path

import pytest

from rocmhub.forge.materializer import ModelMaterializer
from rocmhub.server.orchestrator.models import JobStatus, JobType
from rocmhub.server.orchestrator.worker import execute_job

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


@pytest.mark.network
@pytest.mark.skipif(os.environ.get("ROCMHUB_RUN_QWEN_E2E") != "1", reason="explicit public-model download opt-in required")
def test_real_qwen_materialization_cache_miss_then_hit(tmp_path: Path) -> None:
    workspace = Path(__file__).resolve().parents[1] / "agent-workspace"
    workspace.mkdir(exist_ok=True)
    cache = workspace / "cache"
    def run(run_name: str):
        events = []
        result = execute_job(
            f"job-e2e-{run_name}",
            JobType.PREPARE_MODEL_FOR_AMD,
            {
                "model_id": MODEL,
                "revision": REVISION,
                "target_gpu": "Radeon RX 7900 XTX",
                "target_gfx": "gfx1100",
                "precision": "fp16",
                "runtime": "pytorch_transformers_hip",
                "materialization_mode": "FULL_WEIGHTS",
                "weights_consent": True,
                "expected_capabilities": ["PREPARE_MODEL_FOR_AMD"],
                "cache_policy": "REUSE",
                "output_dir": str(tmp_path / f"forge-{run_name}"),
                "execute_inference": False,
            },
            threading.Event(),
            lambda *event: events.append(event),
            ModelMaterializer(cache_dir=cache, managed_cache=True),
        )
        return result, events

    first, _ = run("first")
    assert first[0] == JobStatus.SUCCEEDED and first[1] == "PREPARED"
    assert first[3]["amd_validated"] is False
    assert first[3]["materialization"]["has_weights"] is True
    assert first[3]["materialization"]["cache_status"] in {"MISS_DOWNLOADED", "HIT_VERIFIED"}
    assert first[3]["materialization"]["materialized_bytes"] > 0

    second, _ = run("second")
    assert second[0] == JobStatus.SUCCEEDED and second[1] == "PREPARED"
    assert second[3]["materialization"]["cache_status"] == "HIT_VERIFIED"
    assert second[3]["amd_validated"] is False
