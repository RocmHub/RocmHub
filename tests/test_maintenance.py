"""Safety tests for pre-public production data cleanup."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from rocmhub.cli.main import main
from rocmhub.server.maintenance import (
    MaintenanceSafetyError,
    execute_development_cleanup,
    inspect_development_data,
)
from rocmhub.server.orchestrator.db import DatabaseManager
from rocmhub.server.orchestrator.models import AgentCapabilities, JobStatus, JobType


def _make_job_database(tmp_path: Path, *, development: bool, physical: bool = False) -> tuple[Path, Path, Path, str]:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    database_path = data_dir / "jobs.db"
    artifact_dir = data_dir / "agent-artifacts"
    database = DatabaseManager(database_path)
    now = datetime.now(timezone.utc)
    capabilities = AgentCapabilities(
        os="Darwin",
        architecture="arm64",
        python_version="3.9.6",
        rocm_detected=False,
        hip_detected=False,
        amd_gpu_count=0,
        capabilities=["PREPARE_MODEL_FOR_AMD"],
    )
    agent_id = "agent_old_dev"
    database.upsert_agent(agent_id, capabilities, hashlib.sha256(b"test-token").hexdigest(), 86_400)
    conn = database._get_connection()
    try:
        conn.execute(
            "UPDATE agents SET name=?,hostname=?,last_seen=?,status='OFFLINE' WHERE agent_id=?",
            (
                "mac-dev-agent" if development else "Preparation Agent",
                "mac-mini" if development else "",
                (now - timedelta(days=2)).isoformat(),
                agent_id,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    job_id = "job_test_cleanup" if development else "job_regular_history"
    database.insert_job(
        job_id=job_id,
        job_type=JobType.PREPARE_MODEL_FOR_AMD,
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        revision="7ae557604adf67be50417f59c2c2f167def9a775",
        status=JobStatus.QUEUED,
        created_at=now.isoformat(),
        timeout_seconds=600,
        output_dir=None,
        request_payload={"model_id": "Qwen/Qwen2.5-0.5B-Instruct"},
    )
    result = {
        "weights": "NOT_DOWNLOADED",
        "amd_validated": physical,
        "agent_observation": {"physical_amd_execution": physical},
        "server_artifacts": [{"name": "model_config.json", "sha256": "abc"}],
    }
    conn = database._get_connection()
    try:
        conn.execute(
            "UPDATE jobs SET status='SUCCEEDED',domain_status=?,result_payload=?,agent_id=? WHERE job_id=?",
            (
                "EXECUTED" if physical else "CONFIG_ONLY",
                json.dumps(result),
                agent_id,
                job_id,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    database.insert_event(job_id, "COMPLETED", "SUCCEEDED", "Preparation completed by connected compute.")
    target = artifact_dir / job_id / "attempt-1" / "model_config.json"
    target.parent.mkdir(parents=True)
    target.write_text("{}\n", encoding="utf-8")
    return data_dir, database_path, artifact_dir, job_id


def test_cleanup_dry_run_counts_records_without_mutating(tmp_path: Path) -> None:
    data_dir, database_path, artifact_dir, job_id = _make_job_database(tmp_path, development=True)
    before = database_path.stat().st_mtime_ns
    plan = inspect_development_data(database_path, data_dir, artifact_dir, 86_400)

    assert plan.safe_to_execute
    assert len(plan.agent_ids) == 1
    assert plan.job_ids == [job_id]
    assert plan.jobs_by_type == {"PREPARE_MODEL_FOR_AMD": 1}
    assert plan.event_count == 1
    assert plan.result_count == 1
    assert plan.artifact_record_count == 1
    assert plan.artifact_file_count == 1
    assert database_path.stat().st_mtime_ns == before
    assert (artifact_dir / job_id).is_dir()


def test_cleanup_backs_up_and_removes_only_marked_dev_data_with_artifacts(tmp_path: Path) -> None:
    data_dir, database_path, artifact_dir, job_id = _make_job_database(tmp_path, development=True)
    plan = inspect_development_data(database_path, data_dir, artifact_dir, 86_400)
    backup_path = execute_development_cleanup(plan, 86_400)

    assert backup_path.is_file()
    assert not (artifact_dir / job_id).exists()
    with sqlite3.connect(str(database_path)) as connection:
        assert connection.execute("SELECT COUNT(*) FROM agents").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM job_events").fetchone()[0] == 0
    with sqlite3.connect(str(backup_path)) as connection:
        assert connection.execute("SELECT COUNT(*) FROM agents").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM job_events").fetchone()[0] == 1


def test_ambiguous_job_blocks_cleanup(tmp_path: Path) -> None:
    data_dir, database_path, artifact_dir, _ = _make_job_database(tmp_path, development=False)
    plan = inspect_development_data(database_path, data_dir, artifact_dir, 86_400)

    assert plan.ambiguous_job_count == 1
    assert not plan.safe_to_execute
    with pytest.raises(MaintenanceSafetyError):
        execute_development_cleanup(plan, 86_400)
    with sqlite3.connect(str(database_path)) as connection:
        assert connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1


def test_physical_amd_evidence_stops_cleanup(tmp_path: Path) -> None:
    data_dir, database_path, artifact_dir, job_id = _make_job_database(tmp_path, development=True, physical=True)
    plan = inspect_development_data(database_path, data_dir, artifact_dir, 86_400)

    assert plan.physical_execution_job_ids == [job_id]
    assert not plan.safe_to_execute


def test_maintenance_cli_requires_explicit_data_directory(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.delenv("ROCMHUB_DATA_DIR", raising=False)
    monkeypatch.delenv("ROCMHUB_DB_PATH", raising=False)
    assert main(["maintenance", "clean-development-data", "--dry-run"]) == 1
    assert "set ROCMHUB_DATA_DIR explicitly" in capsys.readouterr().err


def test_maintenance_cli_defaults_to_read_only_dry_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data_dir = tmp_path / "configured-data"
    data_dir.mkdir()
    DatabaseManager(data_dir / "jobs.db")
    monkeypatch.setenv("ROCMHUB_DATA_DIR", str(data_dir))
    monkeypatch.delenv("ROCMHUB_DB_PATH", raising=False)

    assert main(["maintenance", "clean-development-data"]) == 0
    output = capsys.readouterr().out
    assert "Mode: DRY RUN" in output
    assert str(data_dir / "jobs.db") in output
    assert list(data_dir.glob("jobs-before-development-cleanup-*.sqlite3")) == []
