"""Tests for ROCmHub FastAPI Backend and Job Orchestration (Phase 13)."""

from __future__ import annotations

import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from rocmhub.core.types import ModelSpec
from rocmhub.models.base import RepositoryMetadata
from rocmhub.server.app import create_app
from rocmhub.server.config import ServerConfig
from rocmhub.server.orchestrator.models import (
    JobCreateRequest,
    JobStatus,
    JobType,
)
from rocmhub.server.security import redact_secrets, sanitize_payload, validate_job_path

SAMPLE_COMMIT_SHA = "7ae557604adf67be50417f59c2c2f167def9a775"


@pytest.fixture
def temp_env() -> Any:
    """Create isolated temporary directory for test DB and workspaces."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir).resolve()
        db_path = tmp_path / "test_jobs.db"
        workspace = tmp_path / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        config = ServerConfig(
            host="127.0.0.1",
            port=8000,
            db_path=db_path,
            allowed_workspaces=[workspace, tmp_path],
            max_queue_size=10,
            max_request_bytes=50_000,  # 50 KB for testing size limit
            default_job_timeout_seconds=60,
        )
        app = create_app(config)
        with TestClient(app) as client:
            yield client, config, app.state.job_manager, workspace


class TestSecurityAndSanitization:
    """Security boundaries, path traversal rejection, and secret scrubbing."""

    def test_secret_redaction(self) -> None:
        text = "Using API key hf_abcdef1234567890abcdef1234567890 and Bearer eyJhbGciOiJIUzI1NiIsIn"
        cleaned = redact_secrets(text)
        assert "hf_abcdef1234567890abcdef1234567890" not in cleaned
        assert "***REDACTED***" in cleaned

    def test_sanitize_payload(self) -> None:
        data = {
            "token": "secret_token_12345678",
            "nested": ["Bearer xyz9876543210abcd", 42],
        }
        sanitized = sanitize_payload(data)
        assert "***REDACTED***" in sanitized["token"]
        assert "***REDACTED***" in sanitized["nested"][0]

    def test_validate_job_path_allowed(self, temp_env: Any) -> None:
        _, config, _, workspace = temp_env
        valid_dir = workspace / "test_out"
        resolved = validate_job_path(valid_dir, config.allowed_workspaces)
        assert resolved == valid_dir.resolve()

    def test_validate_job_path_traversal_rejected(self, temp_env: Any) -> None:
        _, config, _, workspace = temp_env
        with pytest.raises(Exception):
            validate_job_path(workspace / ".." / ".." / "etc", config.allowed_workspaces)

    def test_validate_job_path_forbidden_root_rejected(self, temp_env: Any) -> None:
        _, config, _, _ = temp_env
        with pytest.raises(Exception):
            validate_job_path("/etc/forbidden", config.allowed_workspaces)

    def test_request_size_limit_middleware(self, temp_env: Any) -> None:
        client, _, _, _ = temp_env
        # Payload larger than 50KB limit
        large_body = {"data": "x" * 60_000}
        resp = client.post("/api/v1/jobs", json=large_body)
        assert resp.status_code == 413


class TestHealthAndModelEndpoints:
    """Health check and model metadata query endpoints."""

    def test_health_check(self, temp_env: Any) -> None:
        client, _, _, _ = temp_env
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "system" in data
        assert "orchestrator" in data

    @patch("rocmhub.models.huggingface.HuggingFaceModelSource.resolve_revision")
    @patch("rocmhub.models.inspector.ModelInspector.inspect")
    @patch("rocmhub.models.huggingface.HuggingFaceModelSource.get_repository_metadata")
    def test_get_model_info_success(self, mock_repo: Any, mock_inspect: Any, mock_resolve: Any, temp_env: Any) -> None:
        client, _, _, _ = temp_env
        mock_resolve.return_value = SAMPLE_COMMIT_SHA
        mock_inspect.return_value = ModelSpec(
            schema_version="1.0.0",
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
            requested_revision="main",
            commit_sha=SAMPLE_COMMIT_SHA,
            architecture="Qwen2ForCausalLM",
            parameter_count=494032768,
            context_length=32768,
            weights_format="safetensors",
        )
        mock_repo.return_value = RepositoryMetadata(
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
            resolved_commit_sha=SAMPLE_COMMIT_SHA,
            files=["config.json", "model.safetensors"],
            card_data={},
            safetensors_metadata={"total": 494032768},
            pipeline_tag="text-generation",
            tags=["text-generation", "qwen"],
        )

        resp = client.get("/api/v1/models/Qwen/Qwen2.5-0.5B-Instruct")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_id"] == "Qwen/Qwen2.5-0.5B-Instruct"
        assert data["commit_sha"] == SAMPLE_COMMIT_SHA
        assert data["architecture"] == "Qwen2ForCausalLM"
        assert data["parameter_count"] == 494032768

    @patch("rocmhub.models.huggingface.HuggingFaceModelSource.resolve_revision")
    @patch("rocmhub.models.inspector.ModelInspector.inspect")
    def test_forge_plan_endpoint(self, mock_inspect: Any, mock_resolve: Any, temp_env: Any) -> None:
        client, _, _, workspace = temp_env
        mock_resolve.return_value = SAMPLE_COMMIT_SHA
        mock_inspect.return_value = ModelSpec(
            schema_version="1.0.0",
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
            requested_revision="main",
            commit_sha=SAMPLE_COMMIT_SHA,
            architecture="Qwen2ForCausalLM",
            parameter_count=494032768,
            context_length=32768,
            weights_format="safetensors",
        )

        plan_req = {
            "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
            "precision": "fp16",
            "output_dir": str(workspace / "plan_test"),
        }
        resp = client.post("/api/v1/forge/plan", json=plan_req)
        assert resp.status_code == 200
        plan = resp.json()
        assert plan["model_id"] == "Qwen/Qwen2.5-0.5B-Instruct"
        assert plan["revision"] == SAMPLE_COMMIT_SHA
        assert plan["precision"] == "fp16"


class TestJobLifecycleAndOrchestration:
    """Full lifecycle: queue, execution, results, cancellation, concurrency, and recovery."""

    @patch("rocmhub.server.orchestrator.worker._execute_forge_build")
    def test_job_forge_build_lifecycle(self, mock_exec: Any, temp_env: Any) -> None:
        client, _, manager, workspace = temp_env
        out_dir = str(workspace / "bld_1")

        mock_exec.return_value = (
            JobStatus.SUCCEEDED,
            "CONFIG_ONLY",
            out_dir,
            {"status": "CONFIG_ONLY", "model_id": "Qwen/Qwen2.5-0.5B-Instruct"},
            None,
            None,
        )

        req_body = {
            "job_type": "FORGE_BUILD",
            "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
            "precision": "fp16",
            "output_dir": out_dir,
            "allow_full_weights": False,
        }

        resp = client.post("/api/v1/jobs", json=req_body)
        assert resp.status_code == 202
        job_data = resp.json()
        job_id = job_data["job_id"]
        assert job_data["status"] == "QUEUED"

        # Wait for worker thread to process job
        for _ in range(50):
            status_resp = client.get(f"/api/v1/jobs/{job_id}")
            assert status_resp.status_code == 200
            current = status_resp.json()
            if current["status"] in ("SUCCEEDED", "FAILED"):
                break
            time.sleep(0.1)

        assert current["status"] == "SUCCEEDED"
        assert current["domain_status"] == "CONFIG_ONLY"

        # Check result endpoint
        res_resp = client.get(f"/api/v1/jobs/{job_id}/result")
        assert res_resp.status_code == 200
        result_data = res_resp.json()
        assert result_data["job_status"] == "SUCCEEDED"
        assert result_data["domain_status"] == "CONFIG_ONLY"
        assert result_data["result"]["status"] == "CONFIG_ONLY"

    def test_job_cancellation_queued(self, temp_env: Any) -> None:
        client, _, manager, workspace = temp_env

        # Enqueue job
        req_body = {
            "job_type": "FORGE_BUILD",
            "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
            "output_dir": str(workspace / "cancel_test"),
        }

        # Cancel immediately before processing
        resp = client.post("/api/v1/jobs", json=req_body)
        job_id = resp.json()["job_id"]

        cancel_resp = client.post(f"/api/v1/jobs/{job_id}/cancel")
        assert cancel_resp.status_code == 200
        assert cancel_resp.json()["cancelled"] is True

        job_info = {"status": "RUNNING"}
        for _ in range(50):
            job_info = client.get(f"/api/v1/jobs/{job_id}").json()
            if job_info["status"] in ("CANCELLED", "SUCCEEDED", "FAILED"):
                break
            time.sleep(0.05)

        assert job_info["status"] in ("CANCELLED", "SUCCEEDED")

    def test_restart_recovery(self, temp_env: Any) -> None:
        _, config, manager, workspace = temp_env

        # Insert a simulated interrupted job in SQLite
        conn = sqlite3.connect(str(config.db_path))
        conn.execute(
            """
            INSERT INTO jobs (
                job_id, job_type, model_id, revision, status, domain_status,
                created_at, started_at, timeout_seconds, output_dir, request_payload
            ) VALUES ('job_stuck_1', 'FORGE_BUILD', 'test/model', 'main', 'RUNNING', NULL,
                      '2026-09-22T00:00:00Z', '2026-09-22T00:01:00Z', 600, '/tmp', '{}')
            """
        )
        conn.commit()
        conn.close()

        # Re-initialize manager (simulate restart)
        new_manager = create_app(config).state.job_manager
        new_manager.start()

        job = new_manager.get_job("job_stuck_1")
        assert job is not None
        assert job.status == JobStatus.FAILED
        assert "server restart" in (job.error_message or "").lower()
        new_manager.stop()

    @patch("rocmhub.server.orchestrator.worker._execute_forge_build")
    def test_sse_events_streaming(self, mock_exec: Any, temp_env: Any) -> None:
        client, _, manager, workspace = temp_env
        out_dir = str(workspace / "bld_sse")

        mock_exec.return_value = (
            JobStatus.SUCCEEDED,
            "CONFIG_ONLY",
            out_dir,
            {"status": "CONFIG_ONLY"},
            None,
            None,
        )

        req_body = {
            "job_type": "FORGE_BUILD",
            "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
            "output_dir": out_dir,
        }
        resp = client.post("/api/v1/jobs", json=req_body)
        job_id = resp.json()["job_id"]

        # Wait for job to complete
        for _ in range(50):
            current = client.get(f"/api/v1/jobs/{job_id}").json()
            if current["status"] in ("SUCCEEDED", "FAILED"):
                break
            time.sleep(0.1)

        # Connect to SSE endpoint
        sse_resp = client.get(f"/api/v1/jobs/{job_id}/events")
        assert sse_resp.status_code == 200
        body = sse_resp.text
        assert "id:" in body
        assert "event: message" in body
        assert "QUEUE" in body or "INITIALIZATION" in body

    def test_job_not_found(self, temp_env: Any) -> None:
        client, _, _, _ = temp_env
        resp = client.get("/api/v1/jobs/non_existent_id")
        assert resp.status_code == 404

    @patch("rocmhub.server.orchestrator.worker._execute_engineer")
    def test_job_engineer_lifecycle(self, mock_eng: Any, temp_env: Any) -> None:
        client, _, manager, workspace = temp_env
        out_dir = str(workspace / "eng_test")

        mock_eng.return_value = (
            JobStatus.SUCCEEDED,
            "CONFIG_ONLY",
            out_dir,
            {"status": "CONFIG_ONLY", "session_id": "eng_123"},
            None,
            None,
        )

        req_body = {
            "job_type": "ENGINEER",
            "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
            "objective": "PREPARE_AMD",
            "output_dir": out_dir,
        }
        resp = client.post("/api/v1/jobs", json=req_body)
        assert resp.status_code == 202
        job_id = resp.json()["job_id"]

        for _ in range(50):
            current = client.get(f"/api/v1/jobs/{job_id}").json()
            if current["status"] in ("SUCCEEDED", "FAILED"):
                break
            time.sleep(0.05)

        assert current["status"] == "SUCCEEDED"
        assert current["domain_status"] == "CONFIG_ONLY"

    @patch("rocmhub.server.orchestrator.worker._execute_optimization")
    def test_job_optimization_lifecycle(self, mock_opt: Any, temp_env: Any) -> None:
        client, _, manager, workspace = temp_env
        out_dir = str(workspace / "opt_test")

        mock_opt.return_value = (
            JobStatus.SUCCEEDED,
            "CONFIG_ONLY",
            out_dir,
            {"session_id": "opt_123", "baseline": {"status": "CONFIG_ONLY"}},
            None,
            None,
        )

        req_body = {
            "job_type": "OPTIMIZATION",
            "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
            "objective": "MAX_THROUGHPUT",
            "output_dir": out_dir,
        }
        resp = client.post("/api/v1/jobs", json=req_body)
        assert resp.status_code == 202
        job_id = resp.json()["job_id"]

        for _ in range(50):
            current = client.get(f"/api/v1/jobs/{job_id}").json()
            if current["status"] in ("SUCCEEDED", "FAILED"):
                break
            time.sleep(0.05)

        assert current["status"] == "SUCCEEDED"
        assert current["domain_status"] == "CONFIG_ONLY"

    def test_sse_reconnection_replay(self, temp_env: Any) -> None:
        client, _, manager, workspace = temp_env
        job = manager.enqueue_job(
            JobCreateRequest(
                job_type=JobType.FORGE_BUILD,
                model_id="Qwen/Qwen2.5-0.5B-Instruct",
                output_dir=str(workspace / "sse_replay"),
            )
        )
        # Emit multiple sequential events
        manager.emit_event(job.job_id, "PHASE_1", "RUNNING", "Message 1")
        manager.emit_event(job.job_id, "PHASE_2", "RUNNING", "Message 2")
        manager.emit_event(job.job_id, "PHASE_3", "SUCCESS", "Message 3")

        # Request with from_event_id=2 (should only replay events > 2)
        resp = client.get(f"/api/v1/jobs/{job.job_id}/events?from_event_id=2")
        assert resp.status_code == 200
        content = resp.text
        assert "Message 2" in content or "Message 3" in content

    def test_directory_locking_prevents_simultaneous_writes(self, temp_env: Any) -> None:
        client, _, manager, workspace = temp_env
        shared_dir = str(workspace / "shared_dir")

        # Acquire lock manually to simulate active job
        assert manager._acquire_dir_lock(shared_dir) is True
        # Second acquire must fail
        assert manager._acquire_dir_lock(shared_dir) is False

        # Release lock
        manager._release_dir_lock(shared_dir)
        # Now can acquire again
        assert manager._acquire_dir_lock(shared_dir) is True
        manager._release_dir_lock(shared_dir)

    def test_list_jobs_pagination_and_filtering(self, temp_env: Any) -> None:
        client, _, manager, workspace = temp_env

        # Enqueue multiple jobs
        manager.enqueue_job(
            JobCreateRequest(
                job_type=JobType.FORGE_BUILD,
                model_id="Qwen/Qwen2.5-0.5B-Instruct",
                output_dir=str(workspace / "list_1"),
            )
        )
        manager.enqueue_job(
            JobCreateRequest(
                job_type=JobType.ENGINEER,
                model_id="Qwen/Qwen2.5-0.5B-Instruct",
                output_dir=str(workspace / "list_2"),
            )
        )

        # Test GET /api/v1/jobs
        resp = client.get("/api/v1/jobs?limit=10&offset=0")
        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data
        assert data["total"] >= 2
        assert len(data["items"]) >= 2
        assert data["limit"] == 10
        assert data["offset"] == 0

        # Filter by job_type
        resp_filter = client.get("/api/v1/jobs?job_type=ENGINEER")
        assert resp_filter.status_code == 200
        data_filter = resp_filter.json()
        assert all(item["job_type"] == "ENGINEER" for item in data_filter["items"])

    def test_restart_recovery_refined_error_codes(self, temp_env: Any) -> None:
        client, config, manager, workspace = temp_env

        conn = sqlite3.connect(str(config.db_path))
        conn.execute(
            """
            INSERT INTO jobs (
                job_id, job_type, model_id, revision, status, domain_status,
                created_at, started_at, timeout_seconds, output_dir, request_payload
            ) VALUES ('job_run_1', 'FORGE_BUILD', 'test/m1', 'main', 'RUNNING', NULL,
                      '2026-09-22T00:00:00Z', '2026-09-22T00:01:00Z', 600, '/tmp/1', '{}')
            """
        )
        conn.execute(
            """
            INSERT INTO jobs (
                job_id, job_type, model_id, revision, status, domain_status,
                created_at, started_at, timeout_seconds, output_dir, request_payload
            ) VALUES ('job_queue_1', 'FORGE_BUILD', 'test/m2', 'main', 'QUEUED', NULL,
                      '2026-09-22T00:00:00Z', NULL, 600, '/tmp/2', '{}')
            """
        )
        conn.commit()

        # Trigger recovery
        interrupted = manager.db.mark_interrupted_jobs_as_failed("Maintenance restart")
        assert "job_run_1" in interrupted
        assert "job_queue_1" in interrupted

        job_run = manager.get_job("job_run_1")
        assert job_run is not None
        assert job_run.status == JobStatus.FAILED
        assert job_run.error_code == "EXECUTION_INTERRUPTED_BY_RESTART"
        assert "Working directory may contain partial build artifacts" in (job_run.error_message or "")

        job_queue = manager.get_job("job_queue_1")
        assert job_queue is not None
        assert job_queue.status == JobStatus.FAILED
        assert job_queue.error_code == "QUEUE_DISCARDED_ON_RESTART"


class TestApiHardening:
    """Regression tests for Phase 18 API hardening and security boundaries."""

    def test_malformed_job_id_rejected(self, temp_env: Any) -> None:
        client, _, _, _ = temp_env

        # Special chars and traversal patterns
        for invalid_id in ["bad!id", "job@123", "job%23456", "a" * 100]:
            resp = client.get(f"/api/v1/jobs/{invalid_id}")
            assert resp.status_code == 422, f"Expected 422 for job_id '{invalid_id}', got {resp.status_code}"

    def test_validate_job_path_exact_root_rejected(self, temp_env: Any) -> None:
        from rocmhub.core.errors import SecurityBoundaryError
        from rocmhub.server.security import validate_job_path

        _, _, _, workspace = temp_env
        allowed_roots = [workspace]

        # Subdirectory should pass
        valid_sub = validate_job_path(workspace / "sub_build", allowed_roots)
        assert valid_sub == (workspace / "sub_build").resolve()

        # Exact root must fail
        with pytest.raises(SecurityBoundaryError, match="matches an allowed workspace root exactly"):
            validate_job_path(workspace, allowed_roots)

    def test_model_id_path_traversal_rejected(self, temp_env: Any) -> None:
        client, _, _, _ = temp_env

        # Traversal and invalid patterns
        for bad_model in ["..%2F..%2Fetc%2Fpasswd", "bad..repo/name", "bad/repo/with/extra/slashes"]:
            resp = client.get(f"/api/v1/models/{bad_model}")
            assert resp.status_code == 400
            assert "Invalid model identifier" in resp.json()["detail"]

    def test_unhandled_exception_does_not_leak_stacktrace(self, temp_env: Any) -> None:
        _, config, _, _ = temp_env
        from fastapi.testclient import TestClient

        from rocmhub.server.app import create_app

        safe_app = create_app(config)
        safe_client = TestClient(safe_app, raise_server_exceptions=False)

        with patch("rocmhub.server.routes.health.SystemObserver.observe", side_effect=RuntimeError("SecretDatabasePassword123")):
            resp = safe_client.get("/health")
            assert resp.status_code == 500
            data = resp.json()
            assert data["error"] == "InternalServerError"
            assert "SecretDatabasePassword123" not in resp.text
            assert "Traceback" not in resp.text
