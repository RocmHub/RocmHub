"""External Agent API and ownership tests."""

from __future__ import annotations

import hashlib

from fastapi.testclient import TestClient

from rocmhub.server.app import create_app
from rocmhub.server.config import ServerConfig

TOKEN = "test-agent-token"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}


def make_client(tmp_path: object) -> TestClient:
    cfg = ServerConfig(db_path=tmp_path / "agents.db", agent_token_hashes=[hashlib.sha256(TOKEN.encode()).hexdigest()])  # type: ignore[operator]
    return TestClient(create_app(cfg))


def payload() -> dict:
    return {
        "name": "mac-dev-agent",
        "hostname": "mac-mini",
        "capabilities": {
            "os": "Darwin",
            "architecture": "arm64",
            "python_version": "3.9",
            "rocm_detected": False,
            "hip_detected": False,
            "amd_gpu_count": 0,
            "gpu_names": [],
            "gfx_targets": [],
            "capabilities": ["PREPARE_MODEL_FOR_AMD"],
        },
    }


def create_prepare(client: TestClient) -> str:
    response = client.post(
        "/api/v1/jobs",
        json={
            "job_type": "PREPARE_MODEL_FOR_AMD",
            "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
            "target_gpu": "Radeon RX 7900 XTX",
            "target_gfx": "gfx1100",
            "precision": "fp16",
            "runtime": "pytorch_transformers_hip",
            "allow_full_weights": False,
        },
    )
    assert response.status_code == 202
    return response.json()["job_id"]


def test_agent_registration_requires_valid_token(tmp_path: object) -> None:
    with make_client(tmp_path) as client:
        assert client.post("/api/v1/agents/register", json=payload()).status_code == 401
        assert (
            client.post("/api/v1/agents/register", headers={"Authorization": "Bearer bad"}, json=payload()).status_code
            == 401
        )
        response = client.post("/api/v1/agents/register", headers=HEADERS, json=payload())
        assert response.status_code == 200
        assert response.json()["status"] == "ONLINE"
        assert TOKEN not in response.text


def test_mac_agent_claims_config_only_preparation_once(tmp_path: object) -> None:
    with make_client(tmp_path) as client:
        job_id = create_prepare(client)
        registered = client.post("/api/v1/agents/register", headers=HEADERS, json=payload()).json()
        agent_id = registered["agent_id"]
        claimed = client.post(f"/api/v1/agents/{agent_id}/claim", headers=HEADERS)
        assert claimed.status_code == 200 and claimed.json()["job"]["job_id"] == job_id
        assert claimed.json()["job"]["status"] == "RUNNING"
        assert client.post(f"/api/v1/agents/{agent_id}/claim", headers=HEADERS).json()["job"] is None


def test_agent_completion_persists_artifact_result(tmp_path: object) -> None:
    with make_client(tmp_path) as client:
        job_id = create_prepare(client)
        agent_id = client.post("/api/v1/agents/register", headers=HEADERS, json=payload()).json()["agent_id"]
        client.post(f"/api/v1/agents/{agent_id}/claim", headers=HEADERS)
        done = client.post(
            f"/api/v1/agents/{agent_id}/jobs/{job_id}/complete",
            headers=HEADERS,
            json={
                "domain_status": "CONFIG_ONLY",
                "result": {"weights": "NOT_DOWNLOADED", "artifacts": {"runtime_config.json": "abc"}},
                "revision": "deadbeef",
            },
        )
        assert done.status_code == 204
        result = client.get(f"/api/v1/jobs/{job_id}/result").json()
        assert result["job_status"] == "SUCCEEDED"
        assert result["result"]["artifacts"]["runtime_config.json"] == "abc"


def test_agent_heartbeat_reconnect_and_token_revocation(tmp_path: object) -> None:
    with make_client(tmp_path) as client:
        first = client.post("/api/v1/agents/register", headers=HEADERS, json=payload()).json()
        agent_id = first["agent_id"]
        reconnect_headers = {**HEADERS, "X-ROCmHub-Agent-ID": agent_id}
        reconnected = client.post("/api/v1/agents/register", headers=reconnect_headers, json=payload()).json()
        assert reconnected["agent_id"] == agent_id
        assert client.post(f"/api/v1/agents/{agent_id}/heartbeat", headers=HEADERS).status_code == 200
        assert client.post("/api/v1/agents/revoke", headers=HEADERS).status_code == 204
        assert client.post(f"/api/v1/agents/{agent_id}/heartbeat", headers=HEADERS).status_code == 401


def test_cancelled_and_stale_agent_jobs_are_not_executed(tmp_path: object) -> None:
    with make_client(tmp_path) as client:
        cancelled_id = create_prepare(client)
        assert client.post(f"/api/v1/jobs/{cancelled_id}/cancel").json()["status"] == "CANCELLED"
        agent_id = client.post("/api/v1/agents/register", headers=HEADERS, json=payload()).json()["agent_id"]
        assert client.post(f"/api/v1/agents/{agent_id}/claim", headers=HEADERS).json()["job"] is None
        active_id = create_prepare(client)
        assert client.post(f"/api/v1/agents/{agent_id}/claim", headers=HEADERS).json()["job"]["job_id"] == active_id
        manager = client.app.state.job_manager
        released = manager.db.release_stale_claims(timeout_seconds=0)
        assert active_id in released
        assert client.get(f"/api/v1/jobs/{active_id}").json()["status"] == "QUEUED"


def test_server_restart_releases_remote_ownership_and_mac_rejects_gpu_job(tmp_path: object) -> None:
    with make_client(tmp_path) as client:
        # The public contract deliberately has no AMD execution job type yet.
        assert (
            client.post(
                "/api/v1/jobs", json={"job_type": "AMD_EXECUTION", "model_id": "Qwen/Qwen2.5-0.5B-Instruct"}
            ).status_code
            == 422
        )
        job_id = create_prepare(client)
        agent_id = client.post("/api/v1/agents/register", headers=HEADERS, json=payload()).json()["agent_id"]
        assert client.post(f"/api/v1/agents/{agent_id}/claim", headers=HEADERS).status_code == 200
        client.app.state.job_manager.db.mark_interrupted_jobs_as_failed("test restart")
        # Remote work is released, never silently retained as RUNNING or falsely completed.
        assert client.get(f"/api/v1/jobs/{job_id}").json()["status"] == "QUEUED"
