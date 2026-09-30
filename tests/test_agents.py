"""External Agent API and ownership tests."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from rocmhub.server.app import create_app
from rocmhub.server.config import ServerConfig
from rocmhub.server.orchestrator.models import JobStatus

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
            "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
            "target_gpu": "Radeon RX 7900 XTX",
            "target_gfx": "gfx1100",
            "precision": "fp16",
            "runtime": "pytorch_transformers_hip",
            "allow_full_weights": False,
        },
    )
    assert response.status_code == 202
    return response.json()["job_id"]


def with_completion_id(data: dict) -> dict:
    completion = dict(data)
    completion["completion_id"] = hashlib.sha256(
        json.dumps(completion, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return completion


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


def test_full_materialization_contract_persists_explicit_consent(tmp_path: object) -> None:
    with make_client(tmp_path) as client:
        body = {
            "job_type": "PREPARE_MODEL_FOR_AMD",
            "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
            "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
            "target_gpu": "Radeon RX 7900 XTX",
            "target_gfx": "gfx1100",
            "precision": "fp16",
            "runtime": "pytorch_transformers_hip",
            "materialization_mode": "FULL_WEIGHTS",
            "weights_consent": True,
            "expected_capabilities": ["PREPARE_MODEL_FOR_AMD"],
            "cache_policy": "REUSE",
        }
        assert client.post("/api/v1/jobs", json={**body, "weights_consent": False}).status_code == 422
        created = client.post("/api/v1/jobs", json=body)
        assert created.status_code == 202
        agent_id = client.post("/api/v1/agents/register", headers=HEADERS, json=payload()).json()["agent_id"]
        claim = client.post(f"/api/v1/agents/{agent_id}/claim", headers=HEADERS).json()
        assert claim["request_payload"]["materialization_mode"] == "FULL_WEIGHTS"
        assert claim["request_payload"]["weights_consent"] is True
        assert claim["request_payload"]["revision"] == body["revision"]


def test_remote_agent_cancellation_is_visible_to_claim_owner(tmp_path: object) -> None:
    with make_client(tmp_path) as client:
        job_id = create_prepare(client)
        agent_id = client.post("/api/v1/agents/register", headers=HEADERS, json=payload()).json()["agent_id"]
        client.post(f"/api/v1/agents/{agent_id}/claim", headers=HEADERS)
        assert client.get(f"/api/v1/agents/{agent_id}/jobs/{job_id}/status", headers=HEADERS).json() == {"status": "RUNNING"}
        cancelled = client.post(f"/api/v1/jobs/{job_id}/cancel")
        assert cancelled.json()["status"] == "CANCELLED"
        assert client.get(f"/api/v1/agents/{agent_id}/jobs/{job_id}/status", headers=HEADERS).json() == {"status": "CANCELLED"}


def test_server_artifact_allowlist_rejects_model_weight_upload(tmp_path: object) -> None:
    with make_client(tmp_path) as client:
        job_id = create_prepare(client)
        agent_id = client.post("/api/v1/agents/register", headers=HEADERS, json=payload()).json()["agent_id"]
        client.post(f"/api/v1/agents/{agent_id}/claim", headers=HEADERS)
        response = client.post(
            f"/api/v1/agents/{agent_id}/jobs/{job_id}/complete",
            headers=HEADERS,
            json={"domain_status": "CONFIG_ONLY", "revision": "7ae557604adf67be50417f59c2c2f167def9a775", "result": {"weights": "NOT_DOWNLOADED"}, "artifact_files": {"model.safetensors": "not-a-weight"}},
        )
        assert response.status_code == 400


def test_control_plane_rejects_forged_prepared_for_metadata_only_job(tmp_path: object) -> None:
    with make_client(tmp_path) as client:
        job_id = create_prepare(client)
        agent_id = client.post("/api/v1/agents/register", headers=HEADERS, json=payload()).json()["agent_id"]
        client.post(f"/api/v1/agents/{agent_id}/claim", headers=HEADERS)
        response = client.post(
            f"/api/v1/agents/{agent_id}/jobs/{job_id}/complete",
            headers=HEADERS,
            json={
                "domain_status": "PREPARED",
                "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
                "result": {"weights": "MATERIALIZED", "amd_validated": False},
            },
        )
        assert response.status_code == 400
        assert client.get(f"/api/v1/jobs/{job_id}").json()["status"] == "RUNNING"


def test_control_plane_rejects_agent_revision_drift(tmp_path: object) -> None:
    with make_client(tmp_path) as client:
        job = client.post(
            "/api/v1/jobs",
            json={
                "job_type": "PREPARE_MODEL_FOR_AMD",
                "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
                "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
                "materialization_mode": "FULL_WEIGHTS",
                "weights_consent": True,
            },
        ).json()
        job_id = job["job_id"]
        agent_id = client.post("/api/v1/agents/register", headers=HEADERS, json=payload()).json()["agent_id"]
        client.post(f"/api/v1/agents/{agent_id}/claim", headers=HEADERS)
        response = client.post(
            f"/api/v1/agents/{agent_id}/jobs/{job_id}/complete",
            headers=HEADERS,
            json={
                "domain_status": "PREPARED",
                "revision": "a" * 40,
                "result": {},
            },
        )
        assert response.status_code == 400


def test_control_plane_disallows_execution_domain_status_in_milestone_2b(tmp_path: object) -> None:
    with make_client(tmp_path) as client:
        job_id = create_prepare(client)
        agent_id = client.post("/api/v1/agents/register", headers=HEADERS, json=payload()).json()["agent_id"]
        client.post(f"/api/v1/agents/{agent_id}/claim", headers=HEADERS)
        response = client.post(
            f"/api/v1/agents/{agent_id}/jobs/{job_id}/complete",
            headers=HEADERS,
            json={
                "domain_status": "EXECUTED",
                "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
                "result": {"amd_validated": True},
            },
        )
        assert response.status_code == 400


def test_control_plane_rejects_agent_local_cache_paths_in_results(tmp_path: object) -> None:
    with make_client(tmp_path) as client:
        job = client.post(
            "/api/v1/jobs",
            json={
                "job_type": "PREPARE_MODEL_FOR_AMD",
                "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
                "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
                "materialization_mode": "FULL_WEIGHTS",
                "weights_consent": True,
            },
        ).json()
        job_id = job["job_id"]
        agent_id = client.post("/api/v1/agents/register", headers=HEADERS, json=payload()).json()["agent_id"]
        client.post(f"/api/v1/agents/{agent_id}/claim", headers=HEADERS)
        response = client.post(
            f"/api/v1/agents/{agent_id}/jobs/{job_id}/complete",
            headers=HEADERS,
            json={
                "domain_status": "PREPARED",
                "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
                "result": {
                    "materialization": {
                        "mode": "FULL_WEIGHTS",
                        "has_weights": True,
                        "cache_status": "MISS_DOWNLOADED",
                        "integrity": "FULL_SHA256_MANIFEST_VERIFIED",
                    },
                    "agent_observation": {"physical_amd_execution": False},
                    "amd_validated": False,
                    "cache_path": "/Users/agent/private/cache/model",
                },
            },
        )
        assert response.status_code == 400
        assert "path" in response.json()["detail"].lower()


def test_agent_completion_persists_artifact_result(tmp_path: object) -> None:
    with make_client(tmp_path) as client:
        client.app.state.config.artifact_storage_dir = tmp_path / "agent-artifacts"  # type: ignore[operator]
        job_id = create_prepare(client)
        agent_id = client.post("/api/v1/agents/register", headers=HEADERS, json=payload()).json()["agent_id"]
        client.post(f"/api/v1/agents/{agent_id}/claim", headers=HEADERS)
        done = client.post(
            f"/api/v1/agents/{agent_id}/jobs/{job_id}/complete",
            headers=HEADERS,
            json={
                "domain_status": "CONFIG_ONLY",
                "result": {"weights": "NOT_DOWNLOADED", "artifacts": {"runtime_config.json": "abc"}},
                "output_dir": "/private/agent-workspace/job",
                "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
                "artifact_files": {"runtime_config.json": "{\"device\": \"cpu\"}\n"},
            },
        )
        assert done.status_code == 204
        result = client.get(f"/api/v1/jobs/{job_id}/result").json()
        assert result["job_status"] == "SUCCEEDED"
        assert "output_dir" not in result
        assert result["result"]["executor_provenance"]["source"] == "agent_reported"
        assert "agent_id" not in result["result"]["executor_provenance"]
        stored = result["result"]["server_artifacts"]
        assert stored[0]["name"] == "runtime_config.json"
        downloaded = client.get(stored[0]["download_path"])
        assert downloaded.status_code == 200
        assert downloaded.text == "{\"device\": \"cpu\"}\n"
        downloaded_sha = hashlib.sha256(downloaded.content).hexdigest()
        assert stored[0]["sha256"] == downloaded_sha
        assert result["result"]["artifacts"]["runtime_config.json"] == downloaded_sha


def test_downloaded_agent_artifacts_do_not_publish_local_identity_or_paths(tmp_path: object) -> None:
    with make_client(tmp_path) as client:
        client.app.state.config.artifact_storage_dir = tmp_path / "agent-artifacts"  # type: ignore[operator]
        job_id = create_prepare(client)
        agent_id = client.post("/api/v1/agents/register", headers=HEADERS, json=payload()).json()["agent_id"]
        client.post(f"/api/v1/agents/{agent_id}/claim", headers=HEADERS)
        completed = client.post(
            f"/api/v1/agents/{agent_id}/jobs/{job_id}/complete",
            headers=HEADERS,
            json={
                "domain_status": "CONFIG_ONLY",
                "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
                "result": {"weights": "NOT_DOWNLOADED", "amd_validated": False},
                "artifact_files": {
                    "runtime_config.json": json.dumps(
                        {
                            "device": "cpu",
                            "cache_path": "/Users/private-user/.cache/rocmhub",
                            "hostname": "mac-mini",
                        }
                    )
                },
            },
        )
        assert completed.status_code == 204
        result = client.get(f"/api/v1/jobs/{job_id}/result").json()
        download_path = result["result"]["server_artifacts"][0]["download_path"]
        downloaded = client.get(download_path)
        assert downloaded.status_code == 200
        assert "private-user" not in downloaded.text
        assert "mac-mini" not in downloaded.text
        assert "cache_path" not in downloaded.text


def test_agent_artifact_sanitization_provenance_and_download_hashes(tmp_path: object) -> None:
    from rocmhub.agent.runtime import ROCmHubAgent

    source_dir = Path(tmp_path) / "forge-output"  # type: ignore[arg-type]
    source_dir.mkdir()
    source_files = {
        "runtime_config.json": b'{\n  "device": "cuda",\r\n  "weights_path": "/private/agent/cache/model"\r\n}\r\n',
        "model_config.json": b'{ "model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"] }\n',
        "recipe.json": b'{"runtime":"pytorch_transformers_hip"}\n',
        "run_inference.py": b"#!/usr/bin/env python3\r\nprint('ready')\r\n",
    }
    source_files["build_manifest.json"] = json.dumps(
        {"artifacts": {"runtime_config.json": hashlib.sha256(source_files["runtime_config.json"]).hexdigest()}}
    ).encode() + b"\n"
    source_files["checksums.json"] = json.dumps(
        {"files": {"runtime_config.json": hashlib.sha256(source_files["runtime_config.json"]).hexdigest()}}
    ).encode() + b"\n"
    for name, data in source_files.items():
        (source_dir / name).write_bytes(data)
    forge_hashes = {
        name: hashlib.sha256(data).hexdigest()
        for name, data in source_files.items()
        if name in {"runtime_config.json", "model_config.json", "recipe.json", "run_inference.py"}
    }
    upload = ROCmHubAgent._prepare_artifact_upload(str(source_dir), forge_hashes)
    stored_config_sha = hashlib.sha256(upload["files"]["runtime_config.json"].encode()).hexdigest()
    assert json.loads(upload["files"]["build_manifest.json"])["artifacts"]["runtime_config.json"] == stored_config_sha
    assert json.loads(upload["files"]["checksums.json"])["files"]["runtime_config.json"] == stored_config_sha

    with make_client(tmp_path) as client:
        client.app.state.config.artifact_storage_dir = Path(tmp_path) / "agent-artifacts"  # type: ignore[operator]
        job_id = create_prepare(client)
        agent_id = client.post("/api/v1/agents/register", headers=HEADERS, json=payload()).json()["agent_id"]
        claim = client.post(f"/api/v1/agents/{agent_id}/claim", headers=HEADERS).json()["job"]
        done = client.post(
            f"/api/v1/agents/{agent_id}/jobs/{job_id}/complete",
            headers=HEADERS,
            json=with_completion_id({
                "domain_status": "CONFIG_ONLY",
                "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
                "attempt": claim["attempt"],
                "result": {
                    "artifacts": upload["artifacts"],
                    "source_artifacts": upload["source_artifacts"],
                    "artifact_provenance": upload["artifact_provenance"],
                },
                "artifact_files": upload["files"],
            }),
        )
        assert done.status_code == 204
        result = client.get(f"/api/v1/jobs/{job_id}/result").json()["result"]
        server_artifacts = {item["name"]: item for item in result["server_artifacts"]}
        for name, metadata in server_artifacts.items():
            downloaded = client.get(metadata["download_path"])
            assert downloaded.status_code == 200
            assert hashlib.sha256(downloaded.content).hexdigest() == metadata["sha256"]
            assert hashlib.sha256(downloaded.content).hexdigest() == result["artifacts"][name]
            assert downloaded.content.decode("utf-8") == upload["files"][name]
        assert result["artifact_provenance"]["runtime_config.json"]["source_sha256"] == hashlib.sha256(
            source_files["runtime_config.json"]
        ).hexdigest()
        assert result["artifact_provenance"]["runtime_config.json"]["stored_sha256"] == server_artifacts[
            "runtime_config.json"
        ]["sha256"]
        assert result["artifact_provenance"]["runtime_config.json"]["sanitized"] is True
        assert result["artifact_provenance"]["model_config.json"]["sanitized"] is False
        assert server_artifacts["run_inference.py"]["sha256"] == hashlib.sha256(
            source_files["run_inference.py"]
        ).hexdigest()
        assert b"/private/agent" not in client.get(server_artifacts["runtime_config.json"]["download_path"]).content


def test_stale_agent_attempt_cannot_complete_reclaimed_job(tmp_path: object) -> None:
    with make_client(tmp_path) as client:
        client.app.state.config.artifact_storage_dir = tmp_path / "agent-artifacts"  # type: ignore[operator]
        job_id = create_prepare(client)
        agent_id = client.post("/api/v1/agents/register", headers=HEADERS, json=payload()).json()["agent_id"]
        first_claim = client.post(f"/api/v1/agents/{agent_id}/claim", headers=HEADERS).json()["job"]
        assert first_claim["attempt"] == 1
        assert job_id in client.app.state.job_manager.db.release_stale_claims(timeout_seconds=0)
        second_claim = client.post(f"/api/v1/agents/{agent_id}/claim", headers=HEADERS).json()["job"]
        assert second_claim["attempt"] == 2
        stale = client.post(
            f"/api/v1/agents/{agent_id}/jobs/{job_id}/complete",
            headers=HEADERS,
            json=with_completion_id({
                "domain_status": "CONFIG_ONLY",
                "attempt": first_claim["attempt"],
                "result": {},
                "artifact_files": {"runtime_config.json": "{}\n"},
            }),
        )
        assert stale.status_code == 409
        assert client.get(f"/api/v1/jobs/{job_id}").json()["status"] == "RUNNING"
        assert not (Path(tmp_path) / "agent-artifacts" / job_id / "runtime_config.json").exists()  # type: ignore[arg-type]


def test_agent_completion_replay_is_idempotent(tmp_path: object) -> None:
    with make_client(tmp_path) as client:
        client.app.state.config.artifact_storage_dir = tmp_path / "agent-artifacts"  # type: ignore[operator]
        job_id = create_prepare(client)
        agent_id = client.post("/api/v1/agents/register", headers=HEADERS, json=payload()).json()["agent_id"]
        claim = client.post(f"/api/v1/agents/{agent_id}/claim", headers=HEADERS).json()["job"]
        request = with_completion_id({
            "domain_status": "CONFIG_ONLY",
            "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
            "attempt": claim["attempt"],
            "result": {"weights": "NOT_DOWNLOADED"},
            "artifact_files": {"runtime_config.json": "{}\n"},
        })
        endpoint = f"/api/v1/agents/{agent_id}/jobs/{job_id}/complete"
        assert client.post(endpoint, headers=HEADERS, json=request).status_code == 204
        assert client.post(endpoint, headers=HEADERS, json=request).status_code == 204
        altered_replay = dict(request)
        altered_replay["domain_status"] = "PREPARED"
        assert client.post(endpoint, headers=HEADERS, json=altered_replay).status_code == 400
        assert client.get(f"/api/v1/jobs/{job_id}").json()["status"] == "SUCCEEDED"


def test_agent_artifact_upload_rejects_path_escape(tmp_path: object) -> None:
    with make_client(tmp_path) as client:
        job_id = create_prepare(client)
        agent_id = client.post("/api/v1/agents/register", headers=HEADERS, json=payload()).json()["agent_id"]
        client.post(f"/api/v1/agents/{agent_id}/claim", headers=HEADERS)
        rejected = client.post(
            f"/api/v1/agents/{agent_id}/jobs/{job_id}/complete",
            headers=HEADERS,
            json={
                "domain_status": "CONFIG_ONLY",
                "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
                "result": {"weights": "NOT_DOWNLOADED"},
                "artifact_files": {"../private.txt": "nope"},
            },
        )
        assert rejected.status_code == 400


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


def test_agent_discovery_is_current_only_and_does_not_expose_machine_identity(tmp_path: object) -> None:
    with make_client(tmp_path) as client:
        registered = client.post("/api/v1/agents/register", headers=HEADERS, json=payload()).json()
        agent_id = registered["agent_id"]
        assert "name" not in registered and "hostname" not in registered

        presence = client.get("/api/v1/agents")
        assert presence.status_code == 200
        assert presence.json() == [
            {
                "label": "Preparation Agent",
                "status": "ONLINE",
                "has_amd_gpu": False,
                "gpu_names": [],
                "rocm_detected": False,
            }
        ]
        assert "mac-dev-agent" not in presence.text
        assert "mac-mini" not in presence.text
        assert agent_id not in presence.text

        conn = client.app.state.job_manager.db._get_connection()
        try:
            stored = conn.execute("SELECT name, hostname FROM agents WHERE agent_id=?", (agent_id,)).fetchone()
            assert stored["name"] == "Preparation Agent"
            assert stored["hostname"] == ""
            conn.execute("UPDATE agents SET name='mac-dev-agent',hostname='mac-mini' WHERE agent_id=?", (agent_id,))
            conn.commit()
        finally:
            conn.close()
        assert client.post(f"/api/v1/agents/{agent_id}/heartbeat", headers=HEADERS).status_code == 200
        conn = client.app.state.job_manager.db._get_connection()
        try:
            scrubbed = conn.execute("SELECT name, hostname FROM agents WHERE agent_id=?", (agent_id,)).fetchone()
            assert scrubbed["name"] == "Preparation Agent"
            assert scrubbed["hostname"] == ""
            old = (datetime.now(timezone.utc) - timedelta(minutes=2)).isoformat()
            conn.execute("UPDATE agents SET last_seen=? WHERE agent_id=?", (old, agent_id))
            conn.commit()
        finally:
            conn.close()
        assert client.get("/api/v1/agents").json() == []


def test_reconnect_prunes_expired_presence_without_removing_active_claims(tmp_path: object) -> None:
    with make_client(tmp_path) as client:
        manager = client.app.state.job_manager
        client.app.state.config.agent_retention_seconds = 60
        old_agent = client.post("/api/v1/agents/register", headers=HEADERS, json=payload()).json()["agent_id"]
        conn = manager.db._get_connection()
        try:
            expired = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
            conn.execute("UPDATE agents SET last_seen=? WHERE agent_id=?", (expired, old_agent))
            conn.commit()
        finally:
            conn.close()

        current_agent = client.post("/api/v1/agents/register", headers=HEADERS, json=payload()).json()["agent_id"]
        assert old_agent != current_agent
        assert [agent.agent_id for agent in manager.db.list_agents()] == [current_agent]
        assert len(client.get("/api/v1/agents").json()) == 1

        active_job = create_prepare(client)
        client.post(f"/api/v1/agents/{current_agent}/claim", headers=HEADERS)
        conn = manager.db._get_connection()
        try:
            expired = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
            conn.execute("UPDATE agents SET last_seen=? WHERE agent_id=?", (expired, current_agent))
            conn.commit()
        finally:
            conn.close()
        assert manager.db.prune_stale_agents(60) == 0
        assert manager.db.get_agent(current_agent) is not None
        assert client.get(f"/api/v1/jobs/{active_job}").json()["status"] == "RUNNING"


def test_completed_job_evidence_and_activity_survive_agent_pruning(tmp_path: object) -> None:
    with make_client(tmp_path) as client:
        manager = client.app.state.job_manager
        client.app.state.config.artifact_storage_dir = tmp_path / "agent-artifacts"  # type: ignore[operator]
        job_id = create_prepare(client)
        agent_id = client.post("/api/v1/agents/register", headers=HEADERS, json=payload()).json()["agent_id"]
        client.post(f"/api/v1/agents/{agent_id}/claim", headers=HEADERS)
        completed = client.post(
            f"/api/v1/agents/{agent_id}/jobs/{job_id}/complete",
            headers=HEADERS,
            json={
                "domain_status": "CONFIG_ONLY",
                "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
                "result": {
                    "weights": "NOT_DOWNLOADED",
                    "amd_validated": False,
                    "agent_observation": {"physical_amd_execution": False},
                },
            },
        )
        assert completed.status_code == 204
        before = client.get(f"/api/v1/jobs/{job_id}/result").json()
        assert before["result"]["executor_provenance"]["os"] == "Darwin"
        assert before["result"]["agent_observation"]["physical_amd_execution"] is False
        assert "agent_id" not in before
        assert agent_id not in json.dumps(before)

        conn = manager.db._get_connection()
        try:
            expired = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
            conn.execute("UPDATE agents SET last_seen=? WHERE agent_id=?", (expired, agent_id))
            conn.commit()
        finally:
            conn.close()
        assert manager.db.prune_stale_agents(60) == 1

        after = client.get(f"/api/v1/jobs/{job_id}/result").json()
        assert after["result"]["executor_provenance"] == before["result"]["executor_provenance"]
        assert len(manager.db.get_events(job_id)) >= 3
        assert client.get(f"/api/v1/jobs/{job_id}").json()["status"] == "SUCCEEDED"


def test_public_job_and_health_contracts_redact_paths_and_agent_ids(tmp_path: object) -> None:
    with make_client(tmp_path) as client:
        job_id = create_prepare(client)
        manager = client.app.state.job_manager
        agent_id = client.post("/api/v1/agents/register", headers=HEADERS, json=payload()).json()["agent_id"]
        client.post(f"/api/v1/agents/{agent_id}/claim", headers=HEADERS)
        manager.emit_event(
            job_id,
            "PRIVACY_TEST",
            "RUNNING",
            "Local workspace: /Users/private-user/workspace",
            details={"output_dir": "/Users/private-user/workspace", "agent_id": agent_id},
        )
        manager.db.update_job_status(
            job_id,
            status=JobStatus.RUNNING,
            output_dir="/Users/private-user/workspace",
            result_payload={
                "output_dir": "/Users/private-user/workspace",
                "/Users/private-user/private-manifest.json": "sha256-value",
                "nested": {"message": "Cache at /home/private-user/.cache/models", "hostname": "developer-mac"},
                "amd_validated": False,
            },
        )

        job = client.get(f"/api/v1/jobs/{job_id}")
        listing = client.get("/api/v1/jobs?limit=10")
        result = client.get(f"/api/v1/jobs/{job_id}/result")
        exposed = "".join((job.text, listing.text, result.text, json.dumps([e.model_dump() for e in manager.db.get_events(job_id)])))
        assert "output_dir" not in job.json()
        assert "agent_id" not in job.json()
        assert "output_dir" not in result.json()
        assert "private-user" not in exposed
        assert "/Users/" not in exposed
        assert "/home/" not in exposed
        assert "private-manifest.json" not in exposed
        assert agent_id not in exposed

        manager._active_directory_locks.add("/Users/private-user/workspace")
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["orchestrator"]["active_directory_lock_count"] == 1
        assert "active_directory_locks" not in health.json()["orchestrator"]
        assert "/Users/" not in health.text


def test_terminal_jobs_do_not_retain_local_filesystem_paths(tmp_path: object) -> None:
    with make_client(tmp_path) as client:
        job_id = create_prepare(client)
        manager = client.app.state.job_manager
        manager.db.update_job_status(
            job_id,
            status=JobStatus.SUCCEEDED,
            output_dir="/Users/private-user/workspace",
            result_payload={"output_dir": "/Users/private-user/workspace", "status": "CONFIG_ONLY"},
        )
        full_job = manager.db.get_job_full(job_id)
        assert full_job is not None
        assert full_job["output_dir"] is None
        assert "output_dir" not in full_job["request_payload"]
        assert "output_dir" not in full_job["result_payload"]
