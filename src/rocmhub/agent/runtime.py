"""Small, capability-honest remote Agent runtime.

It accepts only the bounded PREPARE_MODEL_FOR_AMD workflow and delegates the
actual preparation to the existing Forge executor.
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import socket
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from rocmhub.forge.materializer import ModelMaterializer
from rocmhub.hardware.detector import SystemObserver
from rocmhub.server.orchestrator.models import JobEvent, JobStatus, JobType
from rocmhub.server.orchestrator.worker import execute_job


class AgentApiError(RuntimeError):
    def __init__(self, reason: str, *, transient: bool) -> None:
        super().__init__(reason)
        self.transient = transient


class AgentApi:
    def __init__(self, server: str, token: str, agent_id: Optional[str] = None) -> None:
        self.server = server.rstrip("/")
        self.token = token
        self.agent_id = agent_id

    def call(self, method: str, path: str, payload: Optional[Dict[str, Any]] = None) -> Any:
        data = json.dumps(payload).encode() if payload is not None else None
        headers = {"Authorization": f"Bearer {self.token}", "Accept": "application/json"}
        if self.agent_id:
            headers["X-ROCmHub-Agent-ID"] = self.agent_id
        if data:
            headers["Content-Type"] = "application/json"
        request = Request(f"{self.server}{path}", data=data, headers=headers, method=method)
        try:
            with urlopen(request, timeout=30) as response:
                raw = response.read().decode()
                return json.loads(raw) if raw else None
        except HTTPError as exc:
            raise AgentApiError(f"HTTP {exc.code}", transient=exc.code in {408, 425, 429} or exc.code >= 500) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise AgentApiError("Agent API is unavailable", transient=True) from exc


class ROCmHubAgent:
    def __init__(
        self,
        server: str,
        token: str,
        name: str,
        workspace: str,
        poll_interval: float = 2.0,
        max_concurrent_jobs: int = 1,
        json_output: bool = False,
    ) -> None:
        self.api = AgentApi(server, token)
        self.name = name
        self.workspace = Path(workspace).expanduser().resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.poll_interval = max(0.2, poll_interval)
        self.max_concurrent_jobs = max(1, max_concurrent_jobs)
        self.json_output = json_output
        self.agent_id: Optional[str] = None
        self._connection_was_lost = False
        self._active_job: Optional[str] = None
        self._active_attempt: Optional[int] = None

    def capabilities(self) -> Dict[str, Any]:
        report = SystemObserver().observe()
        env = report.environment
        gpus = report.gpus
        amd = [gpu for gpu in gpus if gpu.gpu_present and (gpu.gpu_vendor or "").upper() == "AMD"]
        disk = shutil.disk_usage(self.workspace)
        return {
            "os": platform.system(),
            "architecture": platform.machine(),
            "python_version": platform.python_version(),
            "rocm_detected": bool(env.rocm_version),
            "hip_detected": bool(env.hip_version),
            "pytorch_version": env.torch_version,
            "amd_gpu_count": len(amd),
            "gpu_names": [gpu.device_name for gpu in amd if gpu.device_name],
            "gfx_targets": [gpu.gfx_target for gpu in amd if gpu.gfx_target],
            "available_disk_gb": round(disk.free / (1024**3), 2),
            "capabilities": ["PREPARE_MODEL_FOR_AMD"],
        }

    def register(self) -> str:
        response = self.api.call(
            "POST",
            "/api/v1/agents/register",
            {"name": self.name, "hostname": socket.gethostname(), "capabilities": self.capabilities()},
        )
        self.agent_id = response["agent_id"]
        self.api.agent_id = self.agent_id
        self._log({"event": "registered", "agent_id": self.agent_id, "name": self.name})
        return self.agent_id

    def _log(self, event: Dict[str, Any]) -> None:
        if self.json_output:
            event.setdefault("type", event.get("event", "status"))
            print(json.dumps(event, sort_keys=True), flush=True)
        else:
            name = str(event.get("event", "status"))
            job_id = event.get("job_id")
            attempt = event.get("attempt")
            if name == "registered":
                text = str(event.get("agent_id", ""))
            elif name == "claimed":
                text = f"{job_id} attempt={attempt}"
            elif name in {"preparing", "completing", "checking_disk", "cache_miss", "downloading_model", "verifying_cache", "cache_hit", "cache_verified", "model_materialized", "cancelled"}:
                byte_count = event.get("materialized_bytes", event.get("estimated_bytes"))
                text = f"{job_id} bytes={byte_count}" if byte_count is not None else str(job_id or "")
            elif name in {"artifacts_ready", "uploading_artifacts"}:
                text = f"{job_id} count={event.get('count', 0)}"
            elif name == "completed":
                text = f"{job_id} status={event.get('status', 'unknown')}"
            elif name == "connection_error":
                context = f" {job_id} attempt={attempt}" if job_id else ""
                text = f"{event.get('reason', 'Agent API is unavailable')}{context}; retrying"
            elif name == "connection_restored":
                text = f"{job_id} attempt={attempt}" if job_id else "control plane"
            elif name == "terminal_error":
                text = f"{job_id}: {event.get('reason', 'request rejected')}"
            else:
                text = str(event.get("message", ""))
            print(f"[agent] {name}: {text}".rstrip(), flush=True)

    def _connection_error(self, exc: AgentApiError, job_id: Optional[str] = None, attempt: Optional[int] = None) -> None:
        already_lost = self._connection_was_lost
        self._connection_was_lost = True
        if already_lost:
            return
        self._log({
            "event": "connection_error",
            "reason": str(exc)[:120],
            "transient": exc.transient,
            "retrying": exc.transient,
            "job_id": job_id,
            "attempt": attempt,
        })

    def _note_connection_restored(self) -> None:
        if self._connection_was_lost:
            self._log({"event": "connection_restored", "job_id": self._active_job, "attempt": self._active_attempt})
            self._connection_was_lost = False

    def run_forever(self) -> None:
        while True:
            try:
                self.run_once()
                self._note_connection_restored()
            except AgentApiError as exc:
                if exc.transient:
                    self._connection_error(exc, self._active_job, self._active_attempt)
                else:
                    self._log({"event": "terminal_error", "reason": str(exc)[:120], "job_id": self._active_job})
            time.sleep(self.poll_interval)

    def run_once(self) -> bool:
        """Register if needed, heartbeat, and execute at most one claimed job."""
        if not self.agent_id:
            self.register()
        self.api.call("POST", f"/api/v1/agents/{self.agent_id}/heartbeat", {})
        claim = self.api.call("POST", f"/api/v1/agents/{self.agent_id}/claim", {})
        if not claim or not claim.get("job"):
            return False
        self._run_claim(claim["job"], claim["request_payload"])
        self._note_connection_restored()
        return True

    @staticmethod
    def _redact_local_paths(value: Any) -> Any:
        """Keep reproducibility metadata while never sending Agent-local filesystem paths."""
        sensitive_fields = {"build_dir", "output_dir", "weights_path"}
        if isinstance(value, dict):
            return {
                key: (None if key in sensitive_fields else ROCmHubAgent._redact_local_paths(item))
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [ROCmHubAgent._redact_local_paths(item) for item in value]
        return value

    @staticmethod
    def _prepare_artifact_upload(
        output_dir: Optional[str], forge_artifacts: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Prepare allowlisted small text outputs while preserving source and stored hashes."""
        if not output_dir:
            return {"files": {}, "source_artifacts": {}, "artifact_provenance": {}}
        allowed = {
            "build_manifest.json",
            "checksums.json",
            "model_config.json",
            "recipe.json",
            "run_inference.py",
            "runtime_config.json",
        }
        sources: Dict[str, bytes] = {}
        prepared: Dict[str, bytes] = {}
        parsed_json: Dict[str, Any] = {}
        for path in Path(output_dir).iterdir():
            if path.name not in allowed or not path.is_file() or path.stat().st_size > 262_144:
                continue
            try:
                source = path.read_bytes()
                text = source.decode("utf-8")
                source_sha = hashlib.sha256(source).hexdigest()
                expected_sha = (forge_artifacts or {}).get(path.name)
                if expected_sha is not None and expected_sha != source_sha:
                    raise ValueError(f"Forge artifact integrity check failed for {path.name}")
                sources[path.name] = source
                if path.suffix == ".json":
                    original = json.loads(text)
                    cleaned = ROCmHubAgent._redact_local_paths(original)
                    parsed_json[path.name] = cleaned
                    prepared[path.name] = source if cleaned == original else (
                        json.dumps(cleaned, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
                    ).encode("utf-8")
                else:
                    prepared[path.name] = source
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
        # Keep embedded integrity tables truthful after a privacy redaction.
        for json_name, key in (("build_manifest.json", "artifacts"), ("checksums.json", "files")):
            data = parsed_json.get(json_name)
            if not isinstance(data, dict) or not isinstance(data.get(key), dict):
                continue
            updated = dict(data)
            hashes = dict(updated[key])
            for name in hashes:
                if name in prepared and name != json_name:
                    hashes[name] = hashlib.sha256(prepared[name]).hexdigest()
            updated[key] = hashes
            if updated != data:
                parsed_json[json_name] = updated
                prepared[json_name] = (json.dumps(updated, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")

        files = {
            name: data.decode("utf-8")
            for name, data in prepared.items()
            if len(data) <= 262_144
        }
        source_artifacts = {
            name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            for name, data in sources.items() if name in files
        }
        provenance = {
            name: {
                "source_sha256": source_artifacts[name]["sha256"],
                "stored_sha256": hashlib.sha256(prepared[name]).hexdigest(),
                "sanitized": source_artifacts[name]["sha256"] != hashlib.sha256(prepared[name]).hexdigest(),
            }
            for name in files
        }
        return {
            "files": files,
            "source_artifacts": source_artifacts,
            "artifact_provenance": provenance,
            "artifacts": {name: item["stored_sha256"] for name, item in provenance.items()},
        }

    @staticmethod
    def _small_artifact_files(output_dir: Optional[str]) -> Dict[str, str]:
        """Compatibility helper returning just the allowlisted upload text."""
        return cast(Dict[str, str], ROCmHubAgent._prepare_artifact_upload(output_dir)["files"])

    def _run_claim(self, job: Dict[str, Any], payload: Dict[str, Any]) -> None:
        assert self.agent_id
        job_id = job["job_id"]
        attempt = int(job.get("attempt") or 1)
        self._active_job = job_id
        self._active_attempt = attempt
        if job["job_type"] != JobType.PREPARE_MODEL_FOR_AMD.value:
            self.api.call(
                "POST",
                f"/api/v1/agents/{self.agent_id}/jobs/{job_id}/fail",
                {"error_message": "Unsupported job type for this agent", "error_code": "UNSUPPORTED_AGENT_JOB", "attempt": attempt},
            )
            return
        self._log({"event": "claimed", "job_id": job_id, "attempt": attempt})

        stop_heartbeat = threading.Event()

        def keep_claim_alive() -> None:
            while not stop_heartbeat.wait(10.0):
                try:
                    self.api.call("POST", f"/api/v1/agents/{self.agent_id}/heartbeat", {})
                    self._note_connection_restored()
                except AgentApiError as exc:
                    if exc.transient:
                        self._connection_error(exc, job_id, attempt)
                        continue
                    self._log({"event": "terminal_error", "reason": str(exc)[:120], "job_id": job_id, "attempt": attempt})
                    return

        heartbeat_thread = threading.Thread(target=keep_claim_alive, name="rocmhub-agent-heartbeat", daemon=True)
        heartbeat_thread.start()

        def emit(
            phase: str,
            status: str,
            message: str,
            error_code: Optional[str] = None,
            details: Optional[Dict[str, Any]] = None,
        ) -> JobEvent:
            try:
                self.api.call(
                    "POST",
                    f"/api/v1/agents/{self.agent_id}/jobs/{job_id}/events",
                    {"phase": phase, "status": status, "message": message, "details": details, "attempt": attempt},
                )
            except AgentApiError as exc:
                if not exc.transient:
                    if cancel.is_set() and str(exc) == "HTTP 409":
                        return cast(JobEvent, None)
                    raise
                self._connection_error(exc, job_id, attempt)
            return cast(JobEvent, None)

        try:
            cancel = threading.Event()

            def progress(phase: str, details: Dict[str, object]) -> None:
                event_names = {
                    "CACHE_MISS": ("cache_miss", "MATERIALIZATION", "Cache miss; downloading immutable model snapshot"),
                    "DOWNLOADING_MODEL": ("downloading_model", "DOWNLOADING", "Downloading model files to the connected Agent"),
                    "VERIFYING_CACHE": ("verifying_cache", "VERIFYING", "Verifying materialized model files"),
                    "MODEL_MATERIALIZED": ("model_materialized", "PREPARING", "Model files materialized and verified"),
                    "CACHE_HIT": ("cache_hit", "VERIFYING", "Existing managed cache entry verified"),
                }
                mapped = event_names.get(phase)
                if not mapped:
                    return
                event_name, event_phase, message = mapped
                safe_details = {k: v for k, v in details.items() if k in {"estimated_bytes", "materialized_bytes"}}
                self._log({"event": event_name, "job_id": job_id, **safe_details})
                if phase == "CACHE_HIT":
                    self._log({"event": "cache_verified", "job_id": job_id, **safe_details})
                emit(event_phase, "RUNNING", message, None, safe_details)

            def cancellation_requested() -> bool:
                try:
                    state = self.api.call("GET", f"/api/v1/agents/{self.agent_id}/jobs/{job_id}/status")
                except AgentApiError as exc:
                    if exc.transient:
                        self._connection_error(exc, job_id, attempt)
                        return False
                    raise
                if state.get("status") == JobStatus.CANCELLED.value:
                    cancel.set()
                    self._log({"event": "cancelled", "job_id": job_id})
                    return True
                return False

            mode = payload.get("materialization_mode", "METADATA_ONLY")
            materializer = None
            if mode == "FULL_WEIGHTS" and payload.get("weights_consent") is True:
                self._log({"event": "checking_disk", "job_id": job_id})
                emit("CHECKING_DISK", "RUNNING", "Checking Agent-local storage before model download", None, None)
                materializer = ModelMaterializer(
                    cache_dir=self.workspace / "cache",
                    managed_cache=True,
                    progress_callback=progress,
                    cancellation_check=cancellation_requested,
                )
            else:
                self._log({"event": "preparing", "job_id": job_id})
            local_payload = {
                **payload,
                "output_dir": str(self.workspace / job_id),
                "allow_full_weights": False,
                "execute_inference": False,
            }
            if materializer is None:
                status, domain, output_dir, result, error, error_code, revision = execute_job(
                    job_id, JobType.PREPARE_MODEL_FOR_AMD, local_payload, cancel, emit
                )
            else:
                status, domain, output_dir, result, error, error_code, revision = execute_job(
                    job_id, JobType.PREPARE_MODEL_FOR_AMD, local_payload, cancel, emit, materializer
                )
            if status == JobStatus.SUCCEEDED:
                declared_artifacts = (result or {}).get("artifacts", {})
                artifacts = self._prepare_artifact_upload(
                    output_dir, declared_artifacts if isinstance(declared_artifacts, dict) else None
                )
                files = artifacts["files"]
                count = len(files)
                self._log({"event": "artifacts_ready", "job_id": job_id, "attempt": attempt, "count": count})
                artifact_manifest = self._redact_local_paths(result)
                enriched = {
                    "preparation": "PREPARE_MODEL_FOR_AMD",
                    "model_id": payload.get("model_id"),
                    "revision": revision,
                    "weights": "MATERIALIZED" if (result or {}).get("materialization", {}).get("has_weights") else "NOT_DOWNLOADED",
                    "materialization": (result or {}).get("materialization", {}),
                    "materialization_mode": payload.get("materialization_mode", "METADATA_ONLY"),
                    "amd_validated": False,
                    "target_gfx": payload.get("target_gfx"),
                    "runtime": payload.get("runtime"),
                    "artifacts": artifacts["artifacts"],
                    "source_artifacts": artifacts["source_artifacts"],
                    "artifact_provenance": artifacts["artifact_provenance"],
                    "artifact_manifest": artifact_manifest,
                    "agent_observation": {
                        "physical_amd_execution": False,
                        "reason": "Preparation completed without AMD inference or validation.",
                    },
                }
                completion = {
                    "domain_status": domain or "CONFIG_ONLY",
                    "result": enriched,
                    "revision": revision,
                    "artifact_files": files,
                    "attempt": attempt,
                }
                completion_id = hashlib.sha256(
                    json.dumps(completion, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
                ).hexdigest()
                completion["completion_id"] = completion_id
                self._log({"event": "uploading_artifacts", "job_id": job_id, "attempt": attempt, "count": count})
                self._log({"event": "completing", "job_id": job_id, "attempt": attempt})
                while True:
                    try:
                        self.api.call(
                            "POST",
                            f"/api/v1/agents/{self.agent_id}/jobs/{job_id}/complete",
                            completion,
                        )
                        break
                    except AgentApiError as exc:
                        if not exc.transient:
                            raise
                        self._connection_error(exc, job_id, attempt)
                        stop_heartbeat.wait(self.poll_interval)
                self._note_connection_restored()
                self._log({"event": "completed", "job_id": job_id, "attempt": attempt, "status": domain or "CONFIG_ONLY"})
            else:
                if status != JobStatus.CANCELLED:
                    self.api.call(
                        "POST",
                        f"/api/v1/agents/{self.agent_id}/jobs/{job_id}/fail",
                        {"error_message": "Agent preparation failed; inspect local Agent diagnostics.", "error_code": error_code or "PREPARATION_FAILED", "attempt": attempt},
                    )
                self._log({"event": "completed", "job_id": job_id, "attempt": attempt, "status": status.value})
        except AgentApiError:
            raise
        except Exception:
            self.api.call(
                "POST",
                f"/api/v1/agents/{self.agent_id}/jobs/{job_id}/fail",
                {"error_message": "Agent preparation failed", "error_code": "AGENT_EXECUTION_FAILED", "attempt": attempt},
            )
            self._log({"event": "completed", "job_id": job_id, "attempt": attempt, "status": "FAILED"})
        finally:
            stop_heartbeat.set()
            heartbeat_thread.join(timeout=1.0)
            self._active_job = None
            self._active_attempt = None
