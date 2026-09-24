"""Small, capability-honest remote Agent runtime.

It accepts only the bounded PREPARE_MODEL_FOR_AMD workflow and delegates the
actual preparation to the existing Forge executor.
"""

from __future__ import annotations

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

from rocmhub.hardware.detector import SystemObserver
from rocmhub.server.orchestrator.models import JobEvent, JobStatus, JobType
from rocmhub.server.orchestrator.worker import execute_job


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
            raise RuntimeError(f"Agent API request failed ({exc.code})") from exc
        except URLError as exc:
            raise RuntimeError("Agent API is unavailable") from exc


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
            print(json.dumps(event, sort_keys=True), flush=True)
        else:
            print(f"[agent] {event.get('event')}: {event.get('message', event.get('agent_id', ''))}", flush=True)

    def run_forever(self) -> None:
        while True:
            try:
                self.run_once()
            except RuntimeError as exc:
                self._log({"event": "connection_error", "message": str(exc)})
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
    def _small_artifact_files(output_dir: Optional[str]) -> Dict[str, str]:
        """Return the allowlisted textual outputs; never upload weights or arbitrary local files."""
        if not output_dir:
            return {}
        allowed = {
            "build_manifest.json",
            "checksums.json",
            "model_config.json",
            "recipe.json",
            "run_inference.py",
            "runtime_config.json",
        }
        uploaded: Dict[str, str] = {}
        for path in Path(output_dir).iterdir():
            if path.name not in allowed or not path.is_file() or path.stat().st_size > 262_144:
                continue
            try:
                content = path.read_text(encoding="utf-8")
                if path.suffix == ".json":
                    content = json.dumps(ROCmHubAgent._redact_local_paths(json.loads(content)), indent=2, sort_keys=True) + "\n"
                uploaded[path.name] = content
            except UnicodeDecodeError:
                continue
            except json.JSONDecodeError:
                continue
        return uploaded

    def _run_claim(self, job: Dict[str, Any], payload: Dict[str, Any]) -> None:
        assert self.agent_id
        job_id = job["job_id"]
        if job["job_type"] != JobType.PREPARE_MODEL_FOR_AMD.value:
            self.api.call(
                "POST",
                f"/api/v1/agents/{self.agent_id}/jobs/{job_id}/fail",
                {"error_message": "Unsupported job type for this agent", "error_code": "UNSUPPORTED_AGENT_JOB"},
            )
            return
        self._log({"event": "claimed", "job_id": job_id})

        def emit(
            phase: str,
            status: str,
            message: str,
            error_code: Optional[str] = None,
            details: Optional[Dict[str, Any]] = None,
        ) -> JobEvent:
            self.api.call(
                "POST",
                f"/api/v1/agents/{self.agent_id}/jobs/{job_id}/events",
                {"phase": phase, "status": status, "message": message, "details": details},
            )
            return cast(JobEvent, None)

        try:
            local_payload = {
                **payload,
                "output_dir": str(self.workspace / job_id),
                "allow_full_weights": False,
                "execute_inference": False,
            }
            cancel = threading.Event()
            status, domain, output_dir, result, error, error_code, revision = execute_job(
                job_id, JobType.FORGE_BUILD, local_payload, cancel, emit
            )
            if status == JobStatus.SUCCEEDED:
                enriched = {
                    "preparation": "PREPARE_MODEL_FOR_AMD",
                    "weights": "NOT_DOWNLOADED",
                    "target_gfx": payload.get("target_gfx"),
                    "runtime": payload.get("runtime"),
                    "artifacts": (result or {}).get("artifacts", {}),
                    "artifact_manifest": self._redact_local_paths(result),
                    "agent_observation": {
                        "physical_amd_execution": False,
                        "reason": "This Agent performed configuration preparation only.",
                    },
                }
                self.api.call(
                    "POST",
                    f"/api/v1/agents/{self.agent_id}/jobs/{job_id}/complete",
                    {
                        "domain_status": domain or "CONFIG_ONLY",
                        "result": enriched,
                        "revision": revision,
                        "artifact_files": self._small_artifact_files(output_dir),
                    },
                )
            else:
                self.api.call(
                    "POST",
                    f"/api/v1/agents/{self.agent_id}/jobs/{job_id}/fail",
                    {"error_message": error or "Preparation failed", "error_code": error_code or "PREPARATION_FAILED"},
                )
        except Exception:
            self.api.call(
                "POST",
                f"/api/v1/agents/{self.agent_id}/jobs/{job_id}/fail",
                {"error_message": "Agent preparation failed", "error_code": "AGENT_EXECUTION_FAILED"},
            )
