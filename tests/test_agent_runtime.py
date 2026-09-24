"""Remote Agent lifecycle logging and retry tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

from rocmhub.agent import runtime
from rocmhub.server.orchestrator.models import JobStatus


class FakeApi:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Dict[str, Any]]] = []
        self.completion_failures = 1

    def call(self, method: str, path: str, payload: Optional[Dict[str, Any]] = None) -> Any:
        self.calls.append((path, payload or {}))
        if path.endswith("/complete") and self.completion_failures:
            self.completion_failures -= 1
            raise runtime.AgentApiError("Agent API is unavailable", transient=True)
        return None


def test_agent_lifecycle_logs_retry_and_json_records(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    output = tmp_path / "artifacts"
    output.mkdir()
    (output / "runtime_config.json").write_text('{"weights_path":"/private/cache/model","device":"cuda"}\n')
    (output / "run_inference.py").write_text("print('safe')\n")

    def fake_execute_job(
        job_id: str,
        _job_type: Any,
        _payload: Dict[str, Any],
        _cancel: Any,
        emit: Any,
    ) -> tuple[Any, ...]:
        emit("PREPARING", "RUNNING", "Preparing model", None, None)
        return (
            JobStatus.SUCCEEDED,
            "CONFIG_ONLY",
            str(output),
            {"artifacts": {"runtime_config.json": hashlib.sha256((output / "runtime_config.json").read_bytes()).hexdigest()}},
            None,
            None,
            "a" * 40,
        )

    monkeypatch.setattr(runtime, "execute_job", fake_execute_job)
    agent = runtime.ROCmHubAgent(
        server="https://example.invalid",
        token="must-not-be-printed",
        name="test-agent",
        workspace=str(tmp_path / "workspace"),
        poll_interval=0.2,
        json_output=True,
    )
    agent.agent_id = "agent_test"
    agent.api.agent_id = "agent_test"
    api = FakeApi()
    agent.api = api  # type: ignore[assignment]

    agent._run_claim(
        {"job_id": "job_test123", "job_type": "PREPARE_MODEL_FOR_AMD", "attempt": 3},
        {"target_gfx": "gfx1100", "runtime": "pytorch_transformers_hip"},
    )

    lines = capsys.readouterr().out.splitlines()
    events = [json.loads(line) for line in lines]
    event_types = [event["type"] for event in events]
    assert event_types == [
        "claimed",
        "preparing",
        "artifacts_ready",
        "uploading_artifacts",
        "completing",
        "connection_error",
        "connection_restored",
        "completed",
    ]
    assert events[0]["job_id"] == "job_test123" and events[0]["attempt"] == 3
    assert events[2]["count"] == 2
    assert events[5]["transient"] is True and events[5]["retrying"] is True
    assert events[-1]["status"] == "CONFIG_ONLY"
    assert "must-not-be-printed" not in "\n".join(lines)
    completions = [payload for path, payload in api.calls if path.endswith("/complete")]
    assert len(completions) == 2
    assert completions[0] == completions[1]
    assert completions[0]["completion_id"]
    assert "/private/cache" not in json.dumps(completions[0])
    assert completions[0]["result"]["source_artifacts"]["runtime_config.json"]["sha256"]


def test_agent_human_lifecycle_output_identifies_job_attempt_and_status(tmp_path: Path, capsys: Any) -> None:
    agent = runtime.ROCmHubAgent(
        server="https://example.invalid",
        token="not-for-output",
        name="test-agent",
        workspace=str(tmp_path / "workspace"),
    )
    for record in (
        {"event": "claimed", "job_id": "job_abcd1234", "attempt": 2},
        {"event": "preparing", "job_id": "job_abcd1234"},
        {"event": "artifacts_ready", "job_id": "job_abcd1234", "count": 4},
        {"event": "uploading_artifacts", "job_id": "job_abcd1234", "count": 4},
        {"event": "completing", "job_id": "job_abcd1234"},
        {"event": "completed", "job_id": "job_abcd1234", "status": "CONFIG_ONLY"},
    ):
        agent._log(record)
    output = capsys.readouterr().out
    assert "claimed: job_abcd1234 attempt=2" in output
    assert "preparing: job_abcd1234" in output
    assert "artifacts_ready: job_abcd1234 count=4" in output
    assert "uploading_artifacts: job_abcd1234 count=4" in output
    assert "completing: job_abcd1234" in output
    assert "completed: job_abcd1234 status=CONFIG_ONLY" in output
    assert "not-for-output" not in output
