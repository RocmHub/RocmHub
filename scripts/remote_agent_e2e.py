#!/usr/bin/env python3
"""Run one real, bounded remote-Agent preparation job against a deployed control plane.

The script uses the production HTTP API, persists only allowlisted small artifacts
on the server, and never requests model weights or inference execution.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict
from urllib.request import Request, urlopen

from rocmhub.agent.runtime import AgentApi, ROCmHubAgent


def _request_json(server: str, path: str) -> Dict[str, Any]:
    request = Request(f"{server.rstrip('/')}{path}", headers={"Accept": "application/json"})
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _request_bytes(server: str, path: str) -> bytes:
    request = Request(f"{server.rstrip('/')}{path}")
    with urlopen(request, timeout=30) as response:
        return response.read()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Exercise the deployed ROCmHub remote-Agent preparation flow.")
    parser.add_argument("--server", required=True, help="Control-plane URL, e.g. https://api.example.com")
    parser.add_argument("--token", default=os.environ.get("ROCMHUB_AGENT_TOKEN"), help="Agent token or ROCMHUB_AGENT_TOKEN")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="Public model to prepare without weights")
    parser.add_argument("--target-gpu", default="Radeon RX 7900 XTX")
    parser.add_argument("--target-gfx", default="gfx1100")
    parser.add_argument("--timeout", type=int, default=180, help="Maximum smoke-test duration in seconds")
    parser.add_argument("--workspace", default=None, help="Temporary Agent workspace; removed by default")
    parser.add_argument("--keep-workspace", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not args.token:
        print("Missing --token or ROCMHUB_AGENT_TOKEN", file=sys.stderr)
        return 2

    workspace_created = args.workspace is None
    workspace = Path(args.workspace).resolve() if args.workspace else Path(tempfile.mkdtemp(prefix="rocmhub-agent-e2e-"))
    started = time.monotonic()
    try:
        health = _request_json(args.server, "/health")
        if health.get("status") != "healthy":
            raise RuntimeError("Control-plane health check did not return healthy")

        agent = ROCmHubAgent(
            server=args.server,
            token=args.token,
            name=f"remote-e2e-{uuid.uuid4().hex[:8]}",
            workspace=str(workspace),
            poll_interval=0.2,
            json_output=True,
        )
        agent.register()
        api = AgentApi(args.server, args.token)
        created = api.call(
            "POST",
            "/api/v1/jobs",
            {
                "job_type": "PREPARE_MODEL_FOR_AMD",
                "model_id": args.model,
                "target_gpu": args.target_gpu,
                "target_gfx": args.target_gfx,
                "precision": "fp16",
                "runtime": "pytorch_transformers_hip",
                "allow_full_weights": False,
            },
        )
        job_id = created["job_id"]
        agent.run_once()

        while time.monotonic() - started < args.timeout:
            result = _request_json(args.server, f"/api/v1/jobs/{job_id}/result")
            if result["job_status"] == "SUCCEEDED":
                payload = result.get("result") or {}
                artifacts = payload.get("server_artifacts") or []
                if payload.get("weights") != "NOT_DOWNLOADED" or not artifacts:
                    raise RuntimeError("Preparation completed without the required bounded artifact contract")
                first_artifact = artifacts[0]
                if not _request_bytes(args.server, first_artifact["download_path"]):
                    raise RuntimeError("Stored Agent artifact was not downloadable")
                summary = {
                    "status": "passed",
                    "job_id": job_id,
                    "agent_id": agent.agent_id,
                    "domain_status": result.get("domain_status"),
                    "artifact_count": len(artifacts),
                    "weights": payload.get("weights"),
                    "elapsed_seconds": round(time.monotonic() - started, 2),
                }
                print(json.dumps(summary, sort_keys=True))
                return 0
            if result["job_status"] in {"FAILED", "CANCELLED"}:
                raise RuntimeError(f"Preparation ended as {result['job_status']}: {result.get('error_message')}")
            time.sleep(0.5)
        raise RuntimeError(f"Timed out waiting for job {job_id}")
    except Exception as exc:
        print(f"remote-agent e2e failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if workspace_created and not args.keep_workspace:
            shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
