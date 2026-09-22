"""End-to-End Live Demo Smoke Test for ROCmHub Phase 14.

Validates the full interactive scenario against the running FastAPI backend:
1. GET /health (Dashboard state, rocm_available: false on Mac, hardware notice)
2. GET /api/v1/models/Qwen/Qwen2.5-0.5B-Instruct (Real metadata, commit SHA, architecture)
3. POST /api/v1/forge/plan (Deterministic plan, pytorch_transformers_hip recipe)
4. POST /api/v1/jobs (FORGE_BUILD CONFIG_ONLY job)
5. GET /api/v1/jobs/{job_id}/events (SSE progress streaming)
6. GET /api/v1/jobs/{job_id}/result (Manifest, artifacts, CONFIG_ONLY)
7. GET /api/v1/jobs (Job history in SQLite)
8. POST /api/v1/jobs (ENGINEER job PREPARE_AMD)
9. GET /api/v1/jobs/{eng_id}/events (SSE progress streaming for engineer)
10. POST /api/v1/jobs (OPTIMIZATION job MAX_THROUGHPUT, NOT_MEASURED on Mac)
"""

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def http_get(url: str) -> dict:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_post(url: str, data: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def read_sse_events(url: str, max_events: int = 50, timeout_sec: float = 45.0) -> list:
    req = urllib.request.Request(url, headers={"Accept": "text/event-stream"})
    events = []
    start_time = time.time()

    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        current_event = {}
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line:
                if "data" in current_event:
                    events.append(current_event)
                    evt_type = current_event.get("event", "")
                    try:
                        data_obj = json.loads(current_event["data"])
                        st = data_obj.get("status")
                        if st in ("SUCCEEDED", "FAILED", "CANCELLED") or evt_type in ("job_completed", "job_failed", "job_cancelled"):
                            break
                    except Exception:
                        pass
                    if len(events) >= max_events:
                        break
                current_event = {}
                continue

            if line.startswith("id:"):
                current_event["id"] = line[3:].strip()
            elif line.startswith("event:"):
                current_event["event"] = line[6:].strip()
            elif line.startswith("data:"):
                current_event["data"] = line[5:].strip()

            if time.time() - start_time > timeout_sec:
                break

    return events


def wait_for_job_terminal(base_url: str, job_id: str, timeout_sec: float = 15.0) -> dict:
    url = f"{base_url}/api/v1/jobs/{job_id}"
    start = time.time()
    while time.time() - start < timeout_sec:
        data = http_get(url)
        if data.get("status") in ("SUCCEEDED", "FAILED", "CANCELLED"):
            return data
        time.sleep(0.2)
    return http_get(url)


def main():
    print("=" * 70)
    print("ROCmHub Phase 14: Live Demo Smoke Test (Backend + Frontend Integration)")
    print("=" * 70)

    port = 8768
    db_file = Path("/tmp/rocmhub_live_smoke.db")
    if db_file.exists():
        db_file.unlink()

    workspace_dir = Path("/tmp/rocmhub_live_workspace")
    workspace_dir.mkdir(parents=True, exist_ok=True)

    # 1. Start Server
    server_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "rocmhub.server.app:create_app",
        "--factory",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    env = {
        **os.environ,
        "ROCMHUB_DB_PATH": str(db_file),
        "ROCMHUB_ALLOWED_WORKSPACES": f"{workspace_dir},/tmp",
    }
    server_proc = subprocess.Popen(server_cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        # Wait for server ready
        print("[1/10] Starting backend and waiting for /health...")
        health_url = f"http://127.0.0.1:{port}/health"
        health_data = None
        for _ in range(30):
            try:
                health_data = http_get(health_url)
                if health_data.get("status") == "healthy":
                    break
            except Exception:
                time.sleep(0.2)

        assert health_data is not None, "Failed to connect to /health"
        print(f" -> Health Status: {health_data['status']}, API v{health_data['version']}")
        print(f" -> Host OS: {health_data['host_platform']['os']}, Arch: {health_data['host_platform']['arch']}")
        print(f" -> ROCm Available: {health_data['rocm_available']}")
        print(f" -> Notice: {health_data['warnings'][0] if health_data['warnings'] else 'None'}")
        assert health_data["rocm_available"] is False, "Expected False on Mac"

        # 2. Inspect Model
        print("\n[2/10] Inspecting Qwen/Qwen2.5-0.5B-Instruct metadata...")
        model_url = f"http://127.0.0.1:{port}/api/v1/models/Qwen/Qwen2.5-0.5B-Instruct"
        model_meta = http_get(model_url)
        print(f" -> Model ID: {model_meta['model_id']}")
        print(f" -> Immutable Commit SHA: {model_meta['commit_sha']}")
        print(f" -> Architecture: {model_meta['architecture']}")
        print(f" -> Parameters: {model_meta['parameter_count']}")
        print(f" -> Compatibility Verdict: {model_meta['compatibility']['verdict']}")
        assert model_meta["model_id"] == "Qwen/Qwen2.5-0.5B-Instruct"
        assert len(model_meta["commit_sha"]) == 40, "Expected 40-character commit SHA"

        # 3. Create Forge Plan
        print("\n[3/10] Generating deterministic Forge Plan...")
        plan_url = f"http://127.0.0.1:{port}/api/v1/forge/plan"
        plan = http_post(
            plan_url,
            {
                "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
                "precision": "fp16",
                "target_gpu": "gfx90a",
            },
        )
        print(f" -> Plan ID: {plan['plan_id']}")
        print(f" -> Plan Recipe: {plan['recipe_id']} v{plan['recipe_version']}")
        print(f" -> Ordered Steps: {[s['name'] for s in plan['steps']]}")
        assert plan["recipe_id"] == "pytorch_transformers_hip"

        # 4. Enqueue FORGE_BUILD Job
        print("\n[4/10] Submitting FORGE_BUILD job (CONFIG_ONLY)...")
        jobs_url = f"http://127.0.0.1:{port}/api/v1/jobs"
        forge_job = http_post(
            jobs_url,
            {
                "job_type": "FORGE_BUILD",
                "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
                "precision": "fp16",
                "target_gpu": "gfx90a",
                "allow_full_weights": False,
                "output_dir": str(workspace_dir / "bld_smoke"),
            },
        )
        forge_job_id = forge_job["job_id"]
        print(f" -> Job Enqueued: ID={forge_job_id}, Status={forge_job['status']}")

        # 5. Stream SSE Events
        print(f"\n[5/10] Streaming SSE events from /api/v1/jobs/{forge_job_id}/events...")
        sse_url = f"http://127.0.0.1:{port}/api/v1/jobs/{forge_job_id}/events"
        events = read_sse_events(sse_url)
        print(f" -> Captured {len(events)} SSE progress events:")
        for evt in events:
            d = json.loads(evt["data"])
            print(f"    [{d.get('phase', 'EVENT')}] {d.get('status')} - {d.get('message')}")

        # 6. Fetch Job Result
        print(f"\n[6/10] Retrieving final result from /api/v1/jobs/{forge_job_id}/result...")
        result_url = f"http://127.0.0.1:{port}/api/v1/jobs/{forge_job_id}/result"
        result = http_get(result_url)
        print(f" -> Job Status: {result['job_status']}")
        print(f" -> Domain Status: {result['domain_status']}")
        print(f" -> Manifest Artifacts: {list(result['result'].get('artifacts', {}).keys())}")
        assert result["job_status"] == "SUCCEEDED"
        assert result["domain_status"] == "CONFIG_ONLY"

        # 7. Check Job History API
        print("\n[7/10] Querying job history from GET /api/v1/jobs...")
        job_list = http_get(f"http://127.0.0.1:{port}/api/v1/jobs")
        print(f" -> Total Jobs in SQLite: {job_list['total']}")
        print(f" -> Latest Job Revision: {job_list['items'][0]['revision']}")
        assert any(j["job_id"] == forge_job_id for j in job_list["items"])
        assert len(job_list["items"][0]["revision"]) == 40, "Resolved commit SHA must be saved in DB"

        # 8. Run AI Engineer Job
        print("\n[8/10] Submitting AI Engineer PREPARE_AMD job...")
        eng_job = http_post(
            jobs_url,
            {
                "job_type": "ENGINEER",
                "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
                "objective": "PREPARE_AMD",
                "max_attempts": 10,
                "allow_full_weights": False,
                "output_dir": str(workspace_dir / "eng_smoke"),
            },
        )
        eng_job_id = eng_job["job_id"]
        print(f" -> Engineer Job Enqueued: ID={eng_job_id}")

        eng_events = read_sse_events(f"http://127.0.0.1:{port}/api/v1/jobs/{eng_job_id}/events")
        print(f" -> Captured {len(eng_events)} Engineer events:")
        for evt in eng_events:
            d = json.loads(evt["data"])
            print(f"    [{d.get('phase', 'EVENT')}] {d.get('status')} - {d.get('message')}")

        wait_for_job_terminal(f"http://127.0.0.1:{port}", eng_job_id)
        eng_result = http_get(f"http://127.0.0.1:{port}/api/v1/jobs/{eng_job_id}/result")
        print(f" -> Engineer Outcome: {eng_result['job_status']}, Domain: {eng_result['domain_status']}")
        assert eng_result["job_status"] == "SUCCEEDED"
        assert eng_result["domain_status"] == "CONFIG_ONLY"

        # 9. Run Optimization Lab Job
        print("\n[9/10] Submitting Optimization Lab job...")
        opt_job = http_post(
            jobs_url,
            {
                "job_type": "OPTIMIZATION",
                "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
                "objective": "MAX_THROUGHPUT",
                "strategies": ["fp16", "bf16"],
                "allow_full_weights": False,
                "output_dir": str(workspace_dir / "opt_smoke"),
            },
        )
        opt_job_id = opt_job["job_id"]
        print(f" -> Optimization Job Enqueued: ID={opt_job_id}")

        opt_events = read_sse_events(f"http://127.0.0.1:{port}/api/v1/jobs/{opt_job_id}/events")
        print(f" -> Captured {len(opt_events)} Optimization events")

        wait_for_job_terminal(f"http://127.0.0.1:{port}", opt_job_id)
        opt_result = http_get(f"http://127.0.0.1:{port}/api/v1/jobs/{opt_job_id}/result")
        print(f" -> Optimization Status: {opt_result['job_status']}")
        print(f" -> Baseline Domain Status: {opt_result['domain_status']} (Zero synthetic metrics on Mac)")
        assert opt_result["job_status"] == "SUCCEEDED"

        # 10. Summary
        print("\n[10/10] LIVE DEMO E2E VERIFICATION COMPLETED SUCCESSFULLY!")
        print("=" * 70)

    finally:
        server_proc.terminate()
        try:
            server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_proc.kill()


if __name__ == "__main__":
    main()
