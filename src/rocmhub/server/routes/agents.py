"""Authenticated external ROCmHub Agent control-plane API."""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import uuid
from pathlib import Path
from typing import List, Optional, cast

from fastapi import APIRouter, Header, HTTPException, Request, status

from rocmhub.server.orchestrator.models import (
    AgentClaimResponse,
    AgentCompleteRequest,
    AgentEventRequest,
    AgentFailRequest,
    AgentPresenceResponse,
    AgentRegisterRequest,
    AgentResponse,
)
from rocmhub.server.privacy import sanitize_public_value

router = APIRouter(prefix="/api/v1/agents", tags=["Agents"])

_ALLOWED_ARTIFACT_FILES = {
    "build_manifest.json",
    "checksums.json",
    "model_config.json",
    "recipe.json",
    "run_inference.py",
    "runtime_config.json",
}

_PRIVATE_PATH_PATTERN = re.compile(
    r"(?<!:)/(?:Users|home|root|private|tmp|var|Volumes|mnt|opt|usr|etc|srv|app|workspace|agent-workspace)/"
)


def _sanitize_artifact_content(name: str, content: str) -> str:
    """Keep downloaded Agent artifacts useful without publishing local identity or paths."""
    if name.endswith(".json"):
        try:
            parsed = json.loads(content)
        except ValueError:
            pass
        else:
            sanitized = sanitize_public_value(parsed, preserve_null_path_fields=True)
            if sanitized != parsed:
                return json.dumps(sanitized, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    return str(sanitize_public_value(content))


def _contains_agent_local_path(value: object) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in {"build_dir", "output_dir", "weights_path", "cache_path", "local_path"} and item is not None:
                return True
            if _contains_agent_local_path(item):
                return True
    elif isinstance(value, list):
        return any(_contains_agent_local_path(item) for item in value)
    elif isinstance(value, str):
        return bool(_PRIVATE_PATH_PATTERN.search(value))
    return False


def _store_agent_artifacts(
    root: Path,
    job_id: str,
    artifact_files: dict[str, str],
    max_bytes: int,
    attempt: Optional[int] = None,
    completion_id: Optional[str] = None,
) -> list[dict[str, object]]:
    """Store only known, small text artifacts below the configured storage root."""
    if not artifact_files:
        return []
    if len(artifact_files) > len(_ALLOWED_ARTIFACT_FILES):
        raise ValueError("Too many Agent artifacts")

    sanitized_files: dict[str, str] = {}
    for name, content in artifact_files.items():
        if name not in _ALLOWED_ARTIFACT_FILES or Path(name).name != name:
            raise ValueError("Unsupported Agent artifact name")
        sanitized_files[name] = _sanitize_artifact_content(name, content)

    for manifest_name, digest_map in (("build_manifest.json", "artifacts"), ("checksums.json", "files")):
        manifest_content = sanitized_files.get(manifest_name)
        if manifest_content is None:
            continue
        try:
            manifest = json.loads(manifest_content)
        except ValueError:
            continue
        hashes = manifest.get(digest_map) if isinstance(manifest, dict) else None
        if not isinstance(hashes, dict):
            continue
        changed = False
        updated_hashes = dict(hashes)
        for artifact_name, current_hash in updated_hashes.items():
            artifact_content = sanitized_files.get(str(artifact_name))
            if artifact_content is None or artifact_name == manifest_name or not isinstance(current_hash, str):
                continue
            stored_hash = hashlib.sha256(artifact_content.encode("utf-8")).hexdigest()
            if current_hash != stored_hash:
                updated_hashes[artifact_name] = stored_hash
                changed = True
        if changed:
            manifest[digest_map] = updated_hashes
            sanitized_files[manifest_name] = json.dumps(
                manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ) + "\n"

    root = root.resolve()
    job_dir = (root / job_id).resolve()
    if job_dir.parent != root:
        raise ValueError("Invalid job artifact path")
    scope = f"attempt-{attempt or 0}-{completion_id or uuid.uuid4().hex}"
    attempt_dir = (job_dir / scope).resolve()
    if attempt_dir.parent != job_dir:
        raise ValueError("Invalid job artifact path")
    attempt_dir.mkdir(parents=True, exist_ok=True)
    stored: list[dict[str, object]] = []
    for name, content in sanitized_files.items():
        encoded = content.encode("utf-8")
        if len(encoded) > max_bytes:
            raise ValueError(f"Agent artifact '{name}' exceeds the configured size limit")
        target = (attempt_dir / name).resolve()
        if target.parent != attempt_dir:
            raise ValueError("Invalid Agent artifact path")
        temp_target = target.with_suffix(f"{target.suffix}.tmp")
        temp_target.write_bytes(encoded)
        temp_target.replace(target)
        stored.append(
            {
                "name": name,
                "bytes": len(encoded),
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "storage_key": f"{scope}/{name}",
                "download_path": f"/api/v1/jobs/{job_id}/artifacts/{name}",
            }
        )
    return stored


def _token_hash(authorization: Optional[str]) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing agent token")
    token = authorization[7:].strip()
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing agent token")
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _authorize(req: Request, authorization: Optional[str]) -> str:
    digest = _token_hash(authorization)
    if req.app.state.job_manager.db.is_agent_token_revoked(digest) or not any(
        secrets.compare_digest(digest, expected) for expected in req.app.state.config.agent_token_hashes
    ):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid agent token")
    return digest


@router.post("/revoke", status_code=204)
async def revoke_own_token(req: Request, authorization: Optional[str] = Header(default=None)) -> None:
    """Revoke the credential used for this request; raw tokens are never persisted."""
    digest = _authorize(req, authorization)
    req.app.state.job_manager.db.revoke_agent_token(digest)


@router.post("/register", response_model=AgentResponse)
async def register_agent(
    payload: AgentRegisterRequest, req: Request, authorization: Optional[str] = Header(default=None)
) -> AgentResponse:
    digest = _authorize(req, authorization)
    # Stable agent identity is server assigned. An agent reconnects by its supplied id header.
    agent_id = req.headers.get("X-ROCmHub-Agent-ID") or f"agent_{uuid.uuid4().hex[:12]}"
    return cast(
        AgentResponse,
        req.app.state.job_manager.db.upsert_agent(
            agent_id,
            payload.capabilities,
            digest,
            req.app.state.config.agent_retention_seconds,
        ),
    )


@router.get("", response_model=List[AgentPresenceResponse])
async def list_agents(req: Request) -> List[AgentPresenceResponse]:
    req.app.state.job_manager.db.release_stale_claims(req.app.state.config.agent_heartbeat_timeout_seconds)
    return cast(
        List[AgentPresenceResponse],
        req.app.state.job_manager.db.list_active_agent_presence(req.app.state.config.agent_heartbeat_timeout_seconds),
    )


@router.post("/{agent_id}/heartbeat", response_model=AgentResponse)
async def heartbeat(agent_id: str, req: Request, authorization: Optional[str] = Header(default=None)) -> AgentResponse:
    _authorize(req, authorization)
    if not req.app.state.job_manager.agent_heartbeat(agent_id):
        raise HTTPException(status_code=404, detail="Agent not found")
    agent = req.app.state.job_manager.db.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return cast(AgentResponse, agent)


@router.get("/{agent_id}/jobs/{job_id}/status")
async def agent_job_status(agent_id: str, job_id: str, req: Request, authorization: Optional[str] = Header(default=None)) -> dict[str, str]:
    """Allow the authenticated claim owner to observe cancellation while materializing."""
    _authorize(req, authorization)
    job = req.app.state.job_manager.db.get_job_full(job_id)
    if not job or job.get("agent_id") != agent_id:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"status": str(job["status"])}


@router.post("/{agent_id}/claim", response_model=AgentClaimResponse)
async def claim(agent_id: str, req: Request, authorization: Optional[str] = Header(default=None)) -> AgentClaimResponse:
    _authorize(req, authorization)
    agent = req.app.state.job_manager.db.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    claimed = req.app.state.job_manager.claim_agent_job(agent_id, agent.capabilities)
    if not claimed:
        return AgentClaimResponse()
    job, request_payload = claimed
    return AgentClaimResponse(job=job, request_payload=request_payload)


@router.post("/{agent_id}/jobs/{job_id}/events", status_code=204)
async def event(
    agent_id: str,
    job_id: str,
    payload: AgentEventRequest,
    req: Request,
    authorization: Optional[str] = Header(default=None),
) -> None:
    _authorize(req, authorization)
    if not req.app.state.job_manager.agent_event(
        agent_id, job_id, payload.phase, payload.status, payload.message, payload.details, payload.attempt
    ):
        raise HTTPException(status_code=409, detail="Agent does not own this running job")


@router.post("/{agent_id}/jobs/{job_id}/complete", status_code=204)
async def complete(
    agent_id: str,
    job_id: str,
    payload: AgentCompleteRequest,
    req: Request,
    authorization: Optional[str] = Header(default=None),
) -> None:
    _authorize(req, authorization)
    if payload.completion_id:
        completion_body = payload.model_dump(mode="json", exclude={"completion_id"}, exclude_unset=True)
        expected_id = hashlib.sha256(
            json.dumps(completion_body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        if not secrets.compare_digest(payload.completion_id, expected_id):
            raise HTTPException(status_code=400, detail="Invalid completion identifier")
    manager = req.app.state.job_manager
    current = manager.db.get_job_full(job_id)
    # A completion response may be lost after the transaction commits. Accept only
    # an exact replay from the same agent; never turn an unrelated retry into success.
    if (
        current
        and current.get("status") == "SUCCEEDED"
        and current.get("agent_id") == agent_id
        and payload.completion_id
        and (current.get("result_payload") or {}).get("agent_completion_id") == payload.completion_id
    ):
        return
    if not manager.db.agent_owns_running_job(agent_id, job_id, payload.attempt):
        raise HTTPException(status_code=409, detail="Agent does not own this running job")
    if current and current.get("job_type") == "PREPARE_MODEL_FOR_AMD":
        if _contains_agent_local_path(payload.result):
            raise HTTPException(status_code=400, detail="Agent-local filesystem paths are not accepted in job results")
        request_payload = current.get("request_payload") or {}
        intent = request_payload.get("materialization_mode", "METADATA_ONLY")
        if payload.revision != current.get("revision"):
            raise HTTPException(status_code=400, detail="Agent completion revision does not match the immutable job revision")
        if payload.result.get("model_id") not in (None, current.get("model_id")):
            raise HTTPException(status_code=400, detail="Agent completion model does not match the job model")
        if payload.domain_status not in {"CONFIG_ONLY", "PREPARED"}:
            raise HTTPException(status_code=400, detail="This preparation workflow cannot report execution or validation")
        if intent == "METADATA_ONLY":
            metadata_result = payload.result.get("materialization") or {}
            if (
                payload.domain_status != "CONFIG_ONLY"
                or payload.result.get("weights") == "MATERIALIZED"
                or metadata_result.get("has_weights") is True
            ):
                raise HTTPException(status_code=400, detail="Metadata-only jobs cannot complete with materialized weights")
        if intent == "FULL_WEIGHTS":
            materialization = payload.result.get("materialization") or {}
            observation = payload.result.get("agent_observation") or {}
            if (
                request_payload.get("weights_consent") is not True
                or payload.domain_status != "PREPARED"
                or materialization.get("mode") != "FULL_WEIGHTS"
                or materialization.get("has_weights") is not True
                or not isinstance(materialization.get("materialized_bytes"), int)
                or materialization.get("materialized_bytes", 0) <= 0
                or not isinstance(materialization.get("weights_size_bytes"), int)
                or materialization.get("weights_size_bytes", 0) <= 0
                or materialization.get("cache_status") not in {"MISS_DOWNLOADED", "HIT_VERIFIED"}
                or materialization.get("integrity") not in {"FULL_SHA256_MANIFEST_VERIFIED", "MANIFEST_AND_SELECTED_SHA256_VERIFIED"}
                or observation.get("physical_amd_execution") is not False
                or payload.result.get("amd_validated") is not False
                or payload.result.get("weights") != "MATERIALIZED"
            ):
                raise HTTPException(status_code=400, detail="PREPARED requires consent, verified weights, and no AMD validation")
    try:
        stored = _store_agent_artifacts(
            req.app.state.config.artifact_storage_dir,
            job_id,
            payload.artifact_files,
            req.app.state.config.max_agent_artifact_bytes,
            payload.attempt,
            payload.completion_id,
        )
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid Agent artifact: {exc}") from exc
    result = dict(payload.result)
    agent = manager.db.get_agent(agent_id)
    if agent is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    reported = agent.capabilities
    result["executor_provenance"] = {
        "source": "agent_reported",
        "os": reported.os,
        "architecture": reported.architecture,
        "python_version": reported.python_version,
        "rocm_detected": reported.rocm_detected,
        "hip_detected": reported.hip_detected,
        "pytorch_version": reported.pytorch_version,
        "amd_gpu_models": reported.gpu_names if reported.amd_gpu_count > 0 else [],
        "gfx_targets": reported.gfx_targets if reported.amd_gpu_count > 0 else [],
        "capabilities": reported.capabilities,
    }
    result["server_artifacts"] = stored
    result["artifacts"] = {str(item["name"]): str(item["sha256"]) for item in stored}
    if stored:
        provenance = result.get("artifact_provenance")
        if isinstance(provenance, dict):
            for item in stored:
                entry = provenance.get(str(item["name"]))
                if isinstance(entry, dict):
                    entry["stored_sha256"] = item["sha256"]
                    entry["sanitized"] = entry.get("source_sha256") != item["sha256"]
    if payload.completion_id:
        result["agent_completion_id"] = payload.completion_id
    if not manager.complete_agent_job(
        agent_id, job_id, payload.domain_status, result, payload.revision, payload.attempt
    ):
        raise HTTPException(status_code=409, detail="Agent does not own this running job")


@router.post("/{agent_id}/jobs/{job_id}/fail", status_code=204)
async def fail(
    agent_id: str,
    job_id: str,
    payload: AgentFailRequest,
    req: Request,
    authorization: Optional[str] = Header(default=None),
) -> None:
    _authorize(req, authorization)
    if not req.app.state.job_manager.fail_agent_job(
        agent_id, job_id, payload.error_message, payload.error_code, payload.attempt
    ):
        raise HTTPException(status_code=409, detail="Agent does not own this running job")
