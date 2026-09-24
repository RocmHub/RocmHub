"""Authenticated external ROCmHub Agent control-plane API."""

from __future__ import annotations

import hashlib
import secrets
import uuid
from typing import List, Optional, cast

from fastapi import APIRouter, Header, HTTPException, Request, status

from rocmhub.server.orchestrator.models import (
    AgentClaimResponse,
    AgentCompleteRequest,
    AgentEventRequest,
    AgentFailRequest,
    AgentRegisterRequest,
    AgentResponse,
)

router = APIRouter(prefix="/api/v1/agents", tags=["Agents"])


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
            agent_id, payload.name, payload.hostname, payload.capabilities, digest
        ),
    )


@router.get("", response_model=List[AgentResponse])
async def list_agents(req: Request) -> List[AgentResponse]:
    req.app.state.job_manager.db.release_stale_claims(req.app.state.config.agent_heartbeat_timeout_seconds)
    return cast(List[AgentResponse], req.app.state.job_manager.db.list_agents())


@router.post("/{agent_id}/heartbeat", response_model=AgentResponse)
async def heartbeat(agent_id: str, req: Request, authorization: Optional[str] = Header(default=None)) -> AgentResponse:
    _authorize(req, authorization)
    if not req.app.state.job_manager.agent_heartbeat(agent_id):
        raise HTTPException(status_code=404, detail="Agent not found")
    agent = req.app.state.job_manager.db.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return cast(AgentResponse, agent)


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
        agent_id, job_id, payload.phase, payload.status, payload.message, payload.details
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
    if not req.app.state.job_manager.complete_agent_job(
        agent_id, job_id, payload.domain_status, payload.result, payload.output_dir, payload.revision
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
    if not req.app.state.job_manager.fail_agent_job(agent_id, job_id, payload.error_message, payload.error_code):
        raise HTTPException(status_code=409, detail="Agent does not own this running job")
