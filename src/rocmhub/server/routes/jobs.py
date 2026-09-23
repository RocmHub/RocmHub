"""Job management and SSE streaming routes for ROCmHub API."""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Header, HTTPException, Query, Request, status
from fastapi import Path as PathParam
from fastapi.responses import StreamingResponse

from rocmhub.core.errors import SecurityBoundaryError
from rocmhub.server.events import sse_event_stream
from rocmhub.server.orchestrator.manager import JobManager
from rocmhub.server.orchestrator.models import (
    JobCreateRequest,
    JobListResponse,
    JobResponse,
    JobResultResponse,
    JobStatus,
    JobType,
)

router = APIRouter(prefix="/api/v1/jobs", tags=["Jobs"])

JOB_ID_PATTERN = r"^[a-zA-Z0-9_\-]{1,64}$"


@router.get("", response_model=JobListResponse)
async def list_jobs(
    req: Request,
    limit: int = Query(default=20, ge=1, le=100, description="Page limit"),
    offset: int = Query(default=0, ge=0, description="Page offset"),
    status: Optional[JobStatus] = Query(default=None, description="Filter by status"),
    job_type: Optional[JobType] = Query(default=None, description="Filter by job type"),
) -> JobListResponse:
    """List jobs with pagination and filtering, newest first."""
    manager: JobManager = req.app.state.job_manager
    items, total = manager.list_jobs(limit=limit, offset=offset, status=status, job_type=job_type)
    return JobListResponse(items=items, total=total, limit=limit, offset=offset)


@router.post("", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_job(request_body: JobCreateRequest, req: Request) -> JobResponse:
    """Submit a new background job (FORGE_BUILD, ENGINEER, or OPTIMIZATION)."""
    manager: JobManager = req.app.state.job_manager
    try:
        job = manager.enqueue_job(request_body)
        return job
    except SecurityBoundaryError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Security boundary violation: {exc}",
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


@router.get("/{job_id}", response_model=JobResponse)
async def get_job_status(
    job_id: str = PathParam(..., pattern=JOB_ID_PATTERN, description="Unique job identifier"),
    req: Request = None,  # type: ignore[assignment]
) -> JobResponse:
    """Retrieve the current execution status and metadata of a job."""
    manager: JobManager = req.app.state.job_manager
    job = manager.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' was not found.",
        )
    return job


@router.get("/{job_id}/events")
async def get_job_events(
    job_id: str = PathParam(..., pattern=JOB_ID_PATTERN, description="Unique job identifier"),
    req: Request = None,  # type: ignore[assignment]
    from_event_id: int = Query(default=0, ge=0, description="Start streaming from this event ID"),
    last_event_id_header: Optional[str] = Header(default=None, alias="Last-Event-ID"),
) -> StreamingResponse:
    """Stream real-time progress events using Server-Sent Events (SSE). Replays missed events."""
    manager: JobManager = req.app.state.job_manager
    job = manager.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' was not found.",
        )

    # Reconnection logic: header takes precedence if valid integer
    start_id = from_event_id
    if last_event_id_header:
        try:
            start_id = int(last_event_id_header)
        except ValueError:
            pass

    return StreamingResponse(
        sse_event_stream(job_id=job_id, manager=manager, from_event_id=start_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/{job_id}/result", response_model=JobResultResponse)
async def get_job_result(
    job_id: str = PathParam(..., pattern=JOB_ID_PATTERN, description="Unique job identifier"),
    req: Request = None,  # type: ignore[assignment]
) -> JobResultResponse:
    """Retrieve detailed domain results (manifest, report, comparisons) of a finished job."""
    manager: JobManager = req.app.state.job_manager
    result = manager.get_job_result(job_id)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' was not found.",
        )
    return result


@router.post("/{job_id}/cancel")
async def cancel_job(
    job_id: str = PathParam(..., pattern=JOB_ID_PATTERN, description="Unique job identifier"),
    req: Request = None,  # type: ignore[assignment]
) -> Dict[str, Any]:
    """Request cooperative cancellation of a queued or running job."""
    manager = req.app.state.job_manager
    job = manager.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' was not found.",
        )

    success = manager.cancel_job(job_id)
    updated_job = manager.get_job(job_id)

    return {
        "job_id": job_id,
        "cancelled": success,
        "status": updated_job.status.value if updated_job else job.status.value,
    }
