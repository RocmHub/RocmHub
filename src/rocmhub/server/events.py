"""Server-Sent Events (SSE) streaming and event formatting."""

from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator

from rocmhub.server.orchestrator.manager import JobManager
from rocmhub.server.orchestrator.models import JobEvent, JobStatus


def format_sse_event(event: JobEvent) -> str:
    """Format a JobEvent into standard SSE protocol text."""
    payload = json.dumps(event.model_dump(mode="json"))
    return f"id: {event.event_id}\nevent: message\ndata: {payload}\n\n"


async def sse_event_stream(
    job_id: str,
    manager: JobManager,
    from_event_id: int = 0,
) -> AsyncIterator[str]:
    """Stream SSE events for a job, replaying history then subscribing live until terminal state."""
    # 1. Subscribe for live incoming events first to eliminate race condition gaps
    subscriber_queue: asyncio.Queue[JobEvent] = asyncio.Queue()
    manager.register_subscriber(job_id, subscriber_queue)

    try:
        # 2. Replay historical events from SQLite
        past_events = manager.db.get_events(job_id, from_event_id=from_event_id)
        last_seen_id = from_event_id
        for evt in past_events:
            last_seen_id = max(last_seen_id, evt.event_id)
            yield format_sse_event(evt)

        # 3. Check if job is already in a terminal state
        job = manager.get_job(job_id)
        if not job or job.status in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED):
            return

        while True:
            try:
                event = await asyncio.wait_for(subscriber_queue.get(), timeout=15.0)
                # Avoid duplicates if emitted during transition
                if event.event_id > last_seen_id:
                    last_seen_id = event.event_id
                    yield format_sse_event(event)

                # If job reached terminal status, end stream
                if event.status in ("SUCCEEDED", "FAILED", "CANCELLED"):
                    break
            except asyncio.TimeoutError:
                # SSE keep-alive heartbeat comment
                yield ": ping\n\n"

                # Check if job completed while quiet
                current_job = manager.get_job(job_id)
                if current_job and current_job.status in (
                    JobStatus.SUCCEEDED,
                    JobStatus.FAILED,
                    JobStatus.CANCELLED,
                ):
                    break
    finally:
        manager.unregister_subscriber(job_id, subscriber_queue)
