"""JobManager: central coordinator for job lifecycle, queue, directory locking, and events."""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from rocmhub.server.config import ServerConfig
from rocmhub.server.orchestrator.db import DatabaseManager
from rocmhub.server.orchestrator.models import (
    AgentCapabilities,
    JobCreateRequest,
    JobEvent,
    JobResponse,
    JobResultResponse,
    JobStatus,
    JobType,
)
from rocmhub.server.orchestrator.worker import execute_job
from rocmhub.server.security import validate_job_path

logger = logging.getLogger(__name__)


class JobManager:
    """Manages asynchronous job scheduling, lifecycle, events, and concurrency."""

    def __init__(self, config: ServerConfig) -> None:
        self.config = config
        self.db = DatabaseManager(config.db_path)
        self._work_queue: queue.Queue[str] = queue.Queue(maxsize=config.max_queue_size)
        self._cancellation_events: Dict[str, threading.Event] = {}
        self._active_directory_locks: Set[str] = set()
        self._dir_lock_mutex = threading.Lock()
        self._subscriber_queues: Dict[str, List[asyncio.Queue[JobEvent]]] = {}
        self._subscribers_mutex = threading.Lock()
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Start the background worker thread and recover interrupted jobs."""
        interrupted = self.db.mark_interrupted_jobs_as_failed("Job aborted due to server restart / unexpected shutdown")
        if interrupted:
            logger.warning("Recovered and marked %d interrupted jobs as FAILED", len(interrupted))

        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._worker_loop, name="rocmhub-job-worker", daemon=True)
        self._worker_thread.start()
        logger.info("JobManager started with SQLite at %s", self.config.db_path)

    def stop(self) -> None:
        """Gracefully stop background worker thread."""
        self._stop_event.set()
        # Wake up worker queue
        try:
            self._work_queue.put_nowait("")
        except queue.Full:
            pass
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)
        logger.info("JobManager stopped")

    def enqueue_job(self, request: JobCreateRequest) -> JobResponse:
        """Enqueue a new job for background execution."""
        # 1. Validate output directory against allowed workspaces
        output_dir: Optional[str] = None
        if request.output_dir:
            validated_path = validate_job_path(request.output_dir, self.config.allowed_workspaces)
            output_dir = str(validated_path)

        # 2. Check queue capacity
        if self._work_queue.full():
            raise RuntimeError(f"Job queue is full (max {self.config.max_queue_size} jobs).")

        # 3. Create job record
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        now_iso = datetime.now(timezone.utc).isoformat()
        timeout = request.timeout_seconds or self.config.default_job_timeout_seconds

        req_dict = request.model_dump(mode="json")

        job_resp = self.db.insert_job(
            job_id=job_id,
            job_type=request.job_type,
            model_id=request.model_id,
            revision=request.revision,
            status=JobStatus.QUEUED,
            created_at=now_iso,
            timeout_seconds=timeout,
            output_dir=output_dir,
            request_payload=req_dict,
        )

        # 4. Record initial event
        self.emit_event(
            job_id=job_id,
            phase="QUEUE",
            status="QUEUED",
            message=f"Job {job_id} ({request.job_type.value}) enqueued successfully",
            error_code=None,
            details={"model_id": request.model_id, "timeout_seconds": timeout},
        )

        # 5. Existing jobs use the in-process worker. Remote preparation jobs are
        # deliberately left queued for an external authenticated Agent.
        self._cancellation_events[job_id] = threading.Event()
        if request.job_type != JobType.PREPARE_MODEL_FOR_AMD:
            self._work_queue.put(job_id)

        return job_resp

    def claim_agent_job(
        self, agent_id: str, capabilities: AgentCapabilities
    ) -> Optional[Tuple[JobResponse, Dict[str, Any]]]:
        """Atomically assign one honest, eligible queued job to an external Agent."""
        released = self.db.release_stale_claims(self.config.agent_heartbeat_timeout_seconds)
        for job_id in released:
            self.emit_event(
                job_id,
                "AGENT_LOST",
                "QUEUED",
                "Agent heartbeat timed out; job released for reassignment.",
                "AGENT_HEARTBEAT_TIMEOUT",
            )
        can_execute_amd = (
            capabilities.os.lower() == "linux"
            and capabilities.rocm_detected
            and capabilities.hip_detected
            and capabilities.amd_gpu_count > 0
        )
        result = self.db.claim_next_job(agent_id, capabilities.capabilities, can_execute_amd)
        if result:
            job, _ = result
            self.emit_event(
                job.job_id,
                "CLAIMED",
                "RUNNING",
                f"Claimed by agent {agent_id}",
                details={"agent_id": agent_id, "attempt": job.attempt},
            )
        return result

    def agent_heartbeat(self, agent_id: str) -> bool:
        return self.db.heartbeat_agent(agent_id)

    def agent_event(
        self,
        agent_id: str,
        job_id: str,
        phase: str,
        status: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        attempt: Optional[int] = None,
    ) -> bool:
        if not self.db.agent_owns_running_job(agent_id, job_id, attempt):
            return False
        self.db.heartbeat_agent(agent_id)
        self.emit_event(job_id, phase, status, message, details=details)
        return True

    def complete_agent_job(
        self,
        agent_id: str,
        job_id: str,
        domain_status: str,
        result: Dict[str, Any],
        revision: Optional[str],
        attempt: Optional[int] = None,
    ) -> bool:
        now = datetime.now(timezone.utc).isoformat()
        if not self.db.finish_agent_job(
            agent_id,
            job_id,
            attempt,
            JobStatus.SUCCEEDED,
            completed_at=now,
            domain_status=domain_status,
            result_payload=result,
            revision=revision,
        ):
            return False
        self.emit_event(
            job_id,
            "COMPLETED",
            "SUCCEEDED",
            "External agent preparation completed",
            details={"agent_id": agent_id, "domain_status": domain_status},
        )
        return True

    def fail_agent_job(
        self,
        agent_id: str,
        job_id: str,
        error_message: str,
        error_code: str,
        attempt: Optional[int] = None,
    ) -> bool:
        now = datetime.now(timezone.utc).isoformat()
        if not self.db.finish_agent_job(
            agent_id,
            job_id,
            attempt,
            JobStatus.FAILED,
            completed_at=now,
            error_message=error_message,
            error_code=error_code,
        ):
            return False
        self.emit_event(job_id, "COMPLETED", "FAILED", "External agent job failed", error_code)
        return True

    def get_job(self, job_id: str) -> Optional[JobResponse]:
        """Get job status and metadata."""
        return self.db.get_job(job_id)

    def list_jobs(
        self,
        limit: int = 20,
        offset: int = 0,
        status: Optional[JobStatus] = None,
        job_type: Optional[JobType] = None,
    ) -> Tuple[List[JobResponse], int]:
        """List jobs with pagination, filtering, and newest first."""
        return self.db.list_jobs(limit=limit, offset=offset, status=status, job_type=job_type)

    def get_job_result(self, job_id: str) -> Optional[JobResultResponse]:
        """Get domain result payload for a job."""
        full_job = self.db.get_job_full(job_id)
        if not full_job:
            return None

        return JobResultResponse(
            job_id=full_job["job_id"],
            job_type=JobType(full_job["job_type"]),
            job_status=JobStatus(full_job["status"]),
            domain_status=full_job["domain_status"],
            output_dir=full_job["output_dir"],
            completed_at=full_job["completed_at"],
            result=full_job.get("result_payload"),
            error_message=full_job.get("error_message"),
        )

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a queued or running job."""
        job = self.db.get_job(job_id)
        if not job or job.status in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED):
            return False

        # If running, signal cancellation event
        cancel_evt = self._cancellation_events.get(job_id)
        if cancel_evt:
            cancel_evt.set()

        # Remote Agents cannot share the server's in-process cancellation event.
        # Fence the claim immediately; the Agent's bounded status checks then stop
        # cache publication and its terminal request cannot overwrite cancellation.
        if job.agent_id and job.status == JobStatus.RUNNING:
            now_iso = datetime.now(timezone.utc).isoformat()
            self.db.update_job_status(
                job_id=job_id,
                status=JobStatus.CANCELLED,
                completed_at=now_iso,
                error_message="Job cancelled by user request",
                error_code="JOB_CANCELLED",
            )
            self.emit_event(job_id, "CANCELLATION", "CANCELLED", "Remote preparation cancelled by user", "JOB_CANCELLED")
            return True

        # If queued, immediately transition to CANCELLED in DB
        if job.status == JobStatus.QUEUED:
            now_iso = datetime.now(timezone.utc).isoformat()
            self.db.update_job_status(
                job_id=job_id,
                status=JobStatus.CANCELLED,
                completed_at=now_iso,
                error_message="Job cancelled by user while queued",
                error_code="JOB_CANCELLED",
            )
            self.emit_event(
                job_id=job_id,
                phase="QUEUE",
                status="CANCELLED",
                message="Job cancelled by user while waiting in queue",
                error_code="JOB_CANCELLED",
            )

        return True

    def emit_event(
        self,
        job_id: str,
        phase: str,
        status: str,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> JobEvent:
        """Persist event to database and broadcast to active SSE listeners."""
        event = self.db.insert_event(
            job_id=job_id,
            phase=phase,
            status=status,
            message=message,
            error_code=error_code,
            details=details,
        )

        # Broadcast event to any active SSE subscriber queues
        with self._subscribers_mutex:
            subscribers = self._subscriber_queues.get(job_id, [])
            for q in list(subscribers):
                try:
                    q.put_nowait(event)
                except Exception:
                    pass

        return event

    def register_subscriber(self, job_id: str, q: asyncio.Queue[JobEvent]) -> None:
        """Register an asyncio queue to receive live SSE progress events."""
        with self._subscribers_mutex:
            if job_id not in self._subscriber_queues:
                self._subscriber_queues[job_id] = []
            self._subscriber_queues[job_id].append(q)

    def unregister_subscriber(self, job_id: str, q: asyncio.Queue[JobEvent]) -> None:
        """Unregister an asyncio queue."""
        with self._subscribers_mutex:
            if job_id in self._subscriber_queues:
                if q in self._subscriber_queues[job_id]:
                    self._subscriber_queues[job_id].remove(q)
                if not self._subscriber_queues[job_id]:
                    del self._subscriber_queues[job_id]

    def _acquire_dir_lock(self, dir_path: Optional[str]) -> bool:
        """Acquire in-memory lock for a target output directory."""
        if not dir_path:
            return True
        norm_dir = str(Path(dir_path).resolve())
        with self._dir_lock_mutex:
            if norm_dir in self._active_directory_locks:
                return False
            self._active_directory_locks.add(norm_dir)
            return True

    def _release_dir_lock(self, dir_path: Optional[str]) -> None:
        """Release lock on output directory."""
        if not dir_path:
            return
        norm_dir = str(Path(dir_path).resolve())
        with self._dir_lock_mutex:
            self._active_directory_locks.discard(norm_dir)

    def _worker_loop(self) -> None:
        """Main background loop processing jobs sequentially."""
        while not self._stop_event.is_set():
            try:
                job_id = self._work_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if not job_id or self._stop_event.is_set():
                break

            job_data = self.db.get_job_full(job_id)
            if not job_data:
                self._work_queue.task_done()
                continue

            # Check if job was cancelled while queued
            if job_data["status"] == JobStatus.CANCELLED.value:
                self._work_queue.task_done()
                continue

            output_dir = job_data.get("output_dir")

            # Try to acquire directory lock; if already locked by another task, re-queue
            if output_dir and not self._acquire_dir_lock(output_dir):
                logger.info("Directory %s is locked, re-queuing job %s", output_dir, job_id)
                self._work_queue.put(job_id)
                self._work_queue.task_done()
                threading.Event().wait(0.5)
                continue

            cancellation_event = self._cancellation_events.get(job_id, threading.Event())
            started_at = datetime.now(timezone.utc).isoformat()

            # Mark RUNNING in DB
            self.db.update_job_status(
                job_id=job_id,
                status=JobStatus.RUNNING,
                started_at=started_at,
            )

            def emit(
                phase: str,
                status: str,
                msg: str,
                err_code: Optional[str] = None,
                details: Optional[Dict[str, Any]] = None,
            ) -> JobEvent:
                return self.emit_event(job_id, phase, status, msg, err_code, details)

            try:
                (
                    job_status,
                    domain_status,
                    resolved_output_dir,
                    result_payload,
                    error_msg,
                    error_code,
                    resolved_revision,
                ) = execute_job(
                    job_id=job_id,
                    job_type=JobType(job_data["job_type"]),
                    request_data=job_data["request_payload"],
                    cancellation_event=cancellation_event,
                    emit=emit,
                )

                completed_at = datetime.now(timezone.utc).isoformat()
                self.db.update_job_status(
                    job_id=job_id,
                    status=job_status,
                    completed_at=completed_at,
                    domain_status=domain_status,
                    output_dir=resolved_output_dir or output_dir,
                    result_payload=result_payload,
                    error_message=error_msg,
                    error_code=error_code,
                    revision=resolved_revision,
                )
                self.emit_event(
                    job_id=job_id,
                    phase="COMPLETED",
                    status=job_status.value,
                    message=f"Job {job_id} finished with status {job_status.value}",
                    error_code=error_code,
                    details={
                        "domain_status": domain_status,
                        "output_dir": resolved_output_dir or output_dir,
                    },
                )
            except Exception as exc:
                logger.exception("Unexpected error in worker loop for job %s", job_id)
                completed_at = datetime.now(timezone.utc).isoformat()
                self.db.update_job_status(
                    job_id=job_id,
                    status=JobStatus.FAILED,
                    completed_at=completed_at,
                    error_message=str(exc),
                    error_code="WORKER_EXCEPTION",
                )
                self.emit_event(
                    job_id=job_id,
                    phase="COMPLETED",
                    status=JobStatus.FAILED.value,
                    message=f"Worker error: {exc}",
                    error_code="WORKER_EXCEPTION",
                )
            finally:
                self._release_dir_lock(output_dir)
                self._cancellation_events.pop(job_id, None)
                self._work_queue.task_done()
