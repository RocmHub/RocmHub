"""Thread-safe SQLite database access layer for Job Orchestrator."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from rocmhub.server.orchestrator.migrations import apply_migrations
from rocmhub.server.orchestrator.models import JobEvent, JobResponse, JobStatus, JobType


class DatabaseManager:
    """Manages SQLite database connections, migrations, and job/event persistence."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Create a configured SQLite connection."""
        conn = sqlite3.connect(str(self.db_path), timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA synchronous = NORMAL")
        return conn

    def _init_db(self) -> None:
        """Apply schema migrations on startup."""
        with self._lock:
            conn = self._get_connection()
            try:
                apply_migrations(conn)
            finally:
                conn.close()

    def insert_job(
        self,
        job_id: str,
        job_type: JobType,
        model_id: str,
        revision: Optional[str],
        status: JobStatus,
        created_at: str,
        timeout_seconds: int,
        output_dir: Optional[str],
        request_payload: Dict[str, Any],
    ) -> JobResponse:
        """Insert a newly queued job."""
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO jobs (
                        job_id, job_type, model_id, revision, status, domain_status,
                        created_at, started_at, completed_at, timeout_seconds,
                        output_dir, request_payload, result_payload, error_message, error_code
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job_id,
                        job_type.value,
                        model_id,
                        revision,
                        status.value,
                        None,
                        created_at,
                        None,
                        None,
                        timeout_seconds,
                        output_dir,
                        json.dumps(request_payload),
                        None,
                        None,
                        None,
                    ),
                )
                conn.commit()
                return JobResponse(
                    job_id=job_id,
                    job_type=job_type,
                    model_id=model_id,
                    revision=revision,
                    status=status,
                    domain_status=None,
                    created_at=created_at,
                    started_at=None,
                    completed_at=None,
                    timeout_seconds=timeout_seconds,
                    output_dir=output_dir,
                    error_message=None,
                    error_code=None,
                )
            finally:
                conn.close()

    def get_job(self, job_id: str) -> Optional[JobResponse]:
        """Fetch job by ID."""
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
                row = cursor.fetchone()
                if not row:
                    return None
                return JobResponse(
                    job_id=row["job_id"],
                    job_type=JobType(row["job_type"]),
                    model_id=row["model_id"],
                    revision=row["revision"],
                    status=JobStatus(row["status"]),
                    domain_status=row["domain_status"],
                    created_at=row["created_at"],
                    started_at=row["started_at"],
                    completed_at=row["completed_at"],
                    timeout_seconds=row["timeout_seconds"],
                    output_dir=row["output_dir"],
                    error_message=row["error_message"],
                    error_code=row["error_code"],
                )
            finally:
                conn.close()

    def get_job_full(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Fetch job with full request and result payloads."""
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
                row = cursor.fetchone()
                if not row:
                    return None
                data = dict(row)
                if data.get("request_payload"):
                    data["request_payload"] = json.loads(data["request_payload"])
                if data.get("result_payload"):
                    data["result_payload"] = json.loads(data["result_payload"])
                return data
            finally:
                conn.close()

    def update_job_status(
        self,
        job_id: str,
        status: JobStatus,
        started_at: Optional[str] = None,
        completed_at: Optional[str] = None,
        domain_status: Optional[str] = None,
        output_dir: Optional[str] = None,
        result_payload: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        error_code: Optional[str] = None,
    ) -> None:
        """Update job lifecycle status and optional payloads."""
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                updates: List[str] = ["status = ?"]
                params: List[Any] = [status.value]

                if started_at is not None:
                    updates.append("started_at = ?")
                    params.append(started_at)
                if completed_at is not None:
                    updates.append("completed_at = ?")
                    params.append(completed_at)
                if domain_status is not None:
                    updates.append("domain_status = ?")
                    params.append(domain_status)
                if output_dir is not None:
                    updates.append("output_dir = ?")
                    params.append(output_dir)
                if result_payload is not None:
                    updates.append("result_payload = ?")
                    params.append(json.dumps(result_payload))
                if error_message is not None:
                    updates.append("error_message = ?")
                    params.append(error_message)
                if error_code is not None:
                    updates.append("error_code = ?")
                    params.append(error_code)

                params.append(job_id)
                query = f"UPDATE jobs SET {', '.join(updates)} WHERE job_id = ?"
                cursor.execute(query, params)
                conn.commit()
            finally:
                conn.close()

    def insert_event(
        self,
        job_id: str,
        phase: str,
        status: str,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> JobEvent:
        """Insert progress event into database."""
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT COALESCE(MAX(sequence), 0) + 1 FROM job_events WHERE job_id = ?",
                    (job_id,),
                )
                seq = cursor.fetchone()[0]
                timestamp = datetime.now(timezone.utc).isoformat()
                details_json = json.dumps(details) if details is not None else None

                cursor.execute(
                    """
                    INSERT INTO job_events (
                        job_id, sequence, timestamp, phase, status, message, error_code, details
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (job_id, seq, timestamp, phase, status, message, error_code, details_json),
                )
                event_id = cursor.lastrowid
                conn.commit()

                return JobEvent(
                    event_id=int(event_id or 0),
                    job_id=job_id,
                    sequence=seq,
                    timestamp=timestamp,
                    phase=phase,
                    status=status,
                    message=message,
                    error_code=error_code,
                    details=details,
                )
            finally:
                conn.close()

    def get_events(self, job_id: str, from_event_id: int = 0) -> List[JobEvent]:
        """Fetch past events for a job starting from from_event_id."""
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT event_id, job_id, sequence, timestamp, phase, status, message, error_code, details
                    FROM job_events
                    WHERE job_id = ? AND event_id > ?
                    ORDER BY sequence ASC
                    """,
                    (job_id, from_event_id),
                )
                events = []
                for row in cursor.fetchall():
                    details = json.loads(row["details"]) if row["details"] else None
                    events.append(
                        JobEvent(
                            event_id=row["event_id"],
                            job_id=row["job_id"],
                            sequence=row["sequence"],
                            timestamp=row["timestamp"],
                            phase=row["phase"],
                            status=row["status"],
                            message=row["message"],
                            error_code=row["error_code"],
                            details=details,
                        )
                    )
                return events
            finally:
                conn.close()

    def mark_interrupted_jobs_as_failed(self, reason: str) -> List[str]:
        """Mark any RUNNING or QUEUED jobs on server restart as FAILED."""
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT job_id FROM jobs WHERE status IN (?, ?)",
                    (JobStatus.QUEUED.value, JobStatus.RUNNING.value),
                )
                interrupted_ids = [row[0] for row in cursor.fetchall()]
                now_iso = datetime.now(timezone.utc).isoformat()

                for jid in interrupted_ids:
                    cursor.execute(
                        """
                        UPDATE jobs
                        SET status = ?, completed_at = ?, error_message = ?, error_code = ?
                        WHERE job_id = ?
                        """,
                        (JobStatus.FAILED.value, now_iso, reason, "SERVER_RESTART", jid),
                    )
                    # Insert failure event
                    cursor.execute(
                        "SELECT COALESCE(MAX(sequence), 0) + 1 FROM job_events WHERE job_id = ?",
                        (jid,),
                    )
                    seq = cursor.fetchone()[0]
                    cursor.execute(
                        """
                        INSERT INTO job_events (
                            job_id, sequence, timestamp, phase, status, message, error_code
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (jid, seq, now_iso, "SYSTEM", "FAILED", reason, "SERVER_RESTART"),
                    )

                conn.commit()
                return interrupted_ids
            finally:
                conn.close()
