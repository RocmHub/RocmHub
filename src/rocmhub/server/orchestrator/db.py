"""Thread-safe SQLite database access layer for Job Orchestrator."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rocmhub.server.orchestrator.migrations import apply_migrations
from rocmhub.server.orchestrator.models import (
    AgentCapabilities,
    AgentPresenceResponse,
    AgentResponse,
    AgentStatus,
    JobEvent,
    JobResponse,
    JobStatus,
    JobType,
)
from rocmhub.server.privacy import sanitize_public_value


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

    @staticmethod
    def _clear_terminal_local_paths(conn: sqlite3.Connection, job_id: str) -> None:
        """Keep durable job inputs/evidence while dropping no-longer-needed local paths."""
        row = conn.execute("SELECT request_payload FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if not row:
            return
        try:
            request_payload = json.loads(row["request_payload"] or "{}")
        except (TypeError, ValueError):
            request_payload = {}
        safe_request = sanitize_public_value(request_payload)
        conn.execute(
            "UPDATE jobs SET output_dir=NULL, request_payload=? WHERE job_id=?",
            (json.dumps(safe_request), job_id),
        )

    @staticmethod
    def _job_response(row: sqlite3.Row) -> JobResponse:
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
            error_message=sanitize_public_value(row["error_message"]),
            error_code=row["error_code"],
            agent_id=row["agent_id"] if "agent_id" in row.keys() else None,
            claimed_at=row["claimed_at"] if "claimed_at" in row.keys() else None,
            heartbeat_at=row["heartbeat_at"] if "heartbeat_at" in row.keys() else None,
            attempt=row["attempt"] if "attempt" in row.keys() else 0,
        )

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
                return self.get_job(job_id) or JobResponse(
                    job_id=job_id,
                    job_type=job_type,
                    model_id=model_id,
                    revision=revision,
                    status=status,
                    created_at=created_at,
                    timeout_seconds=timeout_seconds,
                    output_dir=output_dir,
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
                return self._job_response(row)
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
        revision: Optional[str] = None,
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
                    params.append(json.dumps(sanitize_public_value(result_payload)))
                if error_message is not None:
                    updates.append("error_message = ?")
                    params.append(sanitize_public_value(error_message))
                if error_code is not None:
                    updates.append("error_code = ?")
                    params.append(error_code)
                if revision is not None:
                    updates.append("revision = ?")
                    params.append(revision)

                params.append(job_id)
                query = f"UPDATE jobs SET {', '.join(updates)} WHERE job_id = ?"
                cursor.execute(query, params)
                if status in {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED}:
                    self._clear_terminal_local_paths(conn, job_id)
                conn.commit()
            finally:
                conn.close()

    def upsert_agent(
        self, agent_id: str, capabilities: AgentCapabilities, token_hash: str, retention_seconds: int
    ) -> AgentResponse:
        """Persist an opaque operational ID and reported capabilities, not host identity."""
        self.prune_stale_agents(retention_seconds)
        has_amd_gpu = capabilities.amd_gpu_count > 0
        product_name = "AMD Compute Agent" if has_amd_gpu else "Preparation Agent"
        safe_capabilities = AgentCapabilities.model_validate(sanitize_public_value(capabilities.model_dump()))
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute(
                    """INSERT INTO agents(agent_id,name,hostname,status,capabilities,token_hash,last_seen,created_at)
                VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(agent_id) DO UPDATE SET name=excluded.name,hostname=excluded.hostname,status='ONLINE',capabilities=excluded.capabilities,token_hash=excluded.token_hash,last_seen=excluded.last_seen""",
                    (
                        agent_id,
                        product_name,
                        "",
                        AgentStatus.ONLINE.value,
                        safe_capabilities.model_dump_json(),
                        token_hash,
                        now,
                        now,
                    ),
                )
                conn.commit()
                return AgentResponse(
                    agent_id=agent_id,
                    status=AgentStatus.ONLINE,
                    capabilities=safe_capabilities,
                    last_seen=now,
                    created_at=now,
                )
            finally:
                conn.close()

    def revoke_agent_token(self, token_hash: str) -> None:
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO revoked_agent_tokens(token_hash,revoked_at) VALUES(?,?)",
                    (token_hash, datetime.now(timezone.utc).isoformat()),
                )
                conn.commit()
            finally:
                conn.close()

    def is_agent_token_revoked(self, token_hash: str) -> bool:
        with self._lock:
            conn = self._get_connection()
            try:
                return (
                    conn.execute("SELECT 1 FROM revoked_agent_tokens WHERE token_hash=?", (token_hash,)).fetchone()
                    is not None
                )
            finally:
                conn.close()

    def list_agents(self) -> List[AgentResponse]:
        with self._lock:
            conn = self._get_connection()
            try:
                rows = conn.execute("SELECT * FROM agents ORDER BY last_seen DESC").fetchall()
                return [
                    AgentResponse(
                        agent_id=r["agent_id"],
                        status=AgentStatus(r["status"]),
                        capabilities=AgentCapabilities.model_validate_json(r["capabilities"]),
                        last_seen=r["last_seen"],
                        created_at=r["created_at"],
                    )
                    for r in rows
                ]
            finally:
                conn.close()

    def get_agent(self, agent_id: str) -> Optional[AgentResponse]:
        with self._lock:
            conn = self._get_connection()
            try:
                row = conn.execute("SELECT * FROM agents WHERE agent_id = ?", (agent_id,)).fetchone()
                if not row:
                    return None
                return AgentResponse(
                    agent_id=row["agent_id"],
                    status=AgentStatus(row["status"]),
                    capabilities=AgentCapabilities.model_validate_json(row["capabilities"]),
                    last_seen=row["last_seen"],
                    created_at=row["created_at"],
                )
            finally:
                conn.close()

    def list_active_agent_presence(self, timeout_seconds: int) -> List[AgentPresenceResponse]:
        """Return fresh Agent presence without exposing IDs or machine identity."""
        cutoff = (datetime.now(timezone.utc) - timedelta(seconds=timeout_seconds)).isoformat()
        with self._lock:
            conn = self._get_connection()
            try:
                rows = conn.execute(
                    """
                    SELECT a.capabilities,
                           EXISTS(
                               SELECT 1 FROM jobs j
                               WHERE j.agent_id = a.agent_id AND j.status = ?
                           ) AS is_busy
                    FROM agents a
                    WHERE a.last_seen >= ?
                    ORDER BY a.last_seen DESC
                    """,
                    (JobStatus.RUNNING.value, cutoff),
                ).fetchall()
                presence: List[AgentPresenceResponse] = []
                for row in rows:
                    capabilities = AgentCapabilities.model_validate_json(row["capabilities"])
                    has_amd_gpu = capabilities.amd_gpu_count > 0
                    gpu_names = capabilities.gpu_names if has_amd_gpu else []
                    label = (gpu_names[0] or "AMD Compute Agent") if has_amd_gpu else "Preparation Agent"
                    presence.append(
                        AgentPresenceResponse(
                            label=label,
                            status="BUSY" if row["is_busy"] else "ONLINE",
                            has_amd_gpu=has_amd_gpu,
                            gpu_names=gpu_names,
                            rocm_detected=capabilities.rocm_detected,
                        )
                    )
                return presence
            finally:
                conn.close()

    def prune_stale_agents(self, retention_seconds: int) -> int:
        """Prune expired presence rows, retaining any Agent that owns a running job."""
        cutoff = (datetime.now(timezone.utc) - timedelta(seconds=retention_seconds)).isoformat()
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE")
                cursor = conn.execute(
                    """
                    DELETE FROM agents
                    WHERE last_seen < ?
                      AND NOT EXISTS (
                          SELECT 1 FROM jobs j
                          WHERE j.agent_id = agents.agent_id AND j.status = ?
                      )
                    """,
                    (cutoff, JobStatus.RUNNING.value),
                )
                conn.commit()
                return int(cursor.rowcount)
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def heartbeat_agent(self, agent_id: str) -> bool:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute(
                    "UPDATE agents SET last_seen=?, status='ONLINE', name='Preparation Agent', hostname='' WHERE agent_id=?",
                    (now, agent_id),
                )
                conn.execute(
                    "UPDATE jobs SET heartbeat_at=? WHERE agent_id=? AND status=?",
                    (now, agent_id, JobStatus.RUNNING.value),
                )
                conn.commit()
                return cur.rowcount == 1
            finally:
                conn.close()

    def claim_next_job(
        self, agent_id: str, capability_names: List[str], can_execute_amd: bool
    ) -> Optional[Tuple[JobResponse, Dict[str, Any]]]:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("BEGIN IMMEDIATE")
                rows = conn.execute(
                    "SELECT * FROM jobs WHERE status=? ORDER BY created_at ASC", (JobStatus.QUEUED.value,)
                ).fetchall()
                for row in rows:
                    payload = json.loads(row["request_payload"])
                    job_type = row["job_type"]
                    required_amd = job_type in ("AMD_EXECUTION", "BENCHMARK") or bool(
                        payload.get("require_amd_execution")
                    )
                    eligible = (
                        job_type == JobType.PREPARE_MODEL_FOR_AMD.value
                        and "PREPARE_MODEL_FOR_AMD" in capability_names
                        and not required_amd
                    ) or (required_amd and can_execute_amd)
                    if not eligible:
                        continue
                    cur = conn.execute(
                        "UPDATE jobs SET status=?,agent_id=?,claimed_at=?,heartbeat_at=?,started_at=?,attempt=attempt+1 WHERE job_id=? AND status=?",
                        (JobStatus.RUNNING.value, agent_id, now, now, now, row["job_id"], JobStatus.QUEUED.value),
                    )
                    if cur.rowcount:
                        claimed = conn.execute("SELECT * FROM jobs WHERE job_id=?", (row["job_id"],)).fetchone()
                        conn.commit()
                        return self._job_response(claimed), payload
                conn.commit()
                return None
            finally:
                conn.close()

    def agent_owns_running_job(self, agent_id: str, job_id: str, attempt: Optional[int] = None) -> bool:
        with self._lock:
            conn = self._get_connection()
            try:
                query = "SELECT 1 FROM jobs WHERE job_id=? AND agent_id=? AND status=?"
                params: tuple[Any, ...] = (job_id, agent_id, JobStatus.RUNNING.value)
                if attempt is not None:
                    query += " AND attempt=?"
                    params += (attempt,)
                return conn.execute(query, params).fetchone() is not None
            finally:
                conn.close()

    def finish_agent_job(
        self,
        agent_id: str,
        job_id: str,
        attempt: Optional[int],
        status: JobStatus,
        *,
        completed_at: str,
        domain_status: Optional[str] = None,
        result_payload: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        error_code: Optional[str] = None,
        revision: Optional[str] = None,
    ) -> bool:
        """Fence a remote agent's terminal update to its current claim attempt."""
        with self._lock:
            conn = self._get_connection()
            try:
                updates = ["status=?", "completed_at=?"]
                params: List[Any] = [status.value, completed_at]
                for column, value in (
                    ("domain_status", domain_status),
                    (
                        "result_payload",
                        json.dumps(sanitize_public_value(result_payload)) if result_payload is not None else None,
                    ),
                    ("error_message", sanitize_public_value(error_message) if error_message is not None else None),
                    ("error_code", error_code),
                    ("revision", revision),
                ):
                    if value is not None:
                        updates.append(f"{column}=?")
                        params.append(value)
                query = f"UPDATE jobs SET {', '.join(updates)} WHERE job_id=? AND agent_id=? AND status=?"
                params.extend((job_id, agent_id, JobStatus.RUNNING.value))
                if attempt is not None:
                    query += " AND attempt=?"
                    params.append(attempt)
                cursor = conn.execute(query, params)
                if cursor.rowcount == 1 and status in {
                    JobStatus.SUCCEEDED,
                    JobStatus.FAILED,
                    JobStatus.CANCELLED,
                }:
                    self._clear_terminal_local_paths(conn, job_id)
                conn.commit()
                return cursor.rowcount == 1
            finally:
                conn.close()

    def release_stale_claims(self, timeout_seconds: int) -> List[str]:
        from datetime import timedelta

        cutoff = (datetime.now(timezone.utc) - timedelta(seconds=timeout_seconds)).isoformat()
        with self._lock:
            conn = self._get_connection()
            try:
                rows = conn.execute(
                    "SELECT job_id FROM jobs WHERE status=? AND heartbeat_at IS NOT NULL AND heartbeat_at < ?",
                    (JobStatus.RUNNING.value, cutoff),
                ).fetchall()
                ids = [r[0] for r in rows]
                conn.execute("UPDATE agents SET status=? WHERE last_seen < ?", (AgentStatus.OFFLINE.value, cutoff))
                for job_id in ids:
                    conn.execute(
                        "UPDATE jobs SET status=?,agent_id=NULL,claimed_at=NULL,heartbeat_at=NULL,error_message=?,error_code=? WHERE job_id=?",
                        (
                            JobStatus.QUEUED.value,
                            "Agent heartbeat timed out; job released for another eligible agent.",
                            "AGENT_HEARTBEAT_TIMEOUT",
                            job_id,
                        ),
                    )
                conn.commit()
                return ids
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
                safe_message = sanitize_public_value(message)
                safe_details = sanitize_public_value(details) if details is not None else None
                details_json = json.dumps(safe_details) if safe_details is not None else None

                cursor.execute(
                    """
                    INSERT INTO job_events (
                        job_id, sequence, timestamp, phase, status, message, error_code, details
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (job_id, seq, timestamp, phase, status, safe_message, error_code, details_json),
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
                    message=safe_message,
                    error_code=error_code,
                    details=safe_details,
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
                    details = sanitize_public_value(json.loads(row["details"])) if row["details"] else None
                    events.append(
                        JobEvent(
                            event_id=row["event_id"],
                            job_id=row["job_id"],
                            sequence=row["sequence"],
                            timestamp=row["timestamp"],
                            phase=row["phase"],
                            status=row["status"],
                            message=sanitize_public_value(row["message"]),
                            error_code=row["error_code"],
                            details=details,
                        )
                    )
                return events
            finally:
                conn.close()

    def mark_interrupted_jobs_as_failed(self, reason: str) -> List[str]:
        """Mark any RUNNING or QUEUED jobs on server restart as FAILED with explicit error codes."""
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT job_id, status, agent_id FROM jobs WHERE status IN (?, ?)",
                    (JobStatus.QUEUED.value, JobStatus.RUNNING.value),
                )
                interrupted_rows = cursor.fetchall()
                interrupted_ids = [row[0] for row in interrupted_rows]
                now_iso = datetime.now(timezone.utc).isoformat()

                for jid, st, agent_id in interrupted_rows:
                    if agent_id:
                        cursor.execute(
                            "UPDATE jobs SET status=?, agent_id=NULL, claimed_at=NULL, heartbeat_at=NULL WHERE job_id=?",
                            (JobStatus.QUEUED.value, jid),
                        )
                        cursor.execute("SELECT COALESCE(MAX(sequence), 0) + 1 FROM job_events WHERE job_id = ?", (jid,))
                        seq = cursor.fetchone()[0]
                        cursor.execute(
                            "INSERT INTO job_events(job_id,sequence,timestamp,phase,status,message,error_code) VALUES(?,?,?,?,?,?,?)",
                            (
                                jid,
                                seq,
                                now_iso,
                                "SYSTEM",
                                "QUEUED",
                                "Server restarted; remote agent claim released for safe reassignment.",
                                "AGENT_CLAIM_RELEASED_ON_RESTART",
                            ),
                        )
                        continue
                    if st == JobStatus.RUNNING.value:
                        err_code = "EXECUTION_INTERRUPTED_BY_RESTART"
                        err_msg = (
                            f"Job execution was interrupted by server restart: {reason}. "
                            "Working directory may contain partial build artifacts."
                        )
                    else:
                        err_code = "QUEUE_DISCARDED_ON_RESTART"
                        err_msg = f"Job was queued but not started before server restart: {reason}."
                    err_msg = sanitize_public_value(err_msg)

                    cursor.execute(
                        """
                        UPDATE jobs
                        SET status = ?, completed_at = ?, error_message = ?, error_code = ?
                        WHERE job_id = ?
                        """,
                        (JobStatus.FAILED.value, now_iso, err_msg, err_code, jid),
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
                        (jid, seq, now_iso, "SYSTEM", "FAILED", err_msg, err_code),
                    )
                    self._clear_terminal_local_paths(conn, str(jid))

                conn.commit()
                return interrupted_ids
            finally:
                conn.close()

    def list_jobs(
        self,
        limit: int = 20,
        offset: int = 0,
        status: Optional[JobStatus] = None,
        job_type: Optional[JobType] = None,
    ) -> Tuple[List[JobResponse], int]:
        """List jobs with pagination, filtering, and newest first."""
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                where_clauses: List[str] = []
                params: List[Any] = []

                if status is not None:
                    where_clauses.append("status = ?")
                    params.append(status.value)
                if job_type is not None:
                    where_clauses.append("job_type = ?")
                    params.append(job_type.value)

                where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

                # Count total matching rows
                count_query = f"SELECT COUNT(*) FROM jobs {where_sql}"
                cursor.execute(count_query, params)
                total = cursor.fetchone()[0]

                # Select paginated results
                select_query = f"SELECT * FROM jobs {where_sql} ORDER BY created_at DESC LIMIT ? OFFSET ?"
                cursor.execute(select_query, params + [limit, offset])
                rows = cursor.fetchall()

                jobs: List[JobResponse] = []
                for row in rows:
                    jobs.append(
                        JobResponse(
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
                    )
                return jobs, total
            finally:
                conn.close()
