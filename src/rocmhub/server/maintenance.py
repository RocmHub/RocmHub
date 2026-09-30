"""Bounded, dry-run-first cleanup for pre-public ROCmHub development data."""

from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

_DEV_TEST_MARKER = re.compile(
    r"(?:mac[-_]dev[-_]agent|mac[-_]mini|remote[-_]e2e(?:[-_][a-z0-9]+)*|"
    r"rocmhub[-_](?:live[-_])?(?:smoke|demo|test)|"
    r"(?:^|[/_. -])(?:test|tests|fixture|dummy|example)(?:[/_. -]|$))",
    re.IGNORECASE,
)
_PHYSICAL_DOMAIN_STATUSES = {"EXECUTED", "VALIDATED", "BENCHMARKED"}
_JOB_ID = re.compile(r"^job_[a-zA-Z0-9_-]{1,64}$")


class MaintenanceSafetyError(RuntimeError):
    """Raised when the target or the records cannot be identified safely."""


@dataclass
class CleanupPlan:
    database_path: Path
    data_dir: Path
    artifact_dir: Path
    agent_ids: List[str] = field(default_factory=list)
    job_ids: List[str] = field(default_factory=list)
    jobs_by_type: Dict[str, int] = field(default_factory=dict)
    event_count: int = 0
    result_count: int = 0
    artifact_record_count: int = 0
    artifact_file_count: int = 0
    ambiguous_agent_count: int = 0
    ambiguous_job_count: int = 0
    in_progress_job_count: int = 0
    protected_agent_count: int = 0
    physical_execution_job_ids: List[str] = field(default_factory=list)
    unsafe_artifact_job_ids: List[str] = field(default_factory=list)

    @property
    def safe_to_execute(self) -> bool:
        return not any(
            (
                self.ambiguous_agent_count,
                self.ambiguous_job_count,
                self.in_progress_job_count,
                self.protected_agent_count,
                self.physical_execution_job_ids,
                self.unsafe_artifact_job_ids,
            )
        )


def _parse_datetime(value: str) -> Optional[datetime]:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _walk_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_strings(item)
    elif isinstance(value, str):
        yield value


def _is_development_job(
    job: sqlite3.Row,
    owner_identity: str,
    event_values: Iterable[Tuple[str, str, str, str]],
    request_payload: Any,
) -> bool:
    values: List[str] = [
        str(job["job_id"] or ""),
        str(job["model_id"] or ""),
        str(job["output_dir"] or ""),
        owner_identity,
    ]
    values.extend(_walk_strings(request_payload))
    for phase, status, message, details in event_values:
        values.extend((phase, status, message, details))
    return any(_DEV_TEST_MARKER.search(value) for value in values)


def _has_physical_execution_evidence(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).lower()
            if normalized in {"amd_validated", "physical_amd_execution", "physical_execution"} and item is True:
                return True
            if normalized in {"execution_status", "domain_status", "validation_status", "benchmark_status"}:
                if isinstance(item, str) and item.upper() in _PHYSICAL_DOMAIN_STATUSES:
                    return True
            if _has_physical_execution_evidence(item):
                return True
    elif isinstance(value, list):
        return any(_has_physical_execution_evidence(item) for item in value)
    return False


def _sqlite_readonly(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise MaintenanceSafetyError(f"Configured SQLite database does not exist: {path}")
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, timeout=30)
    connection.row_factory = sqlite3.Row
    return connection


def _safe_artifact_file_count(root: Path, job_id: str) -> Tuple[int, bool]:
    if not _JOB_ID.fullmatch(job_id):
        return 0, False
    raw_target = root / job_id
    if raw_target.is_symlink():
        return 0, False
    target = raw_target.resolve()
    if target.parent != root.resolve():
        return 0, False
    if not target.exists():
        return 0, True
    if target.is_symlink() or not target.is_dir():
        return 0, False
    count = 0
    for current, directories, files in os.walk(target, followlinks=False):
        current_path = Path(current)
        for directory in directories:
            item = current_path / directory
            if item.is_symlink():
                return count, False
        for filename in files:
            item = current_path / filename
            if item.is_symlink() or not item.is_file():
                return count, False
            count += 1
    return count, True


def inspect_development_data(
    database_path: Path,
    data_dir: Path,
    artifact_dir: Path,
    agent_retention_seconds: int,
    *,
    now: Optional[datetime] = None,
) -> CleanupPlan:
    """Build a read-only removal plan; unclassified records block confirmation."""
    database_path = database_path.expanduser().resolve()
    data_dir = data_dir.expanduser().resolve()
    artifact_dir = artifact_dir.expanduser().resolve()
    if not data_dir.is_dir():
        raise MaintenanceSafetyError(f"Configured ROCmHub data directory does not exist: {data_dir}")
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    retention_cutoff = now - timedelta(seconds=agent_retention_seconds)
    plan = CleanupPlan(database_path=database_path, data_dir=data_dir, artifact_dir=artifact_dir)

    connection = _sqlite_readonly(database_path)
    try:
        tables = {
            str(row[0])
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        required = {"agents", "jobs", "job_events"}
        if not required.issubset(tables):
            raise MaintenanceSafetyError("Configured SQLite database is not a ROCmHub orchestrator database")
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        if not integrity or integrity[0] != "ok":
            raise MaintenanceSafetyError("Configured SQLite database failed its integrity check")

        agents = connection.execute("SELECT agent_id, name, hostname, last_seen FROM agents").fetchall()
        agent_ids: Set[str] = set()
        agent_identity: Dict[str, str] = {}
        busy_agent_ids: Set[str] = set()
        for agent in agents:
            agent_id = str(agent["agent_id"])
            last_seen = _parse_datetime(str(agent["last_seen"]))
            if last_seen is None:
                plan.ambiguous_agent_count += 1
                continue
            identity = " ".join((str(agent["name"] or ""), str(agent["hostname"] or "")))
            agent_identity[agent_id] = identity
            if last_seen < retention_cutoff:
                agent_ids.add(agent_id)
        plan.agent_ids = sorted(agent_ids)

        jobs = connection.execute(
            """SELECT job_id, job_type, model_id, status, domain_status, output_dir,
                      request_payload, result_payload, agent_id
               FROM jobs ORDER BY created_at ASC"""
        ).fetchall()
        event_rows = connection.execute(
            "SELECT job_id, phase, status, message, details FROM job_events ORDER BY event_id ASC"
        ).fetchall()
        events_by_job: Dict[str, List[Tuple[str, str, str, str]]] = {}
        for event in event_rows:
            details = event["details"] or ""
            events_by_job.setdefault(str(event["job_id"]), []).append(
                (
                    str(event["phase"] or ""),
                    str(event["status"] or ""),
                    str(event["message"] or ""),
                    str(details),
                )
            )

        job_ids: List[str] = []
        safe_jobs_by_type: Dict[str, int] = {}
        for job in jobs:
            job_id = str(job["job_id"])
            try:
                request_payload = json.loads(job["request_payload"] or "{}")
                result_payload = json.loads(job["result_payload"] or "null")
            except (TypeError, ValueError):
                plan.ambiguous_job_count += 1
                continue
            if _has_physical_execution_evidence(
                {"domain_status": job["domain_status"], "result": result_payload}
            ):
                plan.physical_execution_job_ids.append(job_id)
            for phase, status, message, details in events_by_job.get(job_id, []):
                if (
                    phase.upper() in _PHYSICAL_DOMAIN_STATUSES
                    or status.upper() in _PHYSICAL_DOMAIN_STATUSES
                    or _has_physical_execution_evidence({"message": message, "details": details})
                ):
                    plan.physical_execution_job_ids.append(job_id)
                    break
            owner_id = str(job["agent_id"] or "")
            if str(job["status"]) == "RUNNING":
                plan.in_progress_job_count += 1
                if owner_id:
                    busy_agent_ids.add(owner_id)
            if not _is_development_job(
                job,
                agent_identity.get(owner_id, ""),
                events_by_job.get(job_id, []),
                request_payload,
            ):
                plan.ambiguous_job_count += 1
                continue
            if str(job["status"]) == "RUNNING":
                continue
            job_ids.append(job_id)
            job_type = str(job["job_type"])
            safe_jobs_by_type[job_type] = safe_jobs_by_type.get(job_type, 0) + 1
            if result_payload is not None:
                plan.result_count += 1
                artifacts = result_payload.get("server_artifacts", []) if isinstance(result_payload, dict) else []
                if isinstance(artifacts, list):
                    plan.artifact_record_count += len(artifacts)
            file_count, safe = _safe_artifact_file_count(artifact_dir, job_id)
            plan.artifact_file_count += file_count
            if not safe:
                plan.unsafe_artifact_job_ids.append(job_id)

        plan.job_ids = sorted(job_ids)
        plan.jobs_by_type = safe_jobs_by_type
        if plan.job_ids:
            placeholders = ",".join("?" for _ in plan.job_ids)
            plan.event_count = int(
                connection.execute(
                    f"SELECT COUNT(*) FROM job_events WHERE job_id IN ({placeholders})", plan.job_ids
                ).fetchone()[0]
            )
        plan.protected_agent_count = len(agent_ids.intersection(busy_agent_ids))
    finally:
        connection.close()
    return plan


def _make_database_backup(database_path: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = database_path.parent / f"jobs-before-development-cleanup-{timestamp}-{uuid.uuid4().hex[:6]}.sqlite3"
    source = sqlite3.connect(str(database_path), timeout=30)
    destination = sqlite3.connect(str(backup_path), timeout=30)
    try:
        source.backup(destination)
        destination.commit()
    except Exception:
        destination.close()
        source.close()
        backup_path.unlink(missing_ok=True)
        raise
    destination.close()
    source.close()
    return backup_path


def execute_development_cleanup(plan: CleanupPlan, agent_retention_seconds: int) -> Path:
    """Back up the configured database, then delete only the inspected stale/test records."""
    if not plan.safe_to_execute:
        raise MaintenanceSafetyError("Cleanup plan is ambiguous or contains protected execution evidence")
    backup_path = _make_database_backup(plan.database_path)
    moved: List[Tuple[Path, Path]] = []
    staging_root = plan.artifact_dir / f".development-cleanup-{uuid.uuid4().hex}"
    database_committed = False
    try:
        if plan.artifact_dir.exists() and plan.job_ids:
            staging_root.mkdir()
            for job_id in plan.job_ids:
                raw_source = plan.artifact_dir / job_id
                if not raw_source.exists():
                    continue
                if raw_source.is_symlink():
                    raise MaintenanceSafetyError("Artifact directory changed after inspection")
                source = raw_source.resolve()
                if source.parent != plan.artifact_dir.resolve() or source.is_symlink():
                    raise MaintenanceSafetyError("Artifact directory changed after inspection")
                target = staging_root / job_id
                os.replace(source, target)
                moved.append((source, target))

        connection = sqlite3.connect(str(plan.database_path), timeout=30)
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("BEGIN IMMEDIATE")
            if plan.job_ids:
                placeholders = ",".join("?" for _ in plan.job_ids)
                current = {
                    str(row[0])
                    for row in connection.execute(
                        f"SELECT job_id FROM jobs WHERE job_id IN ({placeholders}) AND status='RUNNING'",
                        plan.job_ids,
                    ).fetchall()
                }
                if current:
                    raise MaintenanceSafetyError("A selected development job became active; cleanup was stopped")
                connection.execute(f"DELETE FROM job_events WHERE job_id IN ({placeholders})", plan.job_ids)
                deleted_jobs = connection.execute(f"DELETE FROM jobs WHERE job_id IN ({placeholders})", plan.job_ids)
                if deleted_jobs.rowcount != len(plan.job_ids):
                    raise MaintenanceSafetyError("Development jobs changed after inspection; cleanup was stopped")
            if plan.agent_ids:
                cutoff = (datetime.now(timezone.utc) - timedelta(seconds=agent_retention_seconds)).isoformat()
                deleted_agents = 0
                for agent_id in plan.agent_ids:
                    cursor = connection.execute(
                        """DELETE FROM agents
                           WHERE agent_id=? AND last_seen < ?
                             AND NOT EXISTS (
                                 SELECT 1 FROM jobs j
                                 WHERE j.agent_id=agents.agent_id AND j.status='RUNNING'
                             )""",
                        (agent_id, cutoff),
                    )
                    deleted_agents += int(cursor.rowcount)
                if deleted_agents != len(plan.agent_ids):
                    raise MaintenanceSafetyError("Agent presence changed after inspection; cleanup was stopped")
            connection.commit()
            database_committed = True
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

        if staging_root.exists():
            shutil.rmtree(staging_root)
        return backup_path
    except Exception:
        if not database_committed:
            for source, staged in reversed(moved):
                if staged.exists() and not source.exists():
                    os.replace(staged, source)
        if staging_root.exists() and not any(staging_root.iterdir()):
            staging_root.rmdir()
        raise
