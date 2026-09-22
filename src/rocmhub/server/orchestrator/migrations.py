"""SQLite schema migrations for ROCmHub Job Orchestrator."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import List, Tuple

MIGRATION_1_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL,
    description TEXT
);

CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    job_type TEXT NOT NULL,
    model_id TEXT NOT NULL,
    revision TEXT,
    status TEXT NOT NULL,
    domain_status TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    timeout_seconds INTEGER NOT NULL,
    output_dir TEXT,
    request_payload TEXT NOT NULL,
    result_payload TEXT,
    error_message TEXT,
    error_code TEXT
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);

CREATE TABLE IF NOT EXISTS job_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    phase TEXT NOT NULL,
    status TEXT NOT NULL,
    message TEXT NOT NULL,
    error_code TEXT,
    details TEXT,
    FOREIGN KEY(job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_job_events_job_seq ON job_events(job_id, sequence);
CREATE INDEX IF NOT EXISTS idx_job_events_job_id ON job_events(job_id);
"""

MIGRATIONS: List[Tuple[int, str, str]] = [
    (1, "001_initial_schema", MIGRATION_1_SQL),
]


def apply_migrations(conn: sqlite3.Connection) -> None:
    """Apply any pending migrations to the SQLite database."""
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL,
            description TEXT
        )
        """
    )
    conn.commit()

    cursor.execute("SELECT version FROM schema_version")
    applied_versions = {row[0] for row in cursor.fetchall()}

    for version, description, sql in MIGRATIONS:
        if version not in applied_versions:
            cursor.executescript(sql)
            now_iso = datetime.now(timezone.utc).isoformat()
            cursor.execute(
                "INSERT INTO schema_version (version, applied_at, description) VALUES (?, ?, ?)",
                (version, now_iso, description),
            )
            conn.commit()
