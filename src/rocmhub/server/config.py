"""Configuration for ROCmHub local backend server."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field


def _data_dir() -> Path:
    """Return the durable server data directory, without creating it on import."""
    return Path(os.environ.get("ROCMHUB_DATA_DIR", Path.home() / ".rocmhub")).expanduser().resolve()


def _db_path() -> Path:
    return Path(os.environ.get("ROCMHUB_DB_PATH", _data_dir() / "jobs.db")).expanduser().resolve()


class ServerConfig(BaseModel):
    """Configuration settings for ROCmHub FastAPI server."""

    host: str = Field(default="127.0.0.1", description="Host address to bind the server")
    port: int = Field(
        default_factory=lambda: int(os.environ.get("PORT", os.environ.get("ROCMHUB_PORT", "8000"))),
        description="Port number to bind the server",
    )
    data_dir: Path = Field(default_factory=_data_dir, description="Durable server data directory")
    db_path: Path = Field(
        default_factory=_db_path,
        description="Path to SQLite database file",
    )
    artifact_storage_dir: Path = Field(
        default_factory=lambda: _data_dir() / "agent-artifacts",
        description="Durable, bounded storage for small remote Agent artifacts",
    )
    max_agent_artifact_bytes: int = Field(
        default=262_144, ge=1_024, le=1_048_576, description="Maximum bytes per uploaded Agent artifact"
    )
    allowed_workspaces: List[Path] = Field(
        default_factory=lambda: [
            Path.cwd().resolve(),
            (Path.cwd() / "builds").resolve(),
            (Path.cwd() / "optimizations").resolve(),
            Path("/tmp").resolve(),
        ],
        description="Allowed filesystem roots for output directories",
    )
    max_queue_size: int = Field(
        default=50, ge=1, le=500, description="Maximum number of queued jobs allowed"
    )
    max_request_bytes: int = Field(
        default=1_048_576, description="Maximum HTTP request payload size in bytes (1MB)"
    )
    default_job_timeout_seconds: int = Field(
        default=1800, ge=10, le=86400, description="Default job timeout in seconds"
    )
    cors_origins: List[str] = Field(
        default_factory=lambda: [v.strip() for v in os.environ.get("ROCMHUB_CORS_ORIGINS", "http://127.0.0.1:3000,http://localhost:3000,http://127.0.0.1:5173,http://localhost:5173").split(",") if v.strip()],
        description="Allowed CORS origins for local web development",
    )
    agent_heartbeat_timeout_seconds: int = Field(default_factory=lambda: int(os.environ.get("ROCMHUB_AGENT_HEARTBEAT_TIMEOUT_SECONDS", "45")), ge=5, le=3600)
    agent_retention_seconds: int = Field(default_factory=lambda: int(os.environ.get("ROCMHUB_AGENT_RETENTION_SECONDS", "86400")), ge=60, le=31_536_000)
    agent_token_hashes: List[str] = Field(default_factory=lambda: [hashlib.sha256(token.encode()).hexdigest() for token in os.environ.get("ROCMHUB_AGENT_TOKENS", "rocmhub-dev-agent-token").split(",") if token])
