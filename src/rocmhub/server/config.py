"""Configuration for ROCmHub local backend server."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field


class ServerConfig(BaseModel):
    """Configuration settings for ROCmHub FastAPI server."""

    host: str = Field(default="127.0.0.1", description="Host address to bind the server")
    port: int = Field(default=8000, description="Port number to bind the server")
    db_path: Path = Field(
        default_factory=lambda: Path(
            os.environ.get("ROCMHUB_DB_PATH", Path.home() / ".rocmhub" / "jobs.db")
        ).resolve(),
        description="Path to SQLite database file",
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
        default_factory=lambda: ["http://127.0.0.1:3000", "http://localhost:3000"],
        description="Allowed CORS origins for local web development",
    )
