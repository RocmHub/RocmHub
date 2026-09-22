"""Data contracts and schemas for Job Orchestrator."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class JobType(str, Enum):
    """Supported job types."""

    FORGE_BUILD = "FORGE_BUILD"
    ENGINEER = "ENGINEER"
    OPTIMIZATION = "OPTIMIZATION"


class JobStatus(str, Enum):
    """Lifecycle statuses for asynchronous jobs."""

    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class JobCreateRequest(BaseModel):
    """Request schema for initiating a new background job."""

    model_config = ConfigDict(extra="forbid")

    job_type: JobType = Field(..., description="Type of job to execute")
    model_id: str = Field(..., min_length=1, max_length=256, description="Hugging Face model ID")
    revision: Optional[str] = Field(default=None, max_length=128, description="Git commit SHA, branch, or tag")
    target_gpu: Optional[str] = Field(default=None, max_length=64, description="Target GPU device/gfx")
    precision: Optional[str] = Field(default=None, description="Requested precision (e.g. fp16, bf16, fp32)")
    recipe: Optional[str] = Field(default=None, description="Requested build recipe")
    objective: Optional[str] = Field(default=None, description="Optimization objective (e.g. THROUGHPUT, PREPARE_AMD)")
    strategies: Optional[List[str]] = Field(default=None, description="Optimization strategies to explore")
    allow_full_weights: bool = Field(
        default=False, description="Whether to permit downloading full multi-gigabyte weight tensors"
    )
    output_dir: Optional[str] = Field(
        default=None, max_length=512, description="Target filesystem directory for build/run outputs"
    )
    timeout_seconds: Optional[int] = Field(
        default=None, ge=10, le=86400, description="Job execution timeout in seconds"
    )
    max_candidates: Optional[int] = Field(
        default=None, ge=1, le=20, description="Max candidates for optimization jobs"
    )
    max_attempts: Optional[int] = Field(
        default=None, ge=1, le=20, description="Max retry attempts for engineer jobs"
    )
    max_disk_gb: Optional[int] = Field(
        default=None, ge=1, le=500, description="Max disk allowance in GB"
    )


class JobResponse(BaseModel):
    """Public representation of an asynchronous job."""

    model_config = ConfigDict(from_attributes=True)

    job_id: str
    job_type: JobType
    model_id: str
    revision: Optional[str] = None
    status: JobStatus
    domain_status: Optional[str] = None
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    timeout_seconds: int
    output_dir: Optional[str] = None
    error_message: Optional[str] = None
    error_code: Optional[str] = None


class JobEvent(BaseModel):
    """Structured progress event emitted by a job."""

    model_config = ConfigDict(from_attributes=True)

    event_id: int
    job_id: str
    sequence: int
    timestamp: str
    phase: str
    status: str
    message: str
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class JobResultResponse(BaseModel):
    """Detailed domain result of a finished job."""

    job_id: str
    job_type: JobType
    job_status: JobStatus
    domain_status: Optional[str] = None
    output_dir: Optional[str] = None
    completed_at: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
