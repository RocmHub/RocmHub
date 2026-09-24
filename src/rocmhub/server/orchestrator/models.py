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
    PREPARE_MODEL_FOR_AMD = "PREPARE_MODEL_FOR_AMD"


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
    target_gfx: Optional[str] = Field(default=None, max_length=64)
    runtime: Optional[str] = Field(default="pytorch_transformers_hip", max_length=128)


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
    agent_id: Optional[str] = None
    claimed_at: Optional[str] = None
    heartbeat_at: Optional[str] = None
    attempt: int = 0


class AgentStatus(str, Enum):
    ONLINE = "ONLINE"
    BUSY = "BUSY"
    OFFLINE = "OFFLINE"
    DEGRADED = "DEGRADED"


class AgentCapabilities(BaseModel):
    os: str
    architecture: str
    python_version: str
    rocm_detected: bool = False
    hip_detected: bool = False
    pytorch_version: Optional[str] = None
    amd_gpu_count: int = 0
    gpu_names: List[str] = Field(default_factory=list)
    gfx_targets: List[str] = Field(default_factory=list)
    available_memory_mb: Optional[int] = None
    available_disk_gb: Optional[float] = None
    capabilities: List[str] = Field(default_factory=lambda: ["PREPARE_MODEL_FOR_AMD"])


class AgentRegisterRequest(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    hostname: str = Field(min_length=1, max_length=255)
    capabilities: AgentCapabilities


class AgentResponse(BaseModel):
    agent_id: str
    name: str
    hostname: str
    status: AgentStatus
    capabilities: AgentCapabilities
    last_seen: str
    created_at: str


class AgentClaimResponse(BaseModel):
    job: Optional[JobResponse] = None
    request_payload: Optional[Dict[str, Any]] = None


class AgentEventRequest(BaseModel):
    phase: str = Field(min_length=1, max_length=64)
    status: str = Field(min_length=1, max_length=32)
    message: str = Field(min_length=1, max_length=2048)
    details: Optional[Dict[str, Any]] = None
    attempt: Optional[int] = Field(default=None, ge=1)


class AgentCompleteRequest(BaseModel):
    domain_status: str = Field(min_length=1, max_length=64)
    result: Dict[str, Any]
    output_dir: Optional[str] = None
    revision: Optional[str] = None
    artifact_files: Dict[str, str] = Field(
        default_factory=dict,
        description="Small UTF-8 preparation artifacts uploaded by an authenticated Agent",
    )
    attempt: Optional[int] = Field(default=None, ge=1)
    completion_id: Optional[str] = Field(default=None, min_length=1, max_length=64)


class AgentFailRequest(BaseModel):
    error_message: str = Field(min_length=1, max_length=2048)
    error_code: str = Field(default="AGENT_EXECUTION_FAILED", max_length=128)
    attempt: Optional[int] = Field(default=None, ge=1)


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


class JobListResponse(BaseModel):
    """Paginated list of jobs."""

    model_config = ConfigDict(from_attributes=True)

    items: List[JobResponse]
    total: int
    limit: int
    offset: int
