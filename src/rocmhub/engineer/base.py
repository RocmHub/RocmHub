"""Base data models and contracts for Autonomous AI Engineer (Phase 11)."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from rocmhub.core.types import BenchmarkResult, RunResult
from rocmhub.forge.manifest import BuildManifest

CURRENT_ENGINEER_SCHEMA_VERSION = "1.0.0"


def _utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


class EngineerObjective(str, Enum):
    """High-level engineering optimization/preparation goal."""

    BASE_PREPARATION = "BASE_PREPARATION"
    FULL_PREPARATION = "FULL_PREPARATION"
    AMD_EXECUTION = "AMD_EXECUTION"
    MAX_THROUGHPUT = "MAX_THROUGHPUT"
    MIN_LATENCY = "MIN_LATENCY"


class EngineerStatus(str, Enum):
    """Terminal lifecycle status of an AI Engineer session."""

    SUCCESS = "SUCCESS"
    STOPPED_ENVIRONMENT = "STOPPED_ENVIRONMENT"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    FAILED = "FAILED"


class TrajectoryStep(BaseModel):
    """Record of an individual step in the agent's observation-plan-act loop."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    step_index: int = Field(..., ge=0, description="0-indexed sequence position.")
    phase: str = Field(..., description="Execution loop phase (OBSERVE, PLAN, ACT, EVALUATE, REVISE).")
    action: str = Field(..., description="Action name or tool invoked.")
    tool_args: Dict[str, Any] = Field(default_factory=dict, description="Sanitized arguments passed to tool.")
    tool_result: Dict[str, Any] = Field(default_factory=dict, description="Sanitized result returned by tool.")
    observation: Optional[str] = Field(default=None, description="Summary observation after execution.")
    rationale: Optional[str] = Field(default=None, description="Agent's reasoning leading to this action.")
    duration_seconds: float = Field(default=0.0, ge=0.0, description="Duration of step execution in seconds.")
    timestamp: str = Field(default_factory=_utc_now_iso, description="UTC timestamp of step completion.")


class EngineerBudget(BaseModel):
    """Budget and safety boundary constraints for an AI Engineer session."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_attempts: int = Field(default=5, ge=1, description="Maximum number of act/revise attempts.")
    max_execution_time_seconds: float = Field(
        default=600.0, ge=0.001, description="Maximum total runtime in seconds (default 10 mins)."
    )
    max_disk_usage_bytes: int = Field(
        default=20 * 1024**3, ge=1, description="Maximum disk usage allowed in bytes (default 20 GB)."
    )
    allow_full_weights: bool = Field(
        default=False, description="Whether downloading full weight tensors is permitted."
    )


class EngineerRequest(BaseModel):
    """Input specification requesting autonomous preparation or execution of a model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str = Field(..., description="Hugging Face model ID (e.g. Qwen/Qwen2.5-0.5B-Instruct).")
    revision: Optional[str] = Field(default=None, description="Optional revision tag, branch, or commit SHA.")
    target_gpu: Optional[str] = Field(default=None, description="Target AMD GPU gfx architecture (e.g. gfx90a).")
    objective: EngineerObjective = Field(
        default=EngineerObjective.BASE_PREPARATION, description="Engineering goal."
    )
    budget: EngineerBudget = Field(default_factory=EngineerBudget, description="Session constraints.")
    output_dir: Optional[str] = Field(default=None, description="Custom target build directory.")


class EngineerReport(BaseModel):
    """Final structured outcome of an autonomous AI Engineer session."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(default=CURRENT_ENGINEER_SCHEMA_VERSION, description="Schema version.")
    session_id: str = Field(..., description="Deterministic session identifier.")
    model_id: str = Field(..., description="Model ID prepared.")
    revision: str = Field(..., description="Resolved immutable 40-char commit SHA.")
    target_gpu: Optional[str] = Field(default=None, description="Detected or targeted GPU architecture.")
    objective: EngineerObjective = Field(..., description="Requested engineering objective.")
    status: EngineerStatus = Field(..., description="Terminal execution status.")
    started_at: str = Field(..., description="UTC start timestamp.")
    completed_at: str = Field(default_factory=_utc_now_iso, description="UTC completion timestamp.")
    total_duration_seconds: float = Field(default=0.0, ge=0.0, description="Total elapsed runtime.")
    attempts_used: int = Field(default=0, ge=0, description="Number of attempts used.")
    trajectory: List[TrajectoryStep] = Field(default_factory=list, description="Audit trace of all actions.")
    build_manifest: Optional[BuildManifest] = Field(default=None, description="Final Forge build manifest.")
    run_result: Optional[RunResult] = Field(default=None, description="Optional baseline execution result.")
    benchmark_result: Optional[BenchmarkResult] = Field(default=None, description="Optional benchmark result.")
    reasons: List[str] = Field(default_factory=list, description="Explanations for the outcome.")
    errors_encountered: List[str] = Field(default_factory=list, description="Log of recoverable/handled errors.")
    secret_scan_clean: bool = Field(default=True, description="Strict confirmation that no secrets are present.")
