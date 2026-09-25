"""Base data models and types for ROCmHub Model Forge."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

CURRENT_FORGE_SCHEMA_VERSION = "1.0.0"


def _utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


class BuildStatus(str, Enum):
    """Lifecycle status of a forge build."""

    CONFIG_ONLY = "CONFIG_ONLY"
    PREPARED = "PREPARED"
    EXECUTED = "EXECUTED"
    FAILED = "FAILED"


class MaterializationMode(str, Enum):
    """Mode of model file acquisition."""

    METADATA_ONLY = "METADATA_ONLY"
    FULL_WEIGHTS = "FULL_WEIGHTS"


class StepStatus(str, Enum):
    """Execution status of an individual build step."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class BuildStepSpec(BaseModel):
    """Specification of an ordered build step in a forge recipe."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(..., description="Unique name of the build step.")
    description: str = Field(..., description="Human-readable description of what the step does.")
    required: bool = Field(default=True, description="Whether this step is strictly required for success.")


class BuildStepRecord(BaseModel):
    """Recorded result of an executed build step."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(..., description="Unique name of the step.")
    status: StepStatus = Field(..., description="Execution status.")
    duration_seconds: float = Field(default=0.0, ge=0.0, description="Duration in seconds.")
    message: Optional[str] = Field(default=None, description="Informative status or error message.")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Structured step execution details.")


class MaterializedModel(BaseModel):
    """Metadata describing a materialized (downloaded/cached) model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    local_path: str = Field(..., description="Local directory path where model files reside.")
    mode: MaterializationMode = Field(
        default=MaterializationMode.FULL_WEIGHTS, description="Materialization mode."
    )
    files: List[str] = Field(default_factory=list, description="List of materialized file names.")
    has_weights: bool = Field(
        default=False, description="Whether complete weights are verified present."
    )
    license_name: Optional[str] = Field(default=None, description="Detected license name/identifier.")
    weights_size_bytes: int = Field(default=0, ge=0, description="Total size of weights files in bytes.")
    cached: bool = Field(default=False, description="Whether weights were reused from local HF cache.")
    cache_status: Optional[str] = None
    integrity: Optional[str] = None
    materialized_bytes: int = Field(default=0, ge=0)


class ForgePlan(BaseModel):
    """Deterministic build plan for preparing an AI model on AMD GPU."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    plan_id: str = Field(..., description="Deterministic plan identifier.")
    model_id: str = Field(..., description="Source model ID (e.g. Qwen/Qwen2.5-0.5B-Instruct).")
    revision: str = Field(..., min_length=40, max_length=40, description="Immutable 40-character commit SHA.")
    target_gpu: Optional[str] = Field(default=None, description="Target AMD GPU architecture (e.g. gfx90a).")
    precision: str = Field(..., description="Target floating-point precision (fp16, bf16, fp32).")
    recipe_id: str = Field(..., description="Identifier of the matching build recipe.")
    recipe_version: str = Field(..., description="Version of the matching build recipe.")
    output_dir: str = Field(..., description="Target build output directory path.")
    estimated_disk_space_bytes: int = Field(default=0, ge=0, description="Estimated disk space requirement in bytes.")
    steps: List[BuildStepSpec] = Field(default_factory=list, description="Ordered sequence of build steps.")
    compatibility_confirmed: bool = Field(default=False, description="Whether target environment compatibility was confirmed.")
    created_at: str = Field(default_factory=_utc_now_iso, description="UTC timestamp of plan creation.")
    schema_version: str = Field(default=CURRENT_FORGE_SCHEMA_VERSION, description="Forge schema version.")
