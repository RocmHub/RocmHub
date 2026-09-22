"""Base data models and contracts for ROCmHub Optimization Engine (Phase 12)."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from rocmhub.benchmarks.base import BenchmarkConfig
from rocmhub.core.types import BenchmarkResult, RunResult, ValidationReport
from rocmhub.forge.manifest import BuildManifest
from rocmhub.validation.base import ValidationConfig

CURRENT_OPTIMIZATION_SCHEMA_VERSION = "1.0.0"


def _utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


class OptimizationStrategy(str, Enum):
    """Supported optimization strategies."""

    BF16 = "bf16"
    FP16 = "fp16"
    FP32 = "fp32"
    TORCH_COMPILE = "torch_compile"
    INT8 = "int8"
    FP8 = "fp8"


class CandidateStatus(str, Enum):
    """Lifecycle status of an optimization candidate."""

    PLANNED = "PLANNED"
    CONFIG_ONLY = "CONFIG_ONLY"
    PREPARED = "PREPARED"
    EXECUTED = "EXECUTED"
    FAILED = "FAILED"
    UNSUPPORTED = "UNSUPPORTED"


class ComparisonVerdict(str, Enum):
    """Outcome verdict of comparing candidate against baseline."""

    IMPROVED = "IMPROVED"
    REGRESSED = "REGRESSED"
    NO_CHANGE = "NO_CHANGE"
    INCONCLUSIVE = "INCONCLUSIVE"
    NOT_MEASURED = "NOT_MEASURED"


class OptimizationCandidate(BaseModel):
    """Specification and recorded results for an optimization candidate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str = Field(..., description="Deterministic candidate identifier (cand-<sha256[:16]>).")
    model_id: str = Field(..., description="Target model repository ID.")
    revision: str = Field(..., description="Immutable 40-character commit SHA.")
    strategy: OptimizationStrategy = Field(..., description="Applied optimization strategy.")
    precision: str = Field(default="fp16", description="Target model precision (e.g. bf16, fp16, fp32).")
    runtime_flags: Dict[str, Any] = Field(default_factory=dict, description="Custom runtime engine flags.")
    target_gpu: Optional[str] = Field(default=None, description="Target AMD GPU architecture string.")
    build_dir: Optional[str] = Field(default=None, description="Path to candidate build directory.")
    build_manifest: Optional[BuildManifest] = Field(default=None, description="Candidate build manifest.")
    status: CandidateStatus = Field(default=CandidateStatus.PLANNED, description="Candidate lifecycle status.")
    run_result: Optional[RunResult] = Field(default=None, description="Single execution inference result.")
    benchmark_result: Optional[BenchmarkResult] = Field(default=None, description="Benchmark measurement result.")
    validation_report: Optional[ValidationReport] = Field(default=None, description="Correctness & quality report.")
    errors: List[str] = Field(default_factory=list, description="Errors encountered during build or execution.")


class OptimizationBaseline(BaseModel):
    """Fixed, immutable reference baseline against which all candidates are compared."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str = Field(..., description="Base model ID.")
    revision: str = Field(..., description="Immutable 40-character commit SHA.")
    target_gpu: Optional[str] = Field(default=None, description="Target AMD GPU architecture.")
    precision: str = Field(default="fp16", description="Baseline precision (standard fp16).")
    runtime: str = Field(default="pytorch_transformers_hip", description="Runtime adapter.")
    build_dir: Optional[str] = Field(default=None, description="Baseline build directory.")
    build_manifest: Optional[BuildManifest] = Field(default=None, description="Baseline build manifest.")
    run_result: Optional[RunResult] = Field(default=None, description="Baseline single run result.")
    benchmark_result: Optional[BenchmarkResult] = Field(default=None, description="Baseline benchmark result.")
    validation_report: Optional[ValidationReport] = Field(default=None, description="Baseline validation report.")
    status: CandidateStatus = Field(default=CandidateStatus.PLANNED, description="Baseline lifecycle status.")
    measured: bool = Field(default=False, description="Whether real GPU metrics were measured.")
    created_at: str = Field(default_factory=_utc_now_iso, description="Timestamp of baseline recording.")


class ComparisonResult(BaseModel):
    """Objective, reproducible comparative evaluation between candidate and baseline."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str = Field(..., description="Evaluated candidate identifier.")
    strategy: OptimizationStrategy = Field(..., description="Strategy evaluated.")
    verdict: ComparisonVerdict = Field(..., description="Comparative outcome verdict.")
    ttft_speedup: Optional[float] = Field(
        default=None, description="TTFT speedup ratio (baseline_ttft / cand_ttft). >1.0 is faster."
    )
    throughput_speedup: Optional[float] = Field(
        default=None, description="Throughput speedup ratio (cand_thr / baseline_thr). >1.0 is faster."
    )
    vram_reduction_mb: Optional[float] = Field(
        default=None, description="VRAM reduction in MB (baseline_vram - cand_vram). >0 is better."
    )
    qrr_percent: Optional[float] = Field(
        default=None, description="Quality Retention Rate percentage (0-100%)."
    )
    reasons: List[str] = Field(default_factory=list, description="Detailed explanatory reasons for verdict.")


class OptimizationRequest(BaseModel):
    """Input specification requesting multi-variant optimization evaluation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str = Field(..., description="Hugging Face model ID.")
    revision: Optional[str] = Field(default=None, description="Branch, tag, or commit SHA.")
    target_gpu: Optional[str] = Field(default=None, description="Target AMD GPU (e.g. gfx90a, gfx1100).")
    objective: str = Field(default="MAX_THROUGHPUT", description="Target objective (MAX_THROUGHPUT, MIN_LATENCY).")
    strategies: List[OptimizationStrategy] = Field(
        default_factory=lambda: [
            OptimizationStrategy.BF16,
            OptimizationStrategy.FP16,
            OptimizationStrategy.FP32,
        ],
        description="Candidate optimization strategies to evaluate.",
    )
    max_candidates: int = Field(default=3, ge=1, le=10, description="Max candidates to prepare and evaluate.")
    max_execution_time_seconds: float = Field(
        default=600.0, ge=1.0, description="Max allowed total runtime in seconds."
    )
    allow_full_weights: bool = Field(
        default=False, description="Whether downloading full model weight tensors is permitted."
    )
    output_dir: Optional[str] = Field(default=None, description="Custom target directory for builds.")
    benchmark_config: Optional[BenchmarkConfig] = Field(
        default=None, description="Custom benchmark workload configuration."
    )
    validation_config: Optional[ValidationConfig] = Field(
        default=None, description="Custom validation suite configuration."
    )


class OptimizationPlan(BaseModel):
    """Deterministic plan of candidate configurations to prepare and benchmark."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(default=CURRENT_OPTIMIZATION_SCHEMA_VERSION)
    plan_id: str = Field(..., description="Deterministic plan identifier.")
    model_id: str = Field(..., description="Model ID.")
    revision: str = Field(..., description="Resolved immutable 40-char commit SHA.")
    target_gpu: Optional[str] = Field(default=None, description="Target GPU architecture.")
    baseline_spec: Dict[str, Any] = Field(..., description="Baseline configuration parameters.")
    candidate_specs: List[Dict[str, Any]] = Field(..., description="Planned candidate configurations.")
    created_at: str = Field(default_factory=_utc_now_iso)


class OptimizationReport(BaseModel):
    """Final, verified report comparing baseline and all candidate optimization variants."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(default=CURRENT_OPTIMIZATION_SCHEMA_VERSION)
    session_id: str = Field(..., description="Unique optimization session ID.")
    model_id: str = Field(..., description="Model ID.")
    revision: str = Field(..., description="Resolved immutable 40-char commit SHA.")
    target_gpu: Optional[str] = Field(default=None, description="Target GPU architecture.")
    objective: str = Field(..., description="Optimization objective.")
    status: str = Field(..., description="Overall terminal status (SUCCESS, CONFIG_ONLY, STOPPED_ENVIRONMENT, etc.).")
    baseline: OptimizationBaseline = Field(..., description="Fixed reference baseline.")
    candidates: List[OptimizationCandidate] = Field(default_factory=list, description="All evaluated candidates.")
    comparisons: List[ComparisonResult] = Field(default_factory=list, description="Comparative evaluations.")
    best_candidate_id: Optional[str] = Field(default=None, description="Winner candidate ID if verified improved.")
    recommendations: List[str] = Field(default_factory=list, description="Actionable recommendations.")
    errors: List[str] = Field(default_factory=list, description="Errors encountered during session.")
    total_duration_seconds: float = Field(default=0.0, ge=0.0, description="Total elapsed runtime.")
    created_at: str = Field(default_factory=_utc_now_iso)
