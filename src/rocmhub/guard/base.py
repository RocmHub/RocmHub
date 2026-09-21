"""Base definitions, reason codes, and policy defaults for Benchmark Guard."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from rocmhub.core.types import CURRENT_SCHEMA_VERSION, ExperimentRole, GuardPolicy

# Standard reason codes for ReproducibilityReport
NO_BENCHMARK_EXECUTION = "NO_BENCHMARK_EXECUTION"
ARTIFACT_INTEGRITY_FAILED = "ARTIFACT_INTEGRITY_FAILED"
INSUFFICIENT_MEASUREMENT_RUNS = "INSUFFICIENT_MEASUREMENT_RUNS"
EVIDENCE_INCONSISTENT = "EVIDENCE_INCONSISTENT"
SUMMARY_MISMATCH = "SUMMARY_MISMATCH"
ENVIRONMENT_DRIFT = "ENVIRONMENT_DRIFT"
HARDWARE_THROTTLING_DETECTED = "HARDWARE_THROTTLING_DETECTED"
HARDWARE_ECC_ERRORS = "HARDWARE_ECC_ERRORS"
HIGH_VARIABILITY = "HIGH_VARIABILITY"
STABLE_MEASUREMENT = "STABLE_MEASUREMENT"
REFERENCE_DRIFT = "REFERENCE_DRIFT"
TELEMETRY_UNAVAILABLE = "TELEMETRY_UNAVAILABLE"

# Default MVP Guard Policy
DEFAULT_GUARD_POLICY = GuardPolicy(
    policy_version="1.0.0",
    min_measurement_runs=5,
    max_relative_mad_ttft=0.15,        # 15% relative MAD threshold for TTFT
    max_relative_mad_throughput=0.10,  # 10% relative MAD threshold for throughput
    max_relative_mad_latency=0.15,     # 15% relative MAD threshold for total latency
    require_environment_match=True,
    require_complete_benchmark=True,
    strict_telemetry=False,
    max_tolerated_ecc_errors=0,
    summary_recomputation_tolerance=1e-3,
)


class ABBASequence(BaseModel):
    """Data model representing interleaved A/B/B/A experiment execution sequence.

    Designed for future paired comparative evaluation to isolate thermal and cluster drift.
    Note: Phase 9 establishes the data contract only; optimization comparison logic is deferred.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(default=CURRENT_SCHEMA_VERSION, description="Schema version of ABBASequence")
    sequence_id: str = Field(..., description="Unique identifier of this A/B/B/A comparison cycle")
    sequence_order: List[ExperimentRole] = Field(
        default=[
            ExperimentRole.BASELINE,
            ExperimentRole.CANDIDATE,
            ExperimentRole.CANDIDATE,
            ExperimentRole.BASELINE,
        ],
        description="Execution order of experiment roles",
    )
    baseline_experiment_id: str = Field(..., description="Experiment ID of the baseline")
    candidate_experiment_id: str = Field(..., description="Experiment ID of the candidate")
    notes: Optional[str] = Field(default=None, description="Optional annotations regarding test conditions")
