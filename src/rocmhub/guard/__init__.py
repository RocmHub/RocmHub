"""Benchmark Guard & Reproducibility Gate subsystem."""

from __future__ import annotations

from rocmhub.guard.base import (
    ARTIFACT_INTEGRITY_FAILED,
    DEFAULT_GUARD_POLICY,
    ENVIRONMENT_DRIFT,
    EVIDENCE_INCONSISTENT,
    HARDWARE_ECC_ERRORS,
    HARDWARE_THROTTLING_DETECTED,
    HIGH_VARIABILITY,
    INSUFFICIENT_MEASUREMENT_RUNS,
    NO_BENCHMARK_EXECUTION,
    REFERENCE_DRIFT,
    STABLE_MEASUREMENT,
    SUMMARY_MISMATCH,
    TELEMETRY_UNAVAILABLE,
    ABBASequence,
)
from rocmhub.guard.environment import (
    compare_fingerprints,
    extract_environment_fingerprint,
)
from rocmhub.guard.evaluator import BenchmarkGuard
from rocmhub.guard.reference import (
    compare_reference_measurements,
    evaluate_hardware_health,
)
from rocmhub.guard.statistics import (
    compute_mad,
    compute_median,
    compute_relative_mad,
    recompute_benchmark_summary,
    verify_summary_against_raw,
)

__all__ = [
    "BenchmarkGuard",
    "ABBASequence",
    "DEFAULT_GUARD_POLICY",
    "NO_BENCHMARK_EXECUTION",
    "ARTIFACT_INTEGRITY_FAILED",
    "INSUFFICIENT_MEASUREMENT_RUNS",
    "EVIDENCE_INCONSISTENT",
    "SUMMARY_MISMATCH",
    "ENVIRONMENT_DRIFT",
    "HARDWARE_THROTTLING_DETECTED",
    "HARDWARE_ECC_ERRORS",
    "HIGH_VARIABILITY",
    "STABLE_MEASUREMENT",
    "REFERENCE_DRIFT",
    "TELEMETRY_UNAVAILABLE",
    "extract_environment_fingerprint",
    "compare_fingerprints",
    "evaluate_hardware_health",
    "compare_reference_measurements",
    "compute_median",
    "compute_mad",
    "compute_relative_mad",
    "recompute_benchmark_summary",
    "verify_summary_against_raw",
]
