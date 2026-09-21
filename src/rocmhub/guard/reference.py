"""Hardware health checking and reference comparison for Benchmark Guard."""

from __future__ import annotations

from typing import List, Optional, Tuple

from rocmhub.core.types import (
    GuardPolicy,
    HardwareHealthSnapshot,
    ReferenceComparison,
    ReferenceMeasurement,
)
from rocmhub.guard.base import (
    HARDWARE_ECC_ERRORS,
    HARDWARE_THROTTLING_DETECTED,
    REFERENCE_DRIFT,
    TELEMETRY_UNAVAILABLE,
)


def evaluate_hardware_health(
    snapshot: Optional[HardwareHealthSnapshot],
    policy: GuardPolicy,
) -> Tuple[bool, List[str], List[str]]:
    """Evaluate observed hardware health against policy rules.

    Returns:
        Tuple of (healthy: bool, reasons: list[str], warnings: list[str])
    """
    reasons: List[str] = []
    warnings: List[str] = []

    if snapshot is None:
        if policy.strict_telemetry:
            reasons.append(TELEMETRY_UNAVAILABLE)
            return False, reasons, warnings
        warnings.append("Hardware telemetry snapshot is unavailable for this run.")
        return True, reasons, warnings

    healthy = True

    # 1. Throttling check
    if snapshot.throttling_detected is True:
        reasons.append(HARDWARE_THROTTLING_DETECTED)
        healthy = False

    # 2. ECC error check
    if snapshot.ecc_errors is not None and snapshot.ecc_errors > policy.max_tolerated_ecc_errors:
        reasons.append(HARDWARE_ECC_ERRORS)
        healthy = False

    return healthy, reasons, warnings


def compare_reference_measurements(
    before: Optional[ReferenceMeasurement],
    after: Optional[ReferenceMeasurement],
    max_relative_drift: float = 0.05,
) -> Tuple[Optional[bool], Optional[ReferenceComparison], List[str]]:
    """Compare before and after reference calibration measurements.

    If either reference measurement is missing, reference_stable is None (never True).
    If both are present, tests whether drift exceeds max_relative_drift (default 5%).

    Returns:
        Tuple of (reference_stable, comparison_model_or_none, reasons)
    """
    if before is None or after is None:
        return None, None, []

    denom = max(abs(before.value), 1e-9)
    rel_change = abs(after.value - before.value) / denom
    is_stable = rel_change <= max_relative_drift

    comparison = ReferenceComparison(
        reference_id=before.reference_id,
        metric_name=before.metric_name,
        before_value=before.value,
        after_value=after.value,
        relative_change=rel_change,
        stable=is_stable,
    )

    reasons: List[str] = []
    if not is_stable:
        reasons.append(REFERENCE_DRIFT)

    return is_stable, comparison, reasons
