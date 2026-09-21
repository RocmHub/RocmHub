"""Technical correctness verification gates for baseline model inference."""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from rocmhub.core.types import ExecutionStatus, ValidationCase, ValidationRunResult

# Corrupted numeric and encoding tokens resulting from kernel or precision overflow
NUMERIC_CORRUPTION_PATTERNS = [
    re.compile(r"\bnan\b", re.IGNORECASE),
    re.compile(r"\b-nan\b", re.IGNORECASE),
    re.compile(r"\binf\b", re.IGNORECASE),
    re.compile(r"\b-inf\b", re.IGNORECASE),
    re.compile(r"\ufffd"),  # Unicode replacement character indicating byte decode corruption
]


class CorrectnessResult(BaseModel):
    """Aggregated outcome of correctness gate verification."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    passed: bool = Field(..., description="True if all correctness checks passed")
    checks_run: int = Field(default=0, description="Total count of individual checks evaluated")
    checks_passed: int = Field(default=0, description="Count of checks that succeeded")
    checks_failed: int = Field(default=0, description="Count of checks that failed")
    failures: List[str] = Field(default_factory=list, description="Descriptive list of failure reasons")


class CorrectnessEvaluator:
    """Evaluates technical invariants of model inference outputs.

    Correctness is a hard gate:
    - Generation completed without exception;
    - Non-empty output text generated;
    - Positive generated token count (>= 1);
    - Absence of numeric corruption (NaN / Inf strings, replacement characters);
    - Output conforms to expected regex pattern (if specified);
    - Deterministic repeated runs under fixed seed yield identical output.
    """

    @classmethod
    def evaluate_single_run(
        cls,
        case: ValidationCase,
        run: ValidationRunResult,
    ) -> List[str]:
        """Perform correctness checks on a single test case run.

        Args:
            case: Test case definition.
            run: Recorded execution result.

        Returns:
            List of failure messages (empty if all checks passed).
        """
        failures: List[str] = []

        # 1. Check inference completion
        if run.status != ExecutionStatus.SUCCESS:
            failures.append(f"[{case.case_id}] Inference did not complete successfully (status: {run.status.value}, error: {run.error or 'none'})")
            return failures

        # 2. Check output presence
        if run.generated_text is None or not run.generated_text.strip():
            failures.append(f"[{case.case_id}] Output text is missing or whitespace-only")

        # 3. Check token count validity
        if run.generated_tokens is None or run.generated_tokens < 1:
            failures.append(f"[{case.case_id}] Invalid token count: generated_tokens={run.generated_tokens}")

        # 4. Check for NaN / Inf or decode corruption
        if run.generated_text is not None:
            for pattern in NUMERIC_CORRUPTION_PATTERNS:
                if pattern.search(run.generated_text):
                    failures.append(f"[{case.case_id}] Numeric corruption detected in output text matching pattern '{pattern.pattern}'")
                    break

        # 5. Check pattern match if specified
        if case.expected_pattern and run.generated_text is not None:
            if not re.search(case.expected_pattern, run.generated_text):
                failures.append(f"[{case.case_id}] Output text did not match expected regex pattern '{case.expected_pattern}'")

        return failures

    @classmethod
    def evaluate(
        cls,
        cases: List[ValidationCase],
        runs: List[ValidationRunResult],
        repeated_runs: Optional[Dict[str, List[ValidationRunResult]]] = None,
    ) -> CorrectnessResult:
        """Evaluate complete correctness suite over all executed cases.

        Args:
            cases: List of test cases.
            runs: Primary execution results corresponding to cases.
            repeated_runs: Optional dictionary of case_id -> list of repeated runs for determinism checks.

        Returns:
            CorrectnessResult with pass/fail status and failure details.
        """
        cases_map = {c.case_id: c for c in cases}
        all_failures: List[str] = []
        checks_run = 0

        # Per-case technical checks
        for run in runs:
            case = cases_map.get(run.case_id)
            if not case:
                continue

            checks_run += 4  # status, output, token count, corruption
            if case.expected_pattern:
                checks_run += 1

            case_failures = cls.evaluate_single_run(case, run)
            all_failures.extend(case_failures)

        # Determinism check across repeated executions with fixed seed
        if repeated_runs:
            for case_id, rep_list in repeated_runs.items():
                if len(rep_list) >= 2:
                    checks_run += 1
                    first_text = rep_list[0].generated_text
                    for i, r in enumerate(rep_list[1:], start=2):
                        if r.generated_text != first_text:
                            all_failures.append(
                                f"[{case_id}] Determinism violation: iteration {i} output differs from iteration 1 under identical seed"
                            )
                            break

        checks_failed = len(all_failures)
        checks_passed = max(0, checks_run - checks_failed)
        passed = (checks_failed == 0)

        return CorrectnessResult(
            passed=passed,
            checks_run=checks_run,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            failures=all_failures,
        )
