"""ValidationEvaluator orchestrating suite execution, correctness gates, and report generation."""

from __future__ import annotations

import time
from typing import Dict, List, Optional

from rocmhub.core.types import (
    ExecutionStatus,
    ModelSpec,
    ValidationCase,
    ValidationMode,
    ValidationReport,
    ValidationRunResult,
    ValidationVerdict,
)
from rocmhub.runners.base import BaseRunner
from rocmhub.validation.base import ValidationConfig
from rocmhub.validation.correctness import CorrectnessEvaluator
from rocmhub.validation.quality import QualityEvaluator


class ValidationEvaluator:
    """Orchestrates test case execution on runners and compiles the final ValidationReport."""

    def __init__(
        self,
        config: Optional[ValidationConfig] = None,
        correctness_evaluator: Optional[CorrectnessEvaluator] = None,
        quality_evaluator: Optional[QualityEvaluator] = None,
    ) -> None:
        self.config = config or ValidationConfig()
        self.correctness_evaluator = correctness_evaluator or CorrectnessEvaluator()
        self.quality_evaluator = quality_evaluator or QualityEvaluator()

    @staticmethod
    def _execute_case(
        runner: BaseRunner,
        case: ValidationCase,
        max_new_tokens: int,
    ) -> ValidationRunResult:
        """Execute a single validation test case on a runner."""
        t0 = time.perf_counter_ns()
        try:
            run_result = runner.generate(
                prompt=case.prompt,
                max_new_tokens=case.max_new_tokens or max_new_tokens,
            )
            elapsed_ms = (time.perf_counter_ns() - t0) / 1_000_000.0

            return ValidationRunResult(
                case_id=case.case_id,
                status=run_result.status,
                input_tokens=run_result.input_tokens,
                generated_tokens=run_result.generated_tokens,
                generated_text=run_result.generated_text,
                error=run_result.error,
                run_duration_ms=elapsed_ms,
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
            return ValidationRunResult(
                case_id=case.case_id,
                status=ExecutionStatus.FAILED,
                input_tokens=None,
                generated_tokens=None,
                generated_text=None,
                error=str(exc),
                run_duration_ms=elapsed_ms,
            )

    def run_self_validation(
        self,
        runner: BaseRunner,
        model: ModelSpec,
    ) -> ValidationReport:
        """Execute self-validation suite on a baseline model runner.

        In self-validation mode:
        - Evaluates technical correctness invariants;
        - Checks critical cases (failure blocks PASS);
        - Quality/QRR is explicitly NOT measured (remains None).

        Args:
            runner: Loaded baseline runner.
            model: Model specification.

        Returns:
            ValidationReport with PASS, FAIL, or INCONCLUSIVE verdict.
        """
        primary_runs: List[ValidationRunResult] = []
        repeated_runs: Dict[str, List[ValidationRunResult]] = {}
        cases = self.config.cases
        critical_case_ids = {c.case_id for c in cases if c.critical}

        # 1. Execute primary run for each test case
        for case in cases:
            run_res = self._execute_case(runner, case, self.config.max_new_tokens)
            primary_runs.append(run_res)

        # 2. If deterministic mode is requested, execute a second run for determinism check
        if self.config.deterministic:
            for case in cases:
                # Run second iteration
                repeat_res = self._execute_case(runner, case, self.config.max_new_tokens)
                first_run = next(r for r in primary_runs if r.case_id == case.case_id)
                repeated_runs[case.case_id] = [first_run, repeat_res]

        # 3. Evaluate correctness gates
        correctness_res = self.correctness_evaluator.evaluate(
            cases=cases,
            runs=primary_runs,
            repeated_runs=repeated_runs,
        )

        # 4. Evaluate quality (in self-validation mode, quality_measured is False, QRR is None)
        quality_res = self.quality_evaluator.evaluate(
            baseline_runs=primary_runs,
            candidate_runs=None,
        )

        # 5. Case statistics
        total_cases = len(cases)
        completed_cases = sum(1 for r in primary_runs if r.status == ExecutionStatus.SUCCESS)
        failed_cases = sum(1 for r in primary_runs if r.status == ExecutionStatus.FAILED)

        # Check critical cases failures (either execution failed or correctness check failed)
        case_failures_map: Dict[str, List[str]] = {}
        for fail_msg in correctness_res.failures:
            # Extract case_id prefix, e.g. "[case_id]"
            if fail_msg.startswith("[") and "]" in fail_msg:
                cid = fail_msg[1:fail_msg.index("]")]
                case_failures_map.setdefault(cid, []).append(fail_msg)

        critical_cases_failed = 0
        for cid in critical_case_ids:
            run_entry = next((r for r in primary_runs if r.case_id == cid), None)
            if run_entry is None or run_entry.status == ExecutionStatus.FAILED or cid in case_failures_map:
                critical_cases_failed += 1

        # 6. Assign verdict
        reasons: List[str] = []
        warnings: List[str] = []

        if critical_cases_failed > 0:
            verdict = ValidationVerdict.FAIL
            reasons.append(f"{critical_cases_failed} critical validation test case(s) failed.")
        elif not correctness_res.passed:
            verdict = ValidationVerdict.FAIL
            reasons.append(f"Correctness gate failed ({correctness_res.checks_failed} check failures).")
        else:
            verdict = ValidationVerdict.PASS
            reasons.append("All correctness gates and critical validation cases passed.")

        for f in correctness_res.failures:
            reasons.append(f"Failure: {f}")

        return ValidationReport(
            mode=ValidationMode.SELF_VALIDATION,
            model_id=model.model_id,
            baseline_revision=model.commit_sha,
            candidate_revision=None,
            verdict=verdict,
            correctness_passed=correctness_res.passed,
            quality_measured=quality_res.quality_measured,
            qrr_percent=quality_res.qrr_percent,
            cases_total=total_cases,
            cases_completed=completed_cases,
            cases_failed=failed_cases,
            critical_cases_failed=critical_cases_failed,
            case_results=primary_runs,
            reasons=reasons,
            warnings=warnings,
        )

    def run_comparison(
        self,
        baseline_runner: BaseRunner,
        candidate_runner: BaseRunner,
        baseline_model: ModelSpec,
        candidate_model: ModelSpec,
    ) -> ValidationReport:
        """Execute comparative validation between baseline and candidate runners.

        Args:
            baseline_runner: Loaded baseline runner.
            candidate_runner: Loaded candidate runner.
            baseline_model: Baseline ModelSpec.
            candidate_model: Candidate ModelSpec.

        Returns:
            ValidationReport with QRR and regression verdict.
        """
        cases = self.config.cases
        critical_case_ids = {c.case_id for c in cases if c.critical}

        base_runs: List[ValidationRunResult] = []
        cand_runs: List[ValidationRunResult] = []

        for case in cases:
            base_runs.append(self._execute_case(baseline_runner, case, self.config.max_new_tokens))
            cand_runs.append(self._execute_case(candidate_runner, case, self.config.max_new_tokens))

        # Evaluate correctness on candidate
        cand_correctness = self.correctness_evaluator.evaluate(cases=cases, runs=cand_runs)

        # Evaluate quality agreement
        quality_res = self.quality_evaluator.evaluate(baseline_runs=base_runs, candidate_runs=cand_runs)

        total_cases = len(cases)
        completed_cases = sum(1 for r in cand_runs if r.status == ExecutionStatus.SUCCESS)
        failed_cases = sum(1 for r in cand_runs if r.status == ExecutionStatus.FAILED)

        # Count critical failures
        critical_cases_failed = sum(
            1 for r in cand_runs if r.case_id in critical_case_ids and r.status == ExecutionStatus.FAILED
        )

        reasons: List[str] = []
        warnings: List[str] = []

        # Verdict logic
        if critical_cases_failed > 0:
            verdict = ValidationVerdict.FAIL
            reasons.append(f"{critical_cases_failed} critical validation test case(s) failed on candidate.")
        elif not cand_correctness.passed:
            verdict = ValidationVerdict.FAIL
            reasons.append("Correctness gate failed on candidate.")
        elif quality_res.qrr_percent is not None and quality_res.qrr_percent < (self.config.quality_threshold * 100.0):
            verdict = ValidationVerdict.FAIL
            reasons.append(
                f"QRR {quality_res.qrr_percent}% is below required quality threshold of {self.config.quality_threshold * 100.0}%."
            )
        elif not quality_res.quality_measured:
            verdict = ValidationVerdict.INCONCLUSIVE
            reasons.append("Candidate executed, but quality metrics were insufficient to draw conclusions.")
        else:
            verdict = ValidationVerdict.PASS
            reasons.append("Candidate passed correctness and met quality retention thresholds.")

        return ValidationReport(
            mode=ValidationMode.COMPARISON,
            model_id=baseline_model.model_id,
            baseline_revision=baseline_model.commit_sha,
            candidate_revision=candidate_model.commit_sha,
            verdict=verdict,
            correctness_passed=cand_correctness.passed,
            quality_measured=quality_res.quality_measured,
            qrr_percent=quality_res.qrr_percent,
            cases_total=total_cases,
            cases_completed=completed_cases,
            cases_failed=failed_cases,
            critical_cases_failed=critical_cases_failed,
            case_results=cand_runs,
            reasons=reasons,
            warnings=warnings,
        )
