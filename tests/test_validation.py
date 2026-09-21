"""Phase 7: Correctness & Quality Validation — unit test suite.

Tests cover:
- ValidationConfig construction and validators
- DEFAULT_VALIDATION_CASES stability
- CorrectnessEvaluator: per-case checks, pattern matching, determinism, corruption
- QualityEvaluator: self-validation (QRR=None), comparison (QRR computed), edge cases
- ValidationEvaluator: run_self_validation and run_comparison with mock runners
- ValidationReport model_validators (PASS integrity, NOT_MEASURED invariants)
- CLI `rocmhub validate MODEL_ID` — offline preflight gate (no network, no AMD GPU)
"""

from __future__ import annotations

from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from rocmhub.core.types import (
    ExecutionStatus,
    ValidationCase,
    ValidationMode,
    ValidationReport,
    ValidationRunResult,
    ValidationVerdict,
)
from rocmhub.validation.base import DEFAULT_VALIDATION_CASES, ValidationConfig
from rocmhub.validation.correctness import CorrectnessEvaluator
from rocmhub.validation.evaluator import ValidationEvaluator
from rocmhub.validation.quality import (
    ExactTokenAgreementMetric,
    NormalizedTextAgreementMetric,
    QualityEvaluator,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_success_run(
    case_id: str,
    generated_text: str = "Paris",
    generated_tokens: int = 3,
    input_tokens: int = 5,
    run_duration_ms: float = 12.5,
) -> ValidationRunResult:
    return ValidationRunResult(
        case_id=case_id,
        status=ExecutionStatus.SUCCESS,
        input_tokens=input_tokens,
        generated_tokens=generated_tokens,
        generated_text=generated_text,
        run_duration_ms=run_duration_ms,
    )


def _make_failed_run(
    case_id: str,
    error: str = "CUDA out of memory",
) -> ValidationRunResult:
    return ValidationRunResult(
        case_id=case_id,
        status=ExecutionStatus.FAILED,
        input_tokens=None,
        generated_tokens=None,
        generated_text=None,
        error=error,
    )


def _make_skipped_run(case_id: str) -> ValidationRunResult:
    return ValidationRunResult(
        case_id=case_id,
        status=ExecutionStatus.SKIPPED,
        input_tokens=None,
        generated_tokens=None,
        generated_text=None,
    )


# ---------------------------------------------------------------------------
# DEFAULT_VALIDATION_CASES
# ---------------------------------------------------------------------------

class TestDefaultValidationCases:
    def test_three_cases_defined(self) -> None:
        assert len(DEFAULT_VALIDATION_CASES) == 3

    def test_stable_case_ids(self) -> None:
        ids = [c.case_id for c in DEFAULT_VALIDATION_CASES]
        assert "basic_completion_001" in ids
        assert "instruction_following_001" in ids
        assert "deterministic_generation_001" in ids

    def test_two_critical_cases(self) -> None:
        critical_ids = [c.case_id for c in DEFAULT_VALIDATION_CASES if c.critical]
        assert "basic_completion_001" in critical_ids
        assert "instruction_following_001" in critical_ids

    def test_non_critical_case(self) -> None:
        non_critical = [c for c in DEFAULT_VALIDATION_CASES if not c.critical]
        assert len(non_critical) == 1
        assert non_critical[0].case_id == "deterministic_generation_001"

    def test_all_have_prompts(self) -> None:
        for case in DEFAULT_VALIDATION_CASES:
            assert case.prompt and len(case.prompt) > 0

    def test_all_have_positive_max_new_tokens(self) -> None:
        for case in DEFAULT_VALIDATION_CASES:
            assert case.max_new_tokens > 0


# ---------------------------------------------------------------------------
# ValidationConfig
# ---------------------------------------------------------------------------

class TestValidationConfig:
    def test_default_construction(self) -> None:
        cfg = ValidationConfig()
        assert len(cfg.cases) == 3
        assert cfg.max_new_tokens == 16
        assert cfg.quality_threshold == 0.95
        assert cfg.deterministic is True

    def test_custom_max_new_tokens(self) -> None:
        cfg = ValidationConfig(max_new_tokens=32)
        assert cfg.max_new_tokens == 32

    def test_zero_max_new_tokens_raises(self) -> None:
        with pytest.raises(Exception):
            ValidationConfig(max_new_tokens=0)

    def test_negative_max_new_tokens_raises(self) -> None:
        with pytest.raises(Exception):
            ValidationConfig(max_new_tokens=-1)

    def test_empty_cases_raises(self) -> None:
        with pytest.raises(Exception):
            ValidationConfig(cases=[])

    def test_quality_threshold_out_of_range_raises(self) -> None:
        with pytest.raises(Exception):
            ValidationConfig(quality_threshold=1.5)
        with pytest.raises(Exception):
            ValidationConfig(quality_threshold=-0.1)

    def test_boundary_quality_threshold(self) -> None:
        cfg0 = ValidationConfig(quality_threshold=0.0)
        assert cfg0.quality_threshold == 0.0
        cfg1 = ValidationConfig(quality_threshold=1.0)
        assert cfg1.quality_threshold == 1.0

    def test_invalid_precision_raises(self) -> None:
        with pytest.raises(Exception):
            ValidationConfig(precision="int8")

    def test_valid_precisions(self) -> None:
        for prec in ("fp16", "bf16", "fp32"):
            cfg = ValidationConfig(precision=prec)
            assert cfg.precision == prec

    def test_negative_device_id_raises(self) -> None:
        with pytest.raises(Exception):
            ValidationConfig(device_id=-1)

    def test_frozen(self) -> None:
        cfg = ValidationConfig()
        with pytest.raises(Exception):
            cfg.max_new_tokens = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ValidationCase
# ---------------------------------------------------------------------------

class TestValidationCase:
    def test_frozen_immutable(self) -> None:
        case = ValidationCase(
            case_id="test_001", prompt="Hello", max_new_tokens=8, critical=True
        )
        with pytest.raises(Exception):
            case.case_id = "other"  # type: ignore[misc]

    def test_optional_pattern(self) -> None:
        case = ValidationCase(
            case_id="x", prompt="p", expected_pattern=r"\d+"
        )
        assert case.expected_pattern == r"\d+"

    def test_no_extra_fields(self) -> None:
        with pytest.raises(Exception):
            ValidationCase(case_id="x", prompt="p", unknown_field=True)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# ValidationRunResult
# ---------------------------------------------------------------------------

class TestValidationRunResult:
    def test_success_requires_generated_text(self) -> None:
        with pytest.raises(Exception):
            ValidationRunResult(
                case_id="x",
                status=ExecutionStatus.SUCCESS,
                input_tokens=4,
                generated_tokens=2,
                generated_text=None,  # invalid
            )

    def test_success_requires_non_negative_tokens(self) -> None:
        with pytest.raises(Exception):
            ValidationRunResult(
                case_id="x",
                status=ExecutionStatus.SUCCESS,
                input_tokens=4,
                generated_tokens=-1,
                generated_text="hello",
            )

    def test_skipped_must_not_have_text(self) -> None:
        with pytest.raises(Exception):
            ValidationRunResult(
                case_id="x",
                status=ExecutionStatus.SKIPPED,
                input_tokens=None,
                generated_tokens=None,
                generated_text="not allowed",
            )

    def test_not_measured_must_not_have_tokens(self) -> None:
        with pytest.raises(Exception):
            ValidationRunResult(
                case_id="x",
                status=ExecutionStatus.NOT_MEASURED,
                input_tokens=3,  # invalid
                generated_tokens=None,
                generated_text=None,
            )

    def test_valid_success_run(self) -> None:
        run = _make_success_run("x")
        assert run.status == ExecutionStatus.SUCCESS
        assert run.generated_text == "Paris"

    def test_valid_failed_run(self) -> None:
        run = _make_failed_run("y")
        assert run.status == ExecutionStatus.FAILED
        assert run.generated_text is None

    def test_valid_skipped_run(self) -> None:
        run = _make_skipped_run("z")
        assert run.status == ExecutionStatus.SKIPPED


# ---------------------------------------------------------------------------
# ValidationReport model_validator
# ---------------------------------------------------------------------------

class TestValidationReport:
    def _baseline_rev(self) -> str:
        return "abc1234" * 5  # 35 chars

    def _not_measured_report(self) -> ValidationReport:
        return ValidationReport(
            mode=ValidationMode.SELF_VALIDATION,
            model_id="test/model",
            baseline_revision=self._baseline_rev(),
            verdict=ValidationVerdict.NOT_MEASURED,
            correctness_passed=None,
            quality_measured=False,
            qrr_percent=None,
            cases_total=3,
            cases_completed=0,
            cases_failed=0,
            critical_cases_failed=0,
            reasons=["No accelerator"],
        )

    def test_not_measured_is_valid(self) -> None:
        report = self._not_measured_report()
        assert report.verdict == ValidationVerdict.NOT_MEASURED
        assert report.correctness_passed is None
        assert report.qrr_percent is None
        assert report.quality_measured is False

    def test_not_measured_with_correctness_raises(self) -> None:
        with pytest.raises(Exception):
            ValidationReport(
                mode=ValidationMode.SELF_VALIDATION,
                model_id="test/model",
                baseline_revision=self._baseline_rev(),
                verdict=ValidationVerdict.NOT_MEASURED,
                correctness_passed=True,  # invalid
                quality_measured=False,
                qrr_percent=None,
                cases_total=0,
                cases_completed=0,
                cases_failed=0,
                critical_cases_failed=0,
            )

    def test_not_measured_with_qrr_raises(self) -> None:
        with pytest.raises(Exception):
            ValidationReport(
                mode=ValidationMode.SELF_VALIDATION,
                model_id="test/model",
                baseline_revision=self._baseline_rev(),
                verdict=ValidationVerdict.NOT_MEASURED,
                correctness_passed=None,
                quality_measured=False,
                qrr_percent=99.0,  # invalid
                cases_total=0,
                cases_completed=0,
                cases_failed=0,
                critical_cases_failed=0,
            )

    def test_pass_requires_correctness_passed(self) -> None:
        with pytest.raises(Exception):
            ValidationReport(
                mode=ValidationMode.SELF_VALIDATION,
                model_id="test/model",
                baseline_revision=self._baseline_rev(),
                verdict=ValidationVerdict.PASS,
                correctness_passed=False,  # invalid for PASS
                quality_measured=False,
                qrr_percent=None,
                cases_total=3,
                cases_completed=3,
                cases_failed=0,
                critical_cases_failed=0,
            )

    def test_pass_with_critical_failures_raises(self) -> None:
        with pytest.raises(Exception):
            ValidationReport(
                mode=ValidationMode.SELF_VALIDATION,
                model_id="test/model",
                baseline_revision=self._baseline_rev(),
                verdict=ValidationVerdict.PASS,
                correctness_passed=True,
                quality_measured=False,
                qrr_percent=None,
                cases_total=3,
                cases_completed=2,
                cases_failed=0,
                critical_cases_failed=1,  # invalid for PASS
            )

    def test_valid_pass_report(self) -> None:
        report = ValidationReport(
            mode=ValidationMode.SELF_VALIDATION,
            model_id="test/model",
            baseline_revision=self._baseline_rev(),
            verdict=ValidationVerdict.PASS,
            correctness_passed=True,
            quality_measured=False,
            qrr_percent=None,
            cases_total=3,
            cases_completed=3,
            cases_failed=0,
            critical_cases_failed=0,
            reasons=["All correctness gates passed."],
        )
        assert report.verdict == ValidationVerdict.PASS

    def test_fail_report_is_valid(self) -> None:
        report = ValidationReport(
            mode=ValidationMode.SELF_VALIDATION,
            model_id="test/model",
            baseline_revision=self._baseline_rev(),
            verdict=ValidationVerdict.FAIL,
            correctness_passed=False,
            quality_measured=False,
            qrr_percent=None,
            cases_total=3,
            cases_completed=1,
            cases_failed=2,
            critical_cases_failed=1,
            reasons=["Critical case failed."],
        )
        assert report.verdict == ValidationVerdict.FAIL


# ---------------------------------------------------------------------------
# CorrectnessEvaluator
# ---------------------------------------------------------------------------

class TestCorrectnessEvaluator:
    def _case(
        self,
        case_id: str = "c001",
        critical: bool = False,
        expected_pattern: Optional[str] = None,
    ) -> ValidationCase:
        return ValidationCase(
            case_id=case_id,
            prompt="Test prompt",
            max_new_tokens=8,
            critical=critical,
            expected_pattern=expected_pattern,
        )

    def test_single_success_run_passes(self) -> None:
        case = self._case()
        run = _make_success_run(case.case_id, generated_text="Some output")
        failures = CorrectnessEvaluator.evaluate_single_run(case, run)
        assert failures == []

    def test_failed_run_produces_failure_message(self) -> None:
        case = self._case()
        run = _make_failed_run(case.case_id)
        failures = CorrectnessEvaluator.evaluate_single_run(case, run)
        assert len(failures) == 1
        assert case.case_id in failures[0]
        assert "FAILED" in failures[0].upper() or "failed" in failures[0].lower()

    def test_empty_text_fails(self) -> None:
        case = self._case()
        # Create a run with empty text; need to bypass pydantic validator for this
        # We create a special valid run then test the evaluator independently
        run = ValidationRunResult(
            case_id=case.case_id,
            status=ExecutionStatus.SUCCESS,
            input_tokens=4,
            generated_tokens=1,
            generated_text="   ",  # whitespace-only
        )
        failures = CorrectnessEvaluator.evaluate_single_run(case, run)
        assert any("whitespace" in f.lower() or "missing" in f.lower() for f in failures)

    def test_nan_in_output_fails(self) -> None:
        case = self._case()
        run = ValidationRunResult(
            case_id=case.case_id,
            status=ExecutionStatus.SUCCESS,
            input_tokens=3,
            generated_tokens=2,
            generated_text="The value is nan and that's bad",
        )
        failures = CorrectnessEvaluator.evaluate_single_run(case, run)
        assert any("corruption" in f.lower() or "nan" in f.lower() for f in failures)

    def test_inf_in_output_fails(self) -> None:
        case = self._case()
        run = ValidationRunResult(
            case_id=case.case_id,
            status=ExecutionStatus.SUCCESS,
            input_tokens=3,
            generated_tokens=2,
            generated_text="Result: inf overflow",
        )
        failures = CorrectnessEvaluator.evaluate_single_run(case, run)
        assert any("corruption" in f.lower() or "inf" in f.lower() for f in failures)

    def test_unicode_replacement_char_fails(self) -> None:
        case = self._case()
        run = ValidationRunResult(
            case_id=case.case_id,
            status=ExecutionStatus.SUCCESS,
            input_tokens=3,
            generated_tokens=2,
            generated_text="Output\ufffd here",
        )
        failures = CorrectnessEvaluator.evaluate_single_run(case, run)
        assert any("corruption" in f.lower() for f in failures)

    def test_expected_pattern_match_passes(self) -> None:
        case = self._case(expected_pattern=r"Paris")
        run = _make_success_run(case.case_id, generated_text="Paris, France")
        failures = CorrectnessEvaluator.evaluate_single_run(case, run)
        assert failures == []

    def test_expected_pattern_mismatch_fails(self) -> None:
        case = self._case(expected_pattern=r"\d{4}")
        run = _make_success_run(case.case_id, generated_text="No digits here")
        failures = CorrectnessEvaluator.evaluate_single_run(case, run)
        assert any("pattern" in f.lower() for f in failures)

    def test_evaluate_all_success(self) -> None:
        cases = [self._case("c001"), self._case("c002"), self._case("c003")]
        runs = [
            _make_success_run("c001", "text one"),
            _make_success_run("c002", "text two"),
            _make_success_run("c003", "text three"),
        ]
        result = CorrectnessEvaluator.evaluate(cases=cases, runs=runs)
        assert result.passed is True
        assert result.checks_failed == 0
        assert result.failures == []

    def test_evaluate_with_one_failure(self) -> None:
        cases = [self._case("c001"), self._case("c002")]
        runs = [
            _make_success_run("c001", "ok"),
            _make_failed_run("c002", "GPU error"),
        ]
        result = CorrectnessEvaluator.evaluate(cases=cases, runs=runs)
        assert result.passed is False
        assert result.checks_failed >= 1

    def test_evaluate_determinism_pass(self) -> None:
        cases = [self._case("c001")]
        runs = [_make_success_run("c001", "Paris")]
        repeated_runs = {
            "c001": [
                _make_success_run("c001", "Paris"),
                _make_success_run("c001", "Paris"),  # identical
            ]
        }
        result = CorrectnessEvaluator.evaluate(
            cases=cases, runs=runs, repeated_runs=repeated_runs
        )
        assert result.passed is True
        assert not any("determinism" in f.lower() for f in result.failures)

    def test_evaluate_determinism_fail(self) -> None:
        cases = [self._case("c001")]
        runs = [_make_success_run("c001", "Paris")]
        repeated_runs = {
            "c001": [
                _make_success_run("c001", "Paris"),
                _make_success_run("c001", "Lyon"),  # different!
            ]
        }
        result = CorrectnessEvaluator.evaluate(
            cases=cases, runs=runs, repeated_runs=repeated_runs
        )
        assert result.passed is False
        assert any("determinism" in f.lower() for f in result.failures)


# ---------------------------------------------------------------------------
# ExactTokenAgreementMetric
# ---------------------------------------------------------------------------

class TestExactTokenAgreementMetric:
    def setup_method(self) -> None:
        self.metric = ExactTokenAgreementMetric()

    def test_exact_match_returns_one(self) -> None:
        base = _make_success_run("x", "Paris")
        cand = _make_success_run("x", "Paris")
        assert self.metric.evaluate(base, cand) == 1.0

    def test_different_text_returns_zero(self) -> None:
        base = _make_success_run("x", "Paris")
        cand = _make_success_run("x", "Lyon")
        assert self.metric.evaluate(base, cand) == 0.0

    def test_failed_baseline_returns_none(self) -> None:
        base = _make_failed_run("x")
        cand = _make_success_run("x", "Paris")
        assert self.metric.evaluate(base, cand) is None

    def test_failed_candidate_returns_none(self) -> None:
        base = _make_success_run("x", "Paris")
        cand = _make_failed_run("x")
        assert self.metric.evaluate(base, cand) is None

    def test_metric_name(self) -> None:
        assert self.metric.metric_name == "exact_token_agreement"


# ---------------------------------------------------------------------------
# NormalizedTextAgreementMetric
# ---------------------------------------------------------------------------

class TestNormalizedTextAgreementMetric:
    def setup_method(self) -> None:
        self.metric = NormalizedTextAgreementMetric()

    def test_identical_text_gives_one(self) -> None:
        base = _make_success_run("x", "the quick brown fox")
        cand = _make_success_run("x", "the quick brown fox")
        score = self.metric.evaluate(base, cand)
        assert score == pytest.approx(1.0)

    def test_completely_different_text_gives_zero(self) -> None:
        base = _make_success_run("x", "apple orange banana")
        cand = _make_success_run("x", "cat dog mouse")
        score = self.metric.evaluate(base, cand)
        assert score == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        base = _make_success_run("x", "the quick brown fox")
        cand = _make_success_run("x", "the quick red cat")
        score = self.metric.evaluate(base, cand)
        assert score is not None
        assert 0.0 < score < 1.0

    def test_both_empty_gives_one(self) -> None:
        base = _make_success_run("x", "   ", generated_tokens=1)
        cand = _make_success_run("x", "   ", generated_tokens=1)
        score = self.metric.evaluate(base, cand)
        assert score == pytest.approx(1.0)

    def test_failed_run_returns_none(self) -> None:
        base = _make_failed_run("x")
        cand = _make_success_run("x", "text")
        assert self.metric.evaluate(base, cand) is None

    def test_metric_name(self) -> None:
        assert self.metric.metric_name == "normalized_text_agreement"


# ---------------------------------------------------------------------------
# QualityEvaluator
# ---------------------------------------------------------------------------

class TestQualityEvaluator:
    def test_self_validation_returns_not_measured(self) -> None:
        evaluator = QualityEvaluator()
        runs = [_make_success_run("c001", "Paris")]
        result = evaluator.evaluate(baseline_runs=runs, candidate_runs=None)
        assert result.quality_measured is False
        assert result.qrr_percent is None
        assert result.baseline_score is None
        assert result.candidate_score is None

    def test_comparison_exact_match_gives_100_qrr(self) -> None:
        evaluator = QualityEvaluator(metrics=[ExactTokenAgreementMetric()])
        base = [_make_success_run("c001", "Paris")]
        cand = [_make_success_run("c001", "Paris")]
        result = evaluator.evaluate(baseline_runs=base, candidate_runs=cand)
        assert result.quality_measured is True
        assert result.qrr_percent is not None
        assert result.qrr_percent == pytest.approx(100.0)

    def test_comparison_no_match_gives_0_qrr(self) -> None:
        evaluator = QualityEvaluator(metrics=[ExactTokenAgreementMetric()])
        base = [_make_success_run("c001", "Paris")]
        cand = [_make_success_run("c001", "London")]
        result = evaluator.evaluate(baseline_runs=base, candidate_runs=cand)
        assert result.quality_measured is True
        assert result.qrr_percent is not None
        assert result.qrr_percent == pytest.approx(0.0)

    def test_all_failed_runs_returns_not_measured(self) -> None:
        evaluator = QualityEvaluator()
        base = [_make_failed_run("c001")]
        cand = [_make_failed_run("c001")]
        result = evaluator.evaluate(baseline_runs=base, candidate_runs=cand)
        assert result.quality_measured is False
        assert result.qrr_percent is None

    def test_multiple_cases_average(self) -> None:
        evaluator = QualityEvaluator(metrics=[ExactTokenAgreementMetric()])
        base = [
            _make_success_run("c001", "Paris"),
            _make_success_run("c002", "London"),
        ]
        cand = [
            _make_success_run("c001", "Paris"),   # match → 1.0
            _make_success_run("c002", "Berlin"),  # no match → 0.0
        ]
        result = evaluator.evaluate(baseline_runs=base, candidate_runs=cand)
        assert result.quality_measured is True
        assert result.qrr_percent is not None
        assert result.qrr_percent == pytest.approx(50.0, abs=0.1)

    def test_missing_candidate_case_handled(self) -> None:
        evaluator = QualityEvaluator(metrics=[ExactTokenAgreementMetric()])
        base = [
            _make_success_run("c001", "Paris"),
            _make_success_run("c002", "London"),
        ]
        # Only c001 in candidate — c002 is missing
        cand = [_make_success_run("c001", "Paris")]
        result = evaluator.evaluate(baseline_runs=base, candidate_runs=cand)
        assert result.quality_measured is True
        assert "c002" in " ".join(result.details)

    def test_qrr_never_synthetic_in_self_validation(self) -> None:
        """QRR must be None in self-validation mode even when baseline is identical to itself."""
        evaluator = QualityEvaluator()
        runs = [_make_success_run("c001", "Paris")]
        result = evaluator.evaluate(baseline_runs=runs, candidate_runs=None)
        assert result.qrr_percent is None
        assert result.quality_measured is False


# ---------------------------------------------------------------------------
# ValidationEvaluator — run_self_validation with mock runner
# ---------------------------------------------------------------------------

def _build_mock_runner(
    text_map: dict[str, str],
    default_text: str = "test output here",
) -> MagicMock:
    """Build a mock BaseRunner whose generate() returns pre-configured outputs."""
    mock = MagicMock()

    def _generate(prompt: str, max_new_tokens: int = 16) -> MagicMock:
        # Map prompt → generated text
        generated = text_map.get(prompt, default_text)
        result = MagicMock()
        result.status = ExecutionStatus.SUCCESS
        result.generated_text = generated
        result.generated_tokens = len(generated.split())
        result.input_tokens = len(prompt.split())
        result.error = None
        return result

    mock.generate.side_effect = _generate
    return mock


def _build_failing_runner(error_msg: str = "GPU error") -> MagicMock:
    """Build a mock BaseRunner whose generate() always raises an exception."""
    mock = MagicMock()
    mock.generate.side_effect = RuntimeError(error_msg)
    return mock


def _build_model_spec(model_id: str = "test/model", commit_sha: str = "abc123def456" * 3) -> MagicMock:
    spec = MagicMock()
    spec.model_id = model_id
    spec.commit_sha = commit_sha
    return spec


class TestValidationEvaluatorSelfValidation:
    def test_all_pass_gives_pass_verdict(self) -> None:
        cases = [
            ValidationCase(case_id="c001", prompt="Prompt A", max_new_tokens=8, critical=True),
            ValidationCase(case_id="c002", prompt="Prompt B", max_new_tokens=8, critical=False),
        ]
        config = ValidationConfig(cases=cases, max_new_tokens=8, deterministic=False)
        evaluator = ValidationEvaluator(config=config)

        runner = _build_mock_runner({"Prompt A": "Good output", "Prompt B": "Also good"})
        model = _build_model_spec()
        report = evaluator.run_self_validation(runner=runner, model=model)

        assert report.verdict == ValidationVerdict.PASS
        assert report.correctness_passed is True
        assert report.quality_measured is False
        assert report.qrr_percent is None
        assert report.mode == ValidationMode.SELF_VALIDATION
        assert report.cases_total == 2
        assert report.cases_completed == 2
        assert report.cases_failed == 0
        assert report.critical_cases_failed == 0

    def test_critical_failure_gives_fail_verdict(self) -> None:
        cases = [
            ValidationCase(case_id="c001", prompt="Critical prompt", max_new_tokens=8, critical=True),
        ]
        config = ValidationConfig(cases=cases, max_new_tokens=8, deterministic=False)
        evaluator = ValidationEvaluator(config=config)

        runner = _build_failing_runner("CUDA error")
        model = _build_model_spec()
        report = evaluator.run_self_validation(runner=runner, model=model)

        assert report.verdict == ValidationVerdict.FAIL
        assert report.critical_cases_failed >= 1

    def test_non_critical_failure_gives_fail_verdict(self) -> None:
        cases = [
            ValidationCase(case_id="c001", prompt="Safe", max_new_tokens=8, critical=False),
        ]
        config = ValidationConfig(cases=cases, max_new_tokens=8, deterministic=False)
        evaluator = ValidationEvaluator(config=config)

        runner = _build_failing_runner()
        model = _build_model_spec()
        report = evaluator.run_self_validation(runner=runner, model=model)

        # Non-critical failure still fails correctness gate
        assert report.verdict in (ValidationVerdict.FAIL, ValidationVerdict.INCONCLUSIVE)

    def test_qrr_always_none_in_self_validation(self) -> None:
        """Even if runner succeeds with perfect output, QRR must remain None."""
        cases = [
            ValidationCase(case_id="c001", prompt="Hello", max_new_tokens=8, critical=False),
        ]
        config = ValidationConfig(cases=cases, max_new_tokens=8, deterministic=False)
        evaluator = ValidationEvaluator(config=config)
        runner = _build_mock_runner({"Hello": "World"})
        model = _build_model_spec()
        report = evaluator.run_self_validation(runner=runner, model=model)

        assert report.qrr_percent is None
        assert report.quality_measured is False

    def test_report_has_case_results(self) -> None:
        cases = [
            ValidationCase(case_id="c001", prompt="Prompt", max_new_tokens=8, critical=True),
        ]
        config = ValidationConfig(cases=cases, max_new_tokens=8, deterministic=False)
        evaluator = ValidationEvaluator(config=config)
        runner = _build_mock_runner({"Prompt": "response"})
        model = _build_model_spec()
        report = evaluator.run_self_validation(runner=runner, model=model)

        assert len(report.case_results) == 1
        assert report.case_results[0].case_id == "c001"

    def test_model_id_and_revision_in_report(self) -> None:
        cases = [ValidationCase(case_id="c001", prompt="Hi", max_new_tokens=4, critical=False)]
        config = ValidationConfig(cases=cases, max_new_tokens=4, deterministic=False)
        evaluator = ValidationEvaluator(config=config)
        runner = _build_mock_runner({"Hi": "Hello"})
        model = _build_model_spec(model_id="my/model", commit_sha="sha123abc")
        report = evaluator.run_self_validation(runner=runner, model=model)

        assert report.model_id == "my/model"
        assert report.baseline_revision == "sha123abc"
        assert report.candidate_revision is None

    def test_reasons_populated(self) -> None:
        cases = [ValidationCase(case_id="c001", prompt="P", max_new_tokens=4, critical=True)]
        config = ValidationConfig(cases=cases, max_new_tokens=4, deterministic=False)
        evaluator = ValidationEvaluator(config=config)
        runner = _build_mock_runner({"P": "response text"})
        model = _build_model_spec()
        report = evaluator.run_self_validation(runner=runner, model=model)

        assert len(report.reasons) >= 1


# ---------------------------------------------------------------------------
# ValidationEvaluator — run_comparison with mock runners
# ---------------------------------------------------------------------------

class TestValidationEvaluatorComparison:
    def _cases(self) -> list[ValidationCase]:
        return [
            ValidationCase(case_id="c001", prompt="Prompt A", max_new_tokens=8, critical=True),
            ValidationCase(case_id="c002", prompt="Prompt B", max_new_tokens=8, critical=False),
        ]

    def test_identical_outputs_give_pass_with_qrr(self) -> None:
        cases = self._cases()
        config = ValidationConfig(
            cases=cases, max_new_tokens=8, deterministic=False, quality_threshold=0.9
        )
        evaluator = ValidationEvaluator(config=config)

        base_runner = _build_mock_runner({"Prompt A": "Paris", "Prompt B": "French"})
        cand_runner = _build_mock_runner({"Prompt A": "Paris", "Prompt B": "French"})
        base_model = _build_model_spec(commit_sha="base_sha_001" * 2)
        cand_model = _build_model_spec(commit_sha="cand_sha_002" * 2)

        report = evaluator.run_comparison(
            baseline_runner=base_runner,
            candidate_runner=cand_runner,
            baseline_model=base_model,
            candidate_model=cand_model,
        )

        assert report.verdict == ValidationVerdict.PASS
        assert report.quality_measured is True
        assert report.qrr_percent is not None
        assert report.qrr_percent > 90.0
        assert report.mode == ValidationMode.COMPARISON

    def test_candidate_failure_gives_fail(self) -> None:
        cases = self._cases()
        config = ValidationConfig(cases=cases, max_new_tokens=8, deterministic=False)
        evaluator = ValidationEvaluator(config=config)

        base_runner = _build_mock_runner({"Prompt A": "Paris", "Prompt B": "French"})
        cand_runner = _build_failing_runner("GPU crash")
        base_model = _build_model_spec(commit_sha="base_sha" * 5)
        cand_model = _build_model_spec(commit_sha="cand_sha" * 5)

        report = evaluator.run_comparison(
            baseline_runner=base_runner,
            candidate_runner=cand_runner,
            baseline_model=base_model,
            candidate_model=cand_model,
        )

        assert report.verdict == ValidationVerdict.FAIL

    def test_low_qrr_gives_fail(self) -> None:
        """QRR below threshold should produce FAIL verdict."""
        cases = [ValidationCase(case_id="c001", prompt="Q", max_new_tokens=8, critical=False)]
        config = ValidationConfig(
            cases=cases, max_new_tokens=8, deterministic=False, quality_threshold=0.99
        )
        evaluator = ValidationEvaluator(
            config=config,
            quality_evaluator=QualityEvaluator(metrics=[ExactTokenAgreementMetric()]),
        )
        base_runner = _build_mock_runner({"Q": "answer one"})
        cand_runner = _build_mock_runner({"Q": "completely different"})
        base_model = _build_model_spec(commit_sha="b" * 20)
        cand_model = _build_model_spec(commit_sha="c" * 20)

        report = evaluator.run_comparison(
            baseline_runner=base_runner,
            candidate_runner=cand_runner,
            baseline_model=base_model,
            candidate_model=cand_model,
        )

        assert report.verdict == ValidationVerdict.FAIL
        assert report.qrr_percent is not None


# ---------------------------------------------------------------------------
# CLI validate command — offline unit tests (preflight gate)
# ---------------------------------------------------------------------------

class TestCLIValidateCommand:
    """Test rocmhub validate CLI command in offline mode without AMD GPU."""

    def _make_no_accelerator_capability_report(self) -> MagicMock:
        from rocmhub.core.types import CapabilityReport, EvaluationVerdict
        report = MagicMock(spec=CapabilityReport)
        report.verdict = EvaluationVerdict.NO_ACCELERATOR
        reason = MagicMock()
        reason.severity = MagicMock()
        reason.severity.value = "info"
        reason.code = "NO_AMD_GPU"
        reason.message = "No AMD GPU detected on this host."
        report.reasons = [reason]
        report.warnings = []
        return report

    def _make_model_spec(self, model_id: str = "test/model") -> MagicMock:
        spec = MagicMock()
        spec.model_id = model_id
        spec.commit_sha = "deadbeefcafe1234" * 2
        spec.requested_revision = "main"
        return spec

    def _run_cli(self, args: list[str]) -> tuple[int, str, str]:
        """Run CLI via main() with patched I/O; returns (exit_code, stdout, stderr)."""
        import io

        from rocmhub.cli.main import main

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        with (
            patch("sys.stdout", stdout_buf),
            patch("sys.stderr", stderr_buf),
        ):
            code = main(args)
        return code, stdout_buf.getvalue(), stderr_buf.getvalue()

    def test_validate_no_model_id_returns_error(self) -> None:
        code, _, stderr = self._run_cli(["validate"])
        assert code == 1
        assert "model_id" in stderr.lower() or "error" in stderr.lower()

    @patch("rocmhub.cli.main.CapabilityEvaluator")
    @patch("rocmhub.cli.main.SystemObserver")
    @patch("rocmhub.cli.main.ModelInspector")
    @patch("rocmhub.cli.main.HuggingFaceModelSource")
    def test_validate_no_accelerator_exits_2(
        self,
        mock_hf_source: MagicMock,
        mock_inspector_cls: MagicMock,
        mock_observer_cls: MagicMock,
        mock_evaluator_cls: MagicMock,
    ) -> None:
        mock_inspector_cls.return_value.inspect.return_value = self._make_model_spec()
        mock_observer_cls.return_value.observe.return_value = MagicMock()
        mock_evaluator_cls.return_value.evaluate.return_value = (
            self._make_no_accelerator_capability_report()
        )

        code, stdout, _ = self._run_cli(["validate", "test/model"])
        assert code == 2

    @patch("rocmhub.cli.main.CapabilityEvaluator")
    @patch("rocmhub.cli.main.SystemObserver")
    @patch("rocmhub.cli.main.ModelInspector")
    @patch("rocmhub.cli.main.HuggingFaceModelSource")
    def test_validate_no_accelerator_outputs_not_measured_text(
        self,
        mock_hf_source: MagicMock,
        mock_inspector_cls: MagicMock,
        mock_observer_cls: MagicMock,
        mock_evaluator_cls: MagicMock,
    ) -> None:
        mock_inspector_cls.return_value.inspect.return_value = self._make_model_spec()
        mock_observer_cls.return_value.observe.return_value = MagicMock()
        mock_evaluator_cls.return_value.evaluate.return_value = (
            self._make_no_accelerator_capability_report()
        )

        code, stdout, _ = self._run_cli(["validate", "test/model"])
        # Human-readable: should mention skip or preflight
        assert "preflight" in stdout.lower() or "skipped" in stdout.lower() or "not" in stdout.lower()

    @patch("rocmhub.cli.main.CapabilityEvaluator")
    @patch("rocmhub.cli.main.SystemObserver")
    @patch("rocmhub.cli.main.ModelInspector")
    @patch("rocmhub.cli.main.HuggingFaceModelSource")
    def test_validate_no_accelerator_json_output_is_not_measured(
        self,
        mock_hf_source: MagicMock,
        mock_inspector_cls: MagicMock,
        mock_observer_cls: MagicMock,
        mock_evaluator_cls: MagicMock,
    ) -> None:
        import json

        mock_inspector_cls.return_value.inspect.return_value = self._make_model_spec()
        mock_observer_cls.return_value.observe.return_value = MagicMock()
        mock_evaluator_cls.return_value.evaluate.return_value = (
            self._make_no_accelerator_capability_report()
        )

        code, stdout, _ = self._run_cli(["validate", "test/model", "--json"])
        assert code == 2
        data = json.loads(stdout)
        assert data["verdict"] == "NOT_MEASURED"
        assert data["quality_measured"] is False
        assert data["qrr_percent"] is None
        assert data["correctness_passed"] is None
        assert data["cases_completed"] == 0

    @patch("rocmhub.cli.main.CapabilityEvaluator")
    @patch("rocmhub.cli.main.SystemObserver")
    @patch("rocmhub.cli.main.ModelInspector")
    @patch("rocmhub.cli.main.HuggingFaceModelSource")
    def test_validate_no_model_downloads_on_no_accelerator(
        self,
        mock_hf_source: MagicMock,
        mock_inspector_cls: MagicMock,
        mock_observer_cls: MagicMock,
        mock_evaluator_cls: MagicMock,
    ) -> None:
        """Validation must not call runner.load() when preflight fails."""
        from rocmhub.cli.main import main

        mock_inspector_cls.return_value.inspect.return_value = self._make_model_spec()
        mock_observer_cls.return_value.observe.return_value = MagicMock()
        mock_evaluator_cls.return_value.evaluate.return_value = (
            self._make_no_accelerator_capability_report()
        )

        with patch("rocmhub.cli.main.HuggingFaceRunner") as mock_runner_cls:
            import io
            with patch("sys.stdout", io.StringIO()), patch("sys.stderr", io.StringIO()):
                main(["validate", "test/model"])
            # Runner should never be instantiated when preflight fails
            mock_runner_cls.assert_not_called()

    def test_validate_accepts_model_flag(self) -> None:
        """--model flag should be equivalent to positional model_id."""
        from rocmhub.cli.main import build_parser
        parser = build_parser()
        args = parser.parse_args(["validate", "--model", "some/model"])
        assert args.model_opt == "some/model"
        assert args.model_id is None

    def test_validate_parser_defaults(self) -> None:
        from rocmhub.cli.main import build_parser
        parser = build_parser()
        args = parser.parse_args(["validate", "test/model"])
        assert args.revision == "main"
        assert args.device == 0
        assert args.precision == "fp16"
        assert args.max_new_tokens == 16
        assert args.json is False

    def test_validate_parser_custom_args(self) -> None:
        from rocmhub.cli.main import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "validate", "test/model",
            "--revision", "v2",
            "--device", "1",
            "--precision", "bf16",
            "--max-new-tokens", "32",
            "--json",
        ])
        assert args.revision == "v2"
        assert args.device == 1
        assert args.precision == "bf16"
        assert args.max_new_tokens == 32
        assert args.json is True


# ---------------------------------------------------------------------------
# _format_validation_report formatter
# ---------------------------------------------------------------------------

class TestFormatValidationReport:
    def _make_not_measured(self) -> ValidationReport:
        return ValidationReport(
            mode=ValidationMode.SELF_VALIDATION,
            model_id="test/model",
            baseline_revision="abc123" * 6,
            verdict=ValidationVerdict.NOT_MEASURED,
            correctness_passed=None,
            quality_measured=False,
            qrr_percent=None,
            cases_total=3,
            cases_completed=0,
            cases_failed=0,
            critical_cases_failed=0,
            reasons=["No accelerator detected."],
        )

    def _make_pass_report(self) -> ValidationReport:
        run = _make_success_run("c001", "Paris", generated_tokens=2)
        return ValidationReport(
            mode=ValidationMode.SELF_VALIDATION,
            model_id="test/model",
            baseline_revision="def456" * 6,
            verdict=ValidationVerdict.PASS,
            correctness_passed=True,
            quality_measured=False,
            qrr_percent=None,
            cases_total=1,
            cases_completed=1,
            cases_failed=0,
            critical_cases_failed=0,
            case_results=[run],
            reasons=["All gates passed."],
        )

    def test_not_measured_shows_not_measured_verdict(self) -> None:
        from rocmhub.cli.main import _format_validation_report
        output = _format_validation_report(self._make_not_measured())
        assert "NOT_MEASURED" in output

    def test_not_measured_shows_na_qrr(self) -> None:
        from rocmhub.cli.main import _format_validation_report
        output = _format_validation_report(self._make_not_measured())
        assert "n/a" in output

    def test_pass_report_shows_pass(self) -> None:
        from rocmhub.cli.main import _format_validation_report
        output = _format_validation_report(self._make_pass_report())
        assert "PASS" in output

    def test_pass_report_shows_correctness_passed(self) -> None:
        from rocmhub.cli.main import _format_validation_report
        output = _format_validation_report(self._make_pass_report())
        assert "passed" in output.lower()

    def test_pass_report_shows_case_results(self) -> None:
        from rocmhub.cli.main import _format_validation_report
        output = _format_validation_report(self._make_pass_report())
        assert "c001" in output

    def test_pass_report_shows_model_id(self) -> None:
        from rocmhub.cli.main import _format_validation_report
        output = _format_validation_report(self._make_pass_report())
        assert "test/model" in output
