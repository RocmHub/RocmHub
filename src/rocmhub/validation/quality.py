"""Quality evaluation metrics and Quality Retention Rate (QRR) computation."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field

from rocmhub.core.types import ExecutionStatus, ValidationRunResult


class QualityMetric(Protocol):
    """Protocol for comparative quality metrics between baseline and candidate runs."""

    @property
    def metric_name(self) -> str:
        """Name of the quality metric."""
        ...

    def evaluate(
        self,
        baseline: ValidationRunResult,
        candidate: ValidationRunResult,
    ) -> Optional[float]:
        """Compute metric score comparing candidate to baseline.

        Returns:
            Normalized numeric score (typically 0.0 - 1.0), or None if not applicable.
        """
        ...


class ExactTokenAgreementMetric:
    """Evaluates exact character and token agreement between deterministic runs."""

    metric_name = "exact_token_agreement"

    def evaluate(
        self,
        baseline: ValidationRunResult,
        candidate: ValidationRunResult,
    ) -> Optional[float]:
        if (
            baseline.status != ExecutionStatus.SUCCESS
            or candidate.status != ExecutionStatus.SUCCESS
            or baseline.generated_text is None
            or candidate.generated_text is None
        ):
            return None

        return 1.0 if baseline.generated_text == candidate.generated_text else 0.0


class NormalizedTextAgreementMetric:
    """Evaluates normalized token overlap ratio between baseline and candidate generations."""

    metric_name = "normalized_text_agreement"

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        cleaned = re.sub(r"[^\w\s]", " ", text.lower())
        return [w for w in cleaned.split() if w]

    def evaluate(
        self,
        baseline: ValidationRunResult,
        candidate: ValidationRunResult,
    ) -> Optional[float]:
        if (
            baseline.status != ExecutionStatus.SUCCESS
            or candidate.status != ExecutionStatus.SUCCESS
            or baseline.generated_text is None
            or candidate.generated_text is None
        ):
            return None

        base_tokens = set(self._tokenize(baseline.generated_text))
        cand_tokens = set(self._tokenize(candidate.generated_text))

        if not base_tokens and not cand_tokens:
            return 1.0
        if not base_tokens or not cand_tokens:
            return 0.0

        intersection = base_tokens.intersection(cand_tokens)
        union = base_tokens.union(cand_tokens)
        return float(len(intersection)) / float(len(union))


class QualityResult(BaseModel):
    """Consolidated outcome of quality metric evaluation and QRR."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    quality_measured: bool = Field(default=False, description="True if quality was measured in COMPARISON mode")
    baseline_score: Optional[float] = Field(default=None, description="Average quality score of baseline")
    candidate_score: Optional[float] = Field(default=None, description="Average quality score of candidate")
    qrr_percent: Optional[float] = Field(
        default=None,
        description="Quality Retention Rate percentage (candidate_score / baseline_score * 100). None if not measured.",
    )
    metric_scores: Dict[str, float] = Field(default_factory=dict, description="Per-metric average scores")
    details: List[str] = Field(default_factory=list, description="Diagnostic evaluation notes")


class QualityEvaluator:
    """Orchestrates quality metric calculations and computes Quality Retention Rate (QRR).

    Key Invariant:
    - In SELF_VALIDATION mode: quality_measured is False, and QRR is strictly None.
      (Never assigns fake 100% simply because baseline is compared with itself).
    - In COMPARISON mode: QRR = (candidate_score / baseline_score) * 100.
      If baseline score <= 0.0, QRR is None (safe division).
    """

    def __init__(self, metrics: Optional[Sequence[QualityMetric]] = None) -> None:
        self.metrics: Sequence[QualityMetric] = (
            metrics
            if metrics is not None
            else [ExactTokenAgreementMetric(), NormalizedTextAgreementMetric()]
        )

    def evaluate(
        self,
        baseline_runs: Sequence[ValidationRunResult],
        candidate_runs: Optional[Sequence[ValidationRunResult]] = None,
    ) -> QualityResult:
        """Evaluate quality metrics across baseline and optional candidate runs.

        Args:
            baseline_runs: Execution results from baseline runner.
            candidate_runs: Optional execution results from candidate runner.

        Returns:
            QualityResult with QRR and individual metric scores.
        """
        # Self-validation mode: no candidate exists
        if candidate_runs is None:
            return QualityResult(
                quality_measured=False,
                baseline_score=None,
                candidate_score=None,
                qrr_percent=None,
                details=["Self-validation mode: comparative quality metric and QRR were not measured."],
            )

        cand_map = {r.case_id: r for r in candidate_runs}
        metric_sums: Dict[str, float] = {}
        metric_counts: Dict[str, int] = {}
        details: List[str] = []

        for b_run in baseline_runs:
            c_run = cand_map.get(b_run.case_id)
            if not c_run:
                details.append(f"[{b_run.case_id}] Missing candidate run result.")
                continue

            for m in self.metrics:
                score = m.evaluate(b_run, c_run)
                if score is not None:
                    metric_sums[m.metric_name] = metric_sums.get(m.metric_name, 0.0) + score
                    metric_counts[m.metric_name] = metric_counts.get(m.metric_name, 0) + 1

        if not metric_counts:
            return QualityResult(
                quality_measured=False,
                baseline_score=None,
                candidate_score=None,
                qrr_percent=None,
                details=["No valid pairs of baseline/candidate outputs available to compute quality."],
            )

        metric_averages: Dict[str, float] = {
            name: metric_sums[name] / float(metric_counts[name])
            for name in metric_counts
        }

        # Average candidate score across metrics
        cand_score = sum(metric_averages.values()) / float(len(metric_averages))
        # For agreement metrics, baseline ideal reference score is 1.0
        base_score = 1.0

        # Safe QRR computation
        qrr_val: Optional[float] = None
        if base_score > 0.0:
            qrr_val = round((cand_score / base_score) * 100.0, 2)

        return QualityResult(
            quality_measured=True,
            baseline_score=base_score,
            candidate_score=cand_score,
            qrr_percent=qrr_val,
            metric_scores=metric_averages,
            details=details,
        )
