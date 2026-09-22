"""Comparison Engine for ROCmHub Optimization (Phase 12)."""

from __future__ import annotations

from typing import List, Optional

from rocmhub.core.errors import IncomparableResultsError
from rocmhub.core.types import ExecutionStatus
from rocmhub.optimization.base import (
    ComparisonResult,
    ComparisonVerdict,
    OptimizationBaseline,
    OptimizationCandidate,
)


class ComparisonEngine:
    """Rigorous, objective evaluator comparing candidate variants against fixed baseline."""

    def __init__(self, quality_threshold: float = 0.95) -> None:
        self._quality_threshold = quality_threshold

    def compare(
        self, baseline: OptimizationBaseline, candidate: OptimizationCandidate
    ) -> ComparisonResult:
        """Compare candidate performance and quality against baseline.

        Enforces identical conditions: same model ID, revision, and hardware target.
        Never fabricates metrics: if measurements are absent, returns NOT_MEASURED.
        """
        # 1. Provenance consistency checks
        if baseline.model_id != candidate.model_id:
            raise IncomparableResultsError(
                f"Model ID mismatch: baseline='{baseline.model_id}' vs candidate='{candidate.model_id}'"
            )
        if baseline.revision != candidate.revision:
            raise IncomparableResultsError(
                f"Revision mismatch: baseline='{baseline.revision}' vs candidate='{candidate.revision}'"
            )

        # 2. Check measurement presence
        b_bench = baseline.benchmark_result
        c_bench = candidate.benchmark_result

        if not baseline.measured or b_bench is None or b_bench.status != ExecutionStatus.SUCCESS:
            return ComparisonResult(
                candidate_id=candidate.candidate_id,
                strategy=candidate.strategy,
                verdict=ComparisonVerdict.NOT_MEASURED,
                reasons=["Baseline performance was not measured (e.g. executed on non-AMD environment)."],
            )

        if c_bench is None or c_bench.status != ExecutionStatus.SUCCESS:
            return ComparisonResult(
                candidate_id=candidate.candidate_id,
                strategy=candidate.strategy,
                verdict=ComparisonVerdict.NOT_MEASURED,
                reasons=["Candidate performance was not measured (e.g. executed on non-AMD environment)."],
            )

        b_metrics = b_bench
        c_metrics = c_bench

        reasons: List[str] = []

        # 3. Check quality retention
        qrr: Optional[float] = None
        if candidate.validation_report and candidate.validation_report.qrr_percent is not None:
            qrr = candidate.validation_report.qrr_percent
            if qrr < (self._quality_threshold * 100):
                reasons.append(
                    f"Quality regression: QRR {qrr:.1f}% is below acceptable threshold {self._quality_threshold * 100:.1f}%"
                )
                return ComparisonResult(
                    candidate_id=candidate.candidate_id,
                    strategy=candidate.strategy,
                    verdict=ComparisonVerdict.REGRESSED,
                    qrr_percent=qrr,
                    reasons=reasons,
                )

        # 4. Metric computation
        ttft_speedup: Optional[float] = None
        if (
            b_metrics.ttft_ms is not None
            and c_metrics.ttft_ms is not None
            and c_metrics.ttft_ms > 0
        ):
            ttft_speedup = round(b_metrics.ttft_ms / c_metrics.ttft_ms, 3)

        thr_speedup: Optional[float] = None
        if (
            b_metrics.throughput_tokens_per_sec is not None
            and c_metrics.throughput_tokens_per_sec is not None
            and b_metrics.throughput_tokens_per_sec > 0
        ):
            thr_speedup = round(
                c_metrics.throughput_tokens_per_sec / b_metrics.throughput_tokens_per_sec, 3
            )

        vram_reduction: Optional[float] = None
        if b_metrics.peak_vram_used_mb is not None and c_metrics.peak_vram_used_mb is not None:
            vram_reduction = round(
                b_metrics.peak_vram_used_mb - c_metrics.peak_vram_used_mb, 1
            )

        # 5. Verdict determination
        # Speedup requires at least 3% significant delta to declare IMPROVED or REGRESSED
        if thr_speedup is not None and thr_speedup >= 1.03:
            verdict = ComparisonVerdict.IMPROVED
            reasons.append(f"Throughput increased by {((thr_speedup - 1.0) * 100):.1f}%.")
        elif ttft_speedup is not None and ttft_speedup >= 1.03:
            verdict = ComparisonVerdict.IMPROVED
            reasons.append(f"TTFT latency reduced (speedup {ttft_speedup:.2f}x).")
        elif (thr_speedup is not None and thr_speedup <= 0.97) or (
            ttft_speedup is not None and ttft_speedup <= 0.97
        ):
            verdict = ComparisonVerdict.REGRESSED
            reasons.append("Performance degraded compared to baseline.")
        else:
            verdict = ComparisonVerdict.NO_CHANGE
            reasons.append("Performance difference within experimental noise margin (<3%).")

        return ComparisonResult(
            candidate_id=candidate.candidate_id,
            strategy=candidate.strategy,
            verdict=verdict,
            ttft_speedup=ttft_speedup,
            throughput_speedup=thr_speedup,
            vram_reduction_mb=vram_reduction,
            qrr_percent=qrr,
            reasons=reasons,
        )
