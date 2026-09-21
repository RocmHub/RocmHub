"""Statistical calculation and aggregation of benchmark performance metrics."""

from __future__ import annotations

from typing import List, Optional, Sequence

from rocmhub.benchmarks.base import BenchmarkConfig, BenchmarkRunMeasurement
from rocmhub.core.types import BenchmarkResult, ExecutionStatus


def compute_percentile(values: Sequence[float], p: float) -> float:
    """Compute the p-th percentile of values using deterministic linear interpolation.

    Matches numpy.percentile(values, p, method='linear') without requiring NumPy.

    Args:
        values: Non-empty sequence of numeric values.
        p: Target percentile rank between 0.0 and 100.0 inclusive.

    Returns:
        Interpolated percentile value.

    Raises:
        ValueError: If values sequence is empty or p is outside [0, 100].
    """
    if not values:
        raise ValueError("Cannot compute percentile of an empty sequence")
    if not (0.0 <= p <= 100.0):
        raise ValueError(f"Percentile rank p must be between 0.0 and 100.0, got {p}")

    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n == 1:
        return float(sorted_vals[0])

    virtual_idx = (n - 1) * (p / 100.0)
    lower_idx = int(virtual_idx)
    upper_idx = min(lower_idx + 1, n - 1)
    weight = virtual_idx - lower_idx

    return float(sorted_vals[lower_idx] * (1.0 - weight) + sorted_vals[upper_idx] * weight)


def compute_median(values: Sequence[float]) -> float:
    """Compute the 50th percentile (median) of a numeric sequence."""
    return compute_percentile(values, 50.0)


class MetricsCalculator:
    """Stateless aggregator for single-run measurements and multi-run benchmark summaries."""

    @staticmethod
    def calculate_run_measurement(
        run_index: int,
        is_warmup: bool,
        started_at_ns: int,
        finished_at_ns: int,
        token_timestamps_ns: Sequence[int],
        input_tokens: Optional[int],
        generated_tokens: Optional[int],
        peak_vram_used_mb: Optional[float],
        status: ExecutionStatus = ExecutionStatus.SUCCESS,
        error: Optional[str] = None,
    ) -> BenchmarkRunMeasurement:
        """Derive TTFT, ITL sequence, and total latency for a single execution run.

        Exact Metric Definitions:
        - TTFT: Time To First Token = (token_timestamps_ns[0] - started_at_ns) / 1e6 ms.
          Strictly represents moment of request start to moment of first generated token emission.
        - ITL: Inter-Token Latencies = (t_{i+1} - t_i) / 1e6 ms for all consecutive tokens.
          Strictly excludes TTFT. If generated_tokens < 2, ITL is empty.
        - Total Latency: Full duration = (finished_at_ns - started_at_ns) / 1e6 ms.

        Args:
            run_index: Iteration number.
            is_warmup: True if run is an untimed warmup cycle.
            started_at_ns: Monotonic nanosecond timestamp of request start.
            finished_at_ns: Monotonic nanosecond timestamp of generation completion.
            token_timestamps_ns: Sequence of timestamps when generated tokens were emitted.
            input_tokens: Count of prompt tokens.
            generated_tokens: Count of newly decoded tokens.
            peak_vram_used_mb: Peak allocator-observed GPU memory in MB.
            status: SUCCESS or FAILED.
            error: Error message if failed.

        Returns:
            Structured BenchmarkRunMeasurement evidence.
        """
        total_latency_ms = max(0.0, (finished_at_ns - started_at_ns) / 1_000_000.0)

        ttft_ms: Optional[float] = None
        inter_token_latencies_ms: List[float] = []
        first_token_at_ns: Optional[int] = None

        if status == ExecutionStatus.SUCCESS and token_timestamps_ns:
            first_token_at_ns = int(token_timestamps_ns[0])
            ttft_ms = max(0.0, (first_token_at_ns - started_at_ns) / 1_000_000.0)

            # Inter-Token Latency (ITL): consecutive deltas t_{i+1} - t_i (TTFT excluded)
            if len(token_timestamps_ns) >= 2:
                for i in range(1, len(token_timestamps_ns)):
                    delta_ms = max(0.0, (token_timestamps_ns[i] - token_timestamps_ns[i - 1]) / 1_000_000.0)
                    inter_token_latencies_ms.append(delta_ms)

        return BenchmarkRunMeasurement(
            run_index=run_index,
            is_warmup=is_warmup,
            status=status,
            input_tokens=input_tokens,
            generated_tokens=generated_tokens,
            started_at_ns=started_at_ns,
            first_token_at_ns=first_token_at_ns,
            finished_at_ns=finished_at_ns,
            ttft_ms=ttft_ms,
            inter_token_latencies_ms=inter_token_latencies_ms,
            total_latency_ms=total_latency_ms,
            peak_vram_used_mb=peak_vram_used_mb,
            error=error,
        )

    @classmethod
    def aggregate(
        cls,
        config: BenchmarkConfig,
        measurements: Sequence[BenchmarkRunMeasurement],
        model_id: Optional[str] = None,
        model_revision: Optional[str] = None,
        runtime_name: Optional[str] = None,
        raw_measurements_reference: Optional[str] = None,
    ) -> BenchmarkResult:
        """Aggregate multiple measurement runs into a canonical BenchmarkResult.

        Aggregation Rules:
        - Warmup runs are strictly EXCLUDED from all performance summary metrics.
        - If any measurement run has status FAILED, the overall benchmark is FAILED
          and all performance metrics remain None.
        - TTFT headline metric: median across valid measurement runs.
        - Total latency headline metric: median across valid measurement runs.
        - Throughput: median of per-run end-to-end throughput
          (generated_tokens / (last_token_ns - started_at_ns)).
        - Peak memory: maximum allocator-observed memory across valid measurement runs.
        - ITL metrics: concatenated ITL samples across valid measurement runs
          (mean, p50, p90, p99). If generated_tokens < 2, ITL is None.

        Args:
            config: Benchmark configuration.
            measurements: Complete list of warmup and measurement runs.
            model_id: Model repository identifier.
            model_revision: Immutable Git commit SHA.
            runtime_name: Runtime adapter identifier.
            raw_measurements_reference: Optional reference to saved raw runs.

        Returns:
            Validated BenchmarkResult summary.
        """
        # 1. Separate warmup runs from measurement runs
        measurement_runs = [m for m in measurements if not m.is_warmup]
        requested_runs = config.measurement_runs
        completed_runs = sum(1 for m in measurement_runs if m.status == ExecutionStatus.SUCCESS)
        failed_runs = sum(1 for m in measurement_runs if m.status == ExecutionStatus.FAILED)

        # 2. Strict Partial Failure Rule: if any measurement run failed, benchmark fails
        if failed_runs > 0 or completed_runs < requested_runs:
            first_err = next((m.error for m in measurement_runs if m.error), "Unknown measurement run failure")
            return BenchmarkResult(
                status=ExecutionStatus.FAILED,
                error_message=f"Benchmark run failure: {failed_runs} of {requested_runs} runs failed. First error: {first_err}",
                model_id=model_id,
                model_revision=model_revision,
                device_id=config.device_id,
                runtime_name=runtime_name,
                precision=config.precision,
                warmup_runs=config.warmup_runs,
                measurement_runs_requested=requested_runs,
                measurement_runs_completed=completed_runs,
                failed_runs=failed_runs,
                raw_measurements_reference=raw_measurements_reference,
            )

        # 3. Aggregate headline metrics across successful runs
        ttft_samples = [m.ttft_ms for m in measurement_runs if m.ttft_ms is not None]
        median_ttft = compute_median(ttft_samples) if ttft_samples else None

        latency_samples = [m.total_latency_ms for m in measurement_runs]
        median_total_latency = compute_median(latency_samples) if latency_samples else None

        # Per-run end-to-end throughput: generated_tokens / ((last_token_at - started_at) in seconds)
        throughput_samples: List[float] = []
        for m in measurement_runs:
            if m.generated_tokens is not None and m.generated_tokens > 0:
                # If inter-token latencies exist, last token is first_token + sum(inter_token_latencies)
                if m.first_token_at_ns is not None:
                    last_token_ns = m.first_token_at_ns + int(sum(m.inter_token_latencies_ms) * 1_000_000)
                    e2e_duration_s = max(1e-9, (last_token_ns - m.started_at_ns) / 1_000_000_000.0)
                else:
                    e2e_duration_s = max(1e-9, (m.finished_at_ns - m.started_at_ns) / 1_000_000_000.0)
                throughput_samples.append(m.generated_tokens / e2e_duration_s)

        median_throughput = compute_median(throughput_samples) if throughput_samples else None

        # Peak allocator memory: maximum observed across measurement runs
        memory_samples = [m.peak_vram_used_mb for m in measurement_runs if m.peak_vram_used_mb is not None]
        peak_memory = max(memory_samples) if memory_samples else None

        # Inter-Token Latency (ITL) distribution: pooled raw samples across measurement runs
        all_itl_samples: List[float] = []
        for m in measurement_runs:
            all_itl_samples.extend(m.inter_token_latencies_ms)

        itl_mean: Optional[float] = None
        itl_p50: Optional[float] = None
        itl_p90: Optional[float] = None
        itl_p99: Optional[float] = None

        if all_itl_samples:
            itl_mean = sum(all_itl_samples) / float(len(all_itl_samples))
            itl_p50 = compute_percentile(all_itl_samples, 50.0)
            itl_p90 = compute_percentile(all_itl_samples, 90.0)
            itl_p99 = compute_percentile(all_itl_samples, 99.0)

        # Generated tokens count
        token_counts = [m.generated_tokens for m in measurement_runs if m.generated_tokens is not None]
        gen_tokens_summary = int(compute_median([float(c) for c in token_counts])) if token_counts else None

        return BenchmarkResult(
            status=ExecutionStatus.SUCCESS,
            ttft_ms=median_ttft,
            itl_ms_mean=itl_mean,
            itl_ms_p50=itl_p50,
            itl_ms_p90=itl_p90,
            itl_ms_p99=itl_p99,
            throughput_tokens_per_sec=median_throughput,
            peak_vram_used_mb=peak_memory,
            total_latency_ms=median_total_latency,
            generated_tokens_count=gen_tokens_summary,
            raw_latencies_ms=all_itl_samples if all_itl_samples else None,
            model_id=model_id,
            model_revision=model_revision,
            device_id=config.device_id,
            runtime_name=runtime_name,
            precision=config.precision,
            warmup_runs=config.warmup_runs,
            measurement_runs_requested=requested_runs,
            measurement_runs_completed=completed_runs,
            failed_runs=0,
            raw_measurements_reference=raw_measurements_reference,
        )
