"""Robust statistical calculations and independent summary recomputation for Benchmark Guard."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from rocmhub.benchmarks.base import BenchmarkRunMeasurement
from rocmhub.benchmarks.metrics import compute_percentile
from rocmhub.core.types import BenchmarkResult, ExecutionStatus


def compute_median(values: Sequence[float]) -> float:
    """Compute the true sample median of a numeric sequence.

    Args:
        values: Non-empty sequence of numbers.

    Returns:
        Median value.

    Raises:
        ValueError: If sequence is empty.
    """
    if not values:
        raise ValueError("Cannot compute median of an empty sequence")
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 1:
        return float(sorted_vals[mid])
    return float((sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0)


def compute_mad(values: Sequence[float]) -> float:
    """Compute the Median Absolute Deviation (MAD) of a numeric sequence.

    MAD = median(|x_i - median(X)|)

    Args:
        values: Non-empty sequence of numbers.

    Returns:
        Median absolute deviation.

    Raises:
        ValueError: If sequence is empty.
    """
    if not values:
        raise ValueError("Cannot compute MAD of an empty sequence")
    med = compute_median(values)
    deviations = [abs(x - med) for x in values]
    return compute_median(deviations)


def compute_relative_mad(values: Sequence[float]) -> float:
    """Compute Relative MAD (MAD / median) as a robust normalized dispersion metric.

    Args:
        values: Sequence of numbers.

    Returns:
        Normalized relative MAD, or 0.0 if empty or median <= 0.
    """
    if not values:
        return 0.0
    med = compute_median(values)
    if med <= 0:
        return 0.0
    mad = compute_mad(values)
    return float(mad / med)


def recompute_benchmark_summary(
    measurements: Sequence[BenchmarkRunMeasurement | Dict[str, Any]],
) -> Dict[str, Optional[float]]:
    """Recompute headline benchmark metrics directly from raw run measurements.

    Excludes warmup runs. Only valid measurement runs are incorporated.

    Returns:
        Dictionary mapping metric names to recomputed values:
        'median_ttft_ms', 'median_total_latency_ms', 'median_throughput_tokens_per_sec',
        'peak_vram_used_mb', 'itl_mean_ms', 'itl_p50_ms', 'itl_p90_ms', 'itl_p99_ms'.
    """
    # Normalize inputs to objects
    parsed_runs: List[BenchmarkRunMeasurement] = []
    for m in measurements:
        if isinstance(m, BenchmarkRunMeasurement):
            parsed_runs.append(m)
        else:
            parsed_runs.append(BenchmarkRunMeasurement.model_validate(m))

    # Separate warmup runs from measurement runs
    meas_runs = [r for r in parsed_runs if not r.is_warmup and r.status == ExecutionStatus.SUCCESS]
    if not meas_runs:
        return {
            "median_ttft_ms": None,
            "median_total_latency_ms": None,
            "median_throughput_tokens_per_sec": None,
            "peak_vram_used_mb": None,
            "itl_mean_ms": None,
            "itl_p50_ms": None,
            "itl_p90_ms": None,
            "itl_p99_ms": None,
        }

    # 1. TTFT
    ttft_vals = [r.ttft_ms for r in meas_runs if r.ttft_ms is not None]
    median_ttft = compute_median(ttft_vals) if ttft_vals else None

    # 2. Total Latency
    latency_vals = [r.total_latency_ms for r in meas_runs]
    median_latency = compute_median(latency_vals) if latency_vals else None

    # 3. Throughput
    throughput_vals: List[float] = []
    for r in meas_runs:
        if r.generated_tokens is not None and r.generated_tokens > 0:
            if r.first_token_at_ns is not None:
                last_token_ns = r.first_token_at_ns + int(sum(r.inter_token_latencies_ms) * 1_000_000)
                duration_s = max(1e-9, (last_token_ns - r.started_at_ns) / 1_000_000_000.0)
            else:
                duration_s = max(1e-9, (r.finished_at_ns - r.started_at_ns) / 1_000_000_000.0)
            throughput_vals.append(r.generated_tokens / duration_s)
    median_throughput = compute_median(throughput_vals) if throughput_vals else None

    # 4. Peak Memory
    mem_vals = [r.peak_vram_used_mb for r in meas_runs if r.peak_vram_used_mb is not None]
    peak_mem = max(mem_vals) if mem_vals else None

    # 5. ITL distribution
    all_itl: List[float] = []
    for r in meas_runs:
        all_itl.extend(r.inter_token_latencies_ms)

    itl_mean = (sum(all_itl) / len(all_itl)) if all_itl else None
    itl_p50 = compute_percentile(all_itl, 50.0) if all_itl else None
    itl_p90 = compute_percentile(all_itl, 90.0) if all_itl else None
    itl_p99 = compute_percentile(all_itl, 99.0) if all_itl else None

    return {
        "median_ttft_ms": median_ttft,
        "median_total_latency_ms": median_latency,
        "median_throughput_tokens_per_sec": median_throughput,
        "peak_vram_used_mb": peak_mem,
        "itl_mean_ms": itl_mean,
        "itl_p50_ms": itl_p50,
        "itl_p90_ms": itl_p90,
        "itl_p99_ms": itl_p99,
    }


def verify_summary_against_raw(
    summary: BenchmarkResult,
    raw_measurements: Sequence[BenchmarkRunMeasurement | Dict[str, Any]],
    tolerance: float = 1e-3,
) -> List[str]:
    """Compare recorded summary metrics with recomputed raw values.

    Returns a list of error strings describing any discrepancies exceeding tolerance.
    """
    recomputed = recompute_benchmark_summary(raw_measurements)
    mismatches: List[str] = []

    pairs = [
        ("ttft_ms", summary.ttft_ms, recomputed["median_ttft_ms"]),
        ("total_latency_ms", summary.total_latency_ms, recomputed["median_total_latency_ms"]),
        ("throughput_tokens_per_sec", summary.throughput_tokens_per_sec, recomputed["median_throughput_tokens_per_sec"]),
        ("peak_vram_used_mb", summary.peak_vram_used_mb, recomputed["peak_vram_used_mb"]),
        ("itl_ms_mean", summary.itl_ms_mean, recomputed["itl_mean_ms"]),
        ("itl_ms_p50", summary.itl_ms_p50, recomputed["itl_p50_ms"]),
        ("itl_ms_p90", summary.itl_ms_p90, recomputed["itl_p90_ms"]),
        ("itl_ms_p99", summary.itl_ms_p99, recomputed["itl_p99_ms"]),
    ]

    for name, recorded, recalc in pairs:
        if recorded is None and recalc is None:
            continue
        if recorded is None or recalc is None:
            mismatches.append(f"Metric '{name}' presence mismatch: recorded={recorded}, recomputed={recalc}")
            continue

        denom = max(abs(recorded), 1e-9)
        rel_diff = abs(recorded - recalc) / denom
        if rel_diff > tolerance:
            mismatches.append(
                f"Metric '{name}' diverged: recorded={recorded}, recomputed={recalc} (rel_diff={rel_diff:.4f} > {tolerance})"
            )

    return mismatches
