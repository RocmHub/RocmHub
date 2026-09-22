"""Terminal formatting for Optimization Engine reports (Phase 12)."""

from __future__ import annotations

from rocmhub.optimization.base import OptimizationReport


def format_optimization_report_table(report: OptimizationReport) -> str:
    """Format an OptimizationReport into a clean, human-readable terminal table."""
    lines = [
        "ROCmHub Optimization Engine Report",
        "-" * 80,
        f"Session ID:                {report.session_id}",
        f"Model:                     {report.model_id} (revision: {report.revision[:12]}...)",
        f"Target GPU:                {report.target_gpu or 'Auto-detected / None'}",
        f"Objective:                 {report.objective}",
        f"Final Status:              {report.status}",
        f"Elapsed Time:              {report.total_duration_seconds:.2f}s",
        f"Best Candidate:            {report.best_candidate_id or 'None / Inconclusive'}",
        "",
        "Baseline Reference:",
        f"  Precision:               {report.baseline.precision}",
        f"  Runtime:                 {report.baseline.runtime}",
        f"  Build Status:            {report.baseline.status.value}",
        f"  Measured on GPU:         {'yes' if report.baseline.measured else 'no (NOT_MEASURED on non-AMD)'}",
    ]

    b_res = report.baseline.benchmark_result
    if b_res and b_res.ttft_ms is not None:
        lines.append(f"  TTFT:                    {b_res.ttft_ms:.2f} ms")
    if b_res and b_res.throughput_tokens_per_sec is not None:
        lines.append(f"  Throughput:              {b_res.throughput_tokens_per_sec:.2f} tok/s")

    lines.extend([
        "",
        "Optimization Candidates:",
        f"  {'Candidate ID':<20} {'Strategy':<16} {'Precision':<10} {'Status':<14} {'Build Dir'}",
        f"  {'-'*18:<20} {'-'*14:<16} {'-'*8:<10} {'-'*12:<14} {'-'*20}",
    ])

    for cand in report.candidates:
        bdir = cand.build_dir or "None"
        if len(bdir) > 30:
            bdir = "..." + bdir[-27:]
        lines.append(
            f"  {cand.candidate_id:<20} {cand.strategy.value:<16} {cand.precision:<10} {cand.status.value:<14} {bdir}"
        )

    lines.extend([
        "",
        "Comparative Evaluation:",
        f"  {'Candidate ID':<20} {'Strategy':<14} {'Verdict':<14} {'TTFT Spd':<10} {'Thr Spd':<10} {'QRR':<8}",
        f"  {'-'*18:<20} {'-'*12:<14} {'-'*12:<14} {'-'*8:<10} {'-'*8:<10} {'-'*6:<8}",
    ])

    for comp in report.comparisons:
        ttft_str = f"{comp.ttft_speedup:.2f}x" if comp.ttft_speedup is not None else "N/A"
        thr_str = f"{comp.throughput_speedup:.2f}x" if comp.throughput_speedup is not None else "N/A"
        qrr_str = f"{comp.qrr_percent:.1f}%" if comp.qrr_percent is not None else "N/A"
        lines.append(
            f"  {comp.candidate_id:<20} {comp.strategy.value:<14} {comp.verdict.value:<14} {ttft_str:<10} {thr_str:<10} {qrr_str:<8}"
        )

    if report.recommendations:
        lines.extend(["", "Recommendations:"])
        for rec in report.recommendations:
            lines.append(f"  - {rec}")

    if report.errors:
        lines.extend(["", "Errors Encountered:"])
        for err in report.errors:
            lines.append(f"  [ERROR] {err}")

    return "\n".join(lines)
