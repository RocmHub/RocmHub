"""Report formatting utilities for Autonomous AI Engineer (Phase 11)."""

from __future__ import annotations

from typing import List

from rocmhub.engineer.base import EngineerReport


def format_engineer_report_table(report: EngineerReport) -> str:
    """Format EngineerReport into a clean human-readable terminal summary."""
    lines: List[str] = [
        "ROCmHub Autonomous AI Engineer Report",
        "=" * 80,
        f"{'Session ID:':<26} {report.session_id}",
        f"{'Model:':<26} {report.model_id}",
        f"{'Revision:':<26} {report.revision}",
        f"{'Target GPU:':<26} {report.target_gpu or 'Auto-detected / None'}",
        f"{'Objective:':<26} {report.objective.value}",
        f"{'Status:':<26} {report.status.value}",
        f"{'Total Duration:':<26} {report.total_duration_seconds:.2f}s",
        f"{'Attempts Used:':<26} {report.attempts_used}",
        "",
        "Build Summary:",
    ]

    if report.build_manifest:
        bm = report.build_manifest
        lines.append(f"  {'Build ID:':<24} {bm.build_id}")
        lines.append(f"  {'Build Status:':<24} {bm.status.value}")
        lines.append(f"  {'AMD Validated:':<24} {'yes' if bm.amd_validated else 'no'}")
        lines.append(f"  {'Build Directory:':<24} {bm.build_dir}")
        lines.append(f"  {'Runtime:':<24} {bm.runtime}")
    else:
        lines.append("  (No build manifest generated)")

    lines.append("")
    lines.append("Execution Trajectory:")
    for step in report.trajectory:
        obs_short = step.observation or step.tool_result.get("error") or ""
        lines.append(f"  [{step.step_index:02d}] {step.phase:<9} {step.action:<22} ({step.duration_seconds:.3f}s) {obs_short}")

    if report.reasons:
        lines.append("")
        lines.append("Outcome Explanation:")
        for r in report.reasons:
            lines.append(f"  - {r}")

    if report.errors_encountered:
        lines.append("")
        lines.append("Errors Encountered:")
        for err in report.errors_encountered:
            lines.append(f"  - {err}")

    return "\n".join(lines)
