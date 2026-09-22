"""Memory and trajectory persistence for Autonomous AI Engineer (Phase 11)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from rocmhub.engineer.base import EngineerReport, TrajectoryStep
from rocmhub.forge.manifest import assert_no_secrets, sanitize_secrets_in_obj


class TrajectoryStore:
    """Manages audit trajectories and final reports in local JSONL and JSON storage."""

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        self._base_dir = (base_dir or Path(".rocmhub")).resolve()
        self._trajectories_dir = self._base_dir / "trajectories"
        self._reports_dir = self._base_dir / "reports"

        self._trajectories_dir.mkdir(parents=True, exist_ok=True)
        self._reports_dir.mkdir(parents=True, exist_ok=True)

    def append_step(self, session_id: str, step: TrajectoryStep) -> Path:
        """Append a step to the session's JSONL trajectory log."""
        log_file = self._trajectories_dir / f"{session_id}.jsonl"
        sanitized_dict = sanitize_secrets_in_obj(step.model_dump())
        line = json.dumps(sanitized_dict, sort_keys=True)
        assert_no_secrets(line)

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")

        return log_file

    def save_report(self, report: EngineerReport) -> Path:
        """Write finalized EngineerReport to JSON file."""
        report_file = self._reports_dir / f"{report.session_id}.json"
        sanitized_dict = sanitize_secrets_in_obj(report.model_dump())
        content = json.dumps(sanitized_dict, indent=2, sort_keys=True)
        assert_no_secrets(content)

        with open(report_file, "w", encoding="utf-8") as f:
            f.write(content + "\n")

        return report_file

    def load_trajectory(self, session_id: str) -> List[TrajectoryStep]:
        """Read all steps from a session trajectory JSONL log."""
        log_file = self._trajectories_dir / f"{session_id}.jsonl"
        if not log_file.exists():
            return []

        steps: List[TrajectoryStep] = []
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    steps.append(TrajectoryStep.model_validate(data))
        return steps

    def load_report(self, session_id: str) -> Optional[EngineerReport]:
        """Load saved EngineerReport if it exists."""
        report_file = self._reports_dir / f"{session_id}.json"
        if not report_file.exists():
            return None
        with open(report_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return EngineerReport.model_validate(data)
