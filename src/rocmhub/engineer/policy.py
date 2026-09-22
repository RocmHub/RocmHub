"""Budget enforcement, loop detection, and policy rules for Autonomous AI Engineer."""

from __future__ import annotations

import os
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict

from rocmhub.core.errors import (
    AuthRequiredError,
    BudgetExceededError,
    ModelNotFoundError,
    RepeatedFailureError,
    SecurityBoundaryError,
    UnsupportedModelArchitectureError,
)
from rocmhub.engineer.base import EngineerBudget


class FailureSeverity(str, Enum):
    """Classification of errors encountered during agent execution."""

    RECOVERABLE = "RECOVERABLE"
    FATAL = "FATAL"


class LoopDetector:
    """Detects consecutive repeated failures or cyclical action sequences."""

    def __init__(self, max_consecutive_identical_failures: int = 2) -> None:
        self._max_consecutive = max_consecutive_identical_failures
        self._action_failure_counts: Dict[str, int] = {}

    def record_action(self, tool_name: str, tool_args: Dict[str, Any], is_success: bool) -> None:
        """Record an action and verify it doesn't trigger repeated failure loop."""
        import hashlib
        import json

        args_hash = hashlib.sha256(json.dumps(tool_args, sort_keys=True, default=str).encode("utf-8")).hexdigest()
        key = f"{tool_name}:{args_hash}"

        if is_success:
            self._action_failure_counts[key] = 0
        else:
            current_failures = self._action_failure_counts.get(key, 0) + 1
            self._action_failure_counts[key] = current_failures
            if current_failures >= self._max_consecutive:
                raise RepeatedFailureError(
                    f"Action '{tool_name}' failed consecutively {current_failures} times with identical arguments. "
                    "Aborting to prevent infinite loop.",
                    details={"tool_name": tool_name, "consecutive_failures": current_failures},
                )


class BudgetGuard:
    """Enforces execution limits on attempts, wall-clock time, and disk usage."""

    def __init__(self, budget: EngineerBudget) -> None:
        self._budget = budget
        self._start_time = time.monotonic()
        self._attempts_count = 0

    def tick_attempt(self) -> int:
        """Increment attempt counter and verify within budget."""
        self._attempts_count += 1
        if self._attempts_count > self._budget.max_attempts:
            raise BudgetExceededError(
                f"Maximum allowed attempts ({self._budget.max_attempts}) exceeded.",
                details={"attempts_count": self._attempts_count, "max_attempts": self._budget.max_attempts},
            )
        return self._attempts_count

    def check_time(self) -> float:
        """Check elapsed time against budget."""
        elapsed = time.monotonic() - self._start_time
        if elapsed > self._budget.max_execution_time_seconds:
            raise BudgetExceededError(
                f"Maximum execution time ({self._budget.max_execution_time_seconds:.1f}s) exceeded. "
                f"Elapsed: {elapsed:.1f}s.",
                details={"elapsed_seconds": elapsed, "max_seconds": self._budget.max_execution_time_seconds},
            )
        return elapsed

    def check_disk(self, check_path: Path) -> int:
        """Verify disk usage of output path is within budget."""
        if not check_path.exists():
            return 0
        total_size = 0
        if check_path.is_file():
            total_size = check_path.stat().st_size
        else:
            for root, _, files in os.walk(check_path):
                for f in files:
                    try:
                        total_size += (Path(root) / f).stat().st_size
                    except OSError:
                        pass

        if total_size > self._budget.max_disk_usage_bytes:
            raise BudgetExceededError(
                f"Disk usage for '{check_path}' ({total_size / (1024**3):.2f} GB) exceeded budget "
                f"({self._budget.max_disk_usage_bytes / (1024**3):.2f} GB).",
                details={
                    "total_size_bytes": total_size,
                    "max_disk_bytes": self._budget.max_disk_usage_bytes,
                },
            )
        return total_size


class FailureClassifier:
    """Categorizes exceptions to determine if the agent should retry, revise, or abort."""

    @staticmethod
    def classify(exc: Exception) -> FailureSeverity:
        """Determine severity of an exception."""
        fatal_types = (
            AuthRequiredError,
            ModelNotFoundError,
            UnsupportedModelArchitectureError,
            SecurityBoundaryError,
            BudgetExceededError,
            RepeatedFailureError,
        )
        if isinstance(exc, fatal_types):
            return FailureSeverity.FATAL
        return FailureSeverity.RECOVERABLE
