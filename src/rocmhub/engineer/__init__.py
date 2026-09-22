"""Autonomous AI Engineer subsystem for ROCmHub (Phase 11)."""

from rocmhub.engineer.agent import AIEngineer
from rocmhub.engineer.base import (
    CURRENT_ENGINEER_SCHEMA_VERSION,
    EngineerBudget,
    EngineerObjective,
    EngineerReport,
    EngineerRequest,
    EngineerStatus,
    TrajectoryStep,
)
from rocmhub.engineer.memory import TrajectoryStore
from rocmhub.engineer.policy import (
    BudgetGuard,
    FailureClassifier,
    FailureSeverity,
    LoopDetector,
)
from rocmhub.engineer.provider import (
    AgentAction,
    AutonomousRulesProvider,
    LLMProvider,
    OpenAICompatibleProvider,
)
from rocmhub.engineer.reports import format_engineer_report_table
from rocmhub.engineer.tools import (
    ALLOWED_TOOLS,
    ToolRegistry,
    sanitize_untrusted_text,
    validate_safe_path,
)

__all__ = [
    "CURRENT_ENGINEER_SCHEMA_VERSION",
    "AIEngineer",
    "EngineerObjective",
    "EngineerStatus",
    "EngineerBudget",
    "EngineerRequest",
    "EngineerReport",
    "TrajectoryStep",
    "AgentAction",
    "LLMProvider",
    "AutonomousRulesProvider",
    "OpenAICompatibleProvider",
    "ToolRegistry",
    "ALLOWED_TOOLS",
    "validate_safe_path",
    "sanitize_untrusted_text",
    "BudgetGuard",
    "LoopDetector",
    "FailureClassifier",
    "FailureSeverity",
    "TrajectoryStore",
    "format_engineer_report_table",
]
