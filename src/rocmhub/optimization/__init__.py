"""Optimization Engine for ROCmHub (Phase 12)."""

from rocmhub.optimization.base import (
    CandidateStatus,
    ComparisonResult,
    ComparisonVerdict,
    OptimizationBaseline,
    OptimizationCandidate,
    OptimizationPlan,
    OptimizationReport,
    OptimizationRequest,
    OptimizationStrategy,
)
from rocmhub.optimization.baseline import BaselineManager
from rocmhub.optimization.comparison import ComparisonEngine
from rocmhub.optimization.executor import OptimizationExecutor
from rocmhub.optimization.recipes import (
    BF16OptimizationRecipe,
    FP16OptimizationRecipe,
    FP32OptimizationRecipe,
    OptimizationRecipe,
    QuantizationRecipe,
    TorchCompileRecipe,
    get_recipe_for_strategy,
)
from rocmhub.optimization.reports import format_optimization_report_table

__all__ = [
    "CandidateStatus",
    "ComparisonResult",
    "ComparisonVerdict",
    "OptimizationBaseline",
    "OptimizationCandidate",
    "OptimizationPlan",
    "OptimizationReport",
    "OptimizationRequest",
    "OptimizationStrategy",
    "BaselineManager",
    "ComparisonEngine",
    "OptimizationExecutor",
    "OptimizationRecipe",
    "BF16OptimizationRecipe",
    "FP16OptimizationRecipe",
    "FP32OptimizationRecipe",
    "TorchCompileRecipe",
    "QuantizationRecipe",
    "get_recipe_for_strategy",
    "format_optimization_report_table",
]
