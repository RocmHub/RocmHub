"""Core domain models and exceptions for ROCmHub."""

from rocmhub.core.errors import (
    ArtifactPackagingError,
    BenchmarkError,
    ConfigurationError,
    HardwareDetectionError,
    ModelInspectionError,
    ModelResolutionError,
    NotImplementedFeatureError,
    ROCmHubError,
    RunnerExecutionError,
    SchemaValidationError,
)
from rocmhub.core.types import (
    ArtifactManifest,
    BenchmarkResult,
    EnvironmentSpec,
    ExecutionStatus,
    ExperimentSpec,
    HardwareSpec,
    ModelSpec,
)

__all__ = [
    "ROCmHubError",
    "ConfigurationError",
    "SchemaValidationError",
    "HardwareDetectionError",
    "ModelResolutionError",
    "ModelInspectionError",
    "RunnerExecutionError",
    "BenchmarkError",
    "ArtifactPackagingError",
    "NotImplementedFeatureError",
    "ExecutionStatus",
    "ModelSpec",
    "HardwareSpec",
    "EnvironmentSpec",
    "ExperimentSpec",
    "BenchmarkResult",
    "ArtifactManifest",
]
