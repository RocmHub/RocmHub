"""ROCmHub: Automated AI model preparation, optimization, and benchmarking for AMD GPUs and ROCm."""

__version__ = "0.1.0"

from rocmhub.benchmarks import (
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkRunMeasurement,
    MetricsCalculator,
)
from rocmhub.capabilities import CapabilityEvaluator, CapabilityPolicy
from rocmhub.core.errors import ROCmHubError
from rocmhub.core.types import (
    ArtifactManifest,
    BenchmarkResult,
    CapabilityReport,
    DetectionReport,
    DeviceCapabilityAssessment,
    EnvironmentSpec,
    EvaluationReason,
    EvaluationSeverity,
    EvaluationVerdict,
    ExecutionStatus,
    ExperimentSpec,
    HardwareSpec,
    ModelSpec,
    RunResult,
    SystemCapabilities,
    ValidationCase,
    ValidationMode,
    ValidationReport,
    ValidationRunResult,
    ValidationVerdict,
)
from rocmhub.runners import BaseRunner, HuggingFaceRunner
from rocmhub.validation import ValidationConfig, ValidationEvaluator

__all__ = [
    "__version__",
    "ROCmHubError",
    "CapabilityEvaluator",
    "CapabilityPolicy",
    "BaseRunner",
    "HuggingFaceRunner",
    "BenchmarkConfig",
    "BenchmarkRunMeasurement",
    "BenchmarkHarness",
    "MetricsCalculator",
    "ExecutionStatus",
    "ModelSpec",
    "HardwareSpec",
    "EnvironmentSpec",
    "DetectionReport",
    "CapabilityReport",
    "DeviceCapabilityAssessment",
    "EvaluationReason",
    "EvaluationSeverity",
    "EvaluationVerdict",
    "SystemCapabilities",
    "RunResult",
    "ExperimentSpec",
    "BenchmarkResult",
    "ValidationConfig",
    "ValidationEvaluator",
    "ValidationReport",
    "ValidationVerdict",
    "ValidationMode",
    "ValidationCase",
    "ValidationRunResult",
    "ArtifactManifest",
]
