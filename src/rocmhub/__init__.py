"""ROCmHub: Automated AI model preparation, optimization, and benchmarking for AMD GPUs and ROCm."""

__version__ = "0.1.0"

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
    SystemCapabilities,
)

__all__ = [
    "__version__",
    "ROCmHubError",
    "CapabilityEvaluator",
    "CapabilityPolicy",
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
    "ExperimentSpec",
    "BenchmarkResult",
    "ArtifactManifest",
]
