"""Domain-specific exceptions for ROCmHub."""

from __future__ import annotations

from typing import Any, Dict, Optional


class ROCmHubError(Exception):
    """Base exception for all ROCmHub errors."""

    error_code: str = "GENERIC_ERROR"

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"[{self.error_code}] {self.message} (details: {self.details})"
        return f"[{self.error_code}] {self.message}"


class ConfigurationError(ROCmHubError):
    """Raised when configuration or runtime arguments are invalid."""

    error_code = "CONFIGURATION_ERROR"


class SchemaValidationError(ROCmHubError):
    """Raised when data model invariants or schema validations fail."""

    error_code = "SCHEMA_VALIDATION_ERROR"


class HardwareDetectionError(ROCmHubError):
    """Raised when hardware probing encounters an unrecoverable failure."""

    error_code = "HARDWARE_DETECTION_ERROR"


class ModelResolutionError(ROCmHubError):
    """Base exception for errors during model resolution."""

    error_code = "MODEL_RESOLUTION_ERROR"


class ModelNotFoundError(ModelResolutionError):
    """Raised when the requested model repository does not exist."""

    error_code = "MODEL_NOT_FOUND"


class AuthRequiredError(ModelResolutionError):
    """Raised when access to a gated or private model repository requires authentication."""

    error_code = "AUTH_REQUIRED"


class RevisionNotFoundError(ModelResolutionError):
    """Raised when the specified revision (branch, tag, or commit SHA) is not found."""

    error_code = "REVISION_NOT_FOUND"


class NetworkError(ROCmHubError):
    """Raised when a network connectivity or timeout failure occurs."""

    error_code = "NETWORK_ERROR"


class ModelInspectionError(ROCmHubError):
    """Base exception for static inspection failures."""

    error_code = "MODEL_INSPECTION_ERROR"


class InvalidModelMetadataError(ModelInspectionError):
    """Raised when model metadata or configuration files are corrupt or unparseable."""

    error_code = "INVALID_MODEL_METADATA"


class RemoteCodeRequiredError(ModelInspectionError):
    """Raised when a model requires custom remote code execution for inspection."""

    error_code = "REMOTE_CODE_REQUIRED"


class RunnerExecutionError(ROCmHubError):
    """Base exception for inference runner failures during setup, warmup, or generation."""

    error_code = "RUNNER_EXECUTION_ERROR"


class RunnerNotReadyError(RunnerExecutionError):
    """Raised when generate() or an execution method is called before load()."""

    error_code = "RUNNER_NOT_READY"


class ModelLoadError(RunnerExecutionError):
    """Raised when model weights or tokenizer fail to load onto the target device."""

    error_code = "MODEL_LOAD_FAILED"


class DeviceNotAvailableError(RunnerExecutionError):
    """Raised when the requested accelerator device is not available or invalid."""

    error_code = "DEVICE_NOT_AVAILABLE"


class UnsupportedPrecisionError(RunnerExecutionError):
    """Raised when the requested precision is unsupported or invalid."""

    error_code = "UNSUPPORTED_PRECISION"


class UnsupportedModelTypeError(RunnerExecutionError):
    """Raised when the model architecture does not match the runner's supported baseline class."""

    error_code = "UNSUPPORTED_MODEL_TYPE"


class GenerationError(RunnerExecutionError):
    """Raised when token generation or decoding encounters a runtime failure."""

    error_code = "GENERATION_FAILED"


class BenchmarkError(ROCmHubError):
    """Raised when benchmark harness fails to collect or compute metrics."""

    error_code = "BENCHMARK_ERROR"


class InvalidBenchmarkConfigError(BenchmarkError):
    """Raised when benchmark configuration parameters violate validation bounds."""

    error_code = "INVALID_BENCHMARK_CONFIG"


class BenchmarkHarnessError(BenchmarkError):
    """Raised when the benchmark harness encounters an unrecoverable execution failure."""

    error_code = "BENCHMARK_HARNESS_FAILED"


class ArtifactPackagingError(ROCmHubError):
    """Raised when artifact manifest, packaging, or checksum validation fails."""

    error_code = "ARTIFACT_PACKAGING_ERROR"


class NotImplementedFeatureError(ROCmHubError):
    """Raised when a feature planned for a future phase is requested."""

    error_code = "NOT_IMPLEMENTED"
