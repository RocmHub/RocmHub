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


class ValidationError(ROCmHubError):
    """Base exception for model correctness and quality validation failures."""

    error_code = "VALIDATION_ERROR"


class InvalidValidationConfigError(ValidationError):
    """Raised when validation suite configuration parameters violate validation bounds."""

    error_code = "INVALID_VALIDATION_CONFIG"


class CorrectnessGateFailedError(ValidationError):
    """Raised when hard correctness verification gates fail."""

    error_code = "CORRECTNESS_GATE_FAILED"


class QualityGateFailedError(ValidationError):
    """Raised when comparative quality regression thresholds are violated."""

    error_code = "QUALITY_GATE_FAILED"


class ArtifactError(ROCmHubError):
    """Base exception for artifact builder, store, and integrity failures."""

    error_code = "ARTIFACT_ERROR"


class ArtifactPackagingError(ArtifactError):
    """Raised when artifact manifest, packaging, or checksum validation fails."""

    error_code = "ARTIFACT_PACKAGING_ERROR"


class ArtifactConflictError(ArtifactError):
    """Raised when an artifact already exists and contents differ (preventing silent overwrite)."""

    error_code = "ARTIFACT_CONFLICT"


class ArtifactIntegrityError(ArtifactError):
    """Raised when an artifact bundle fails checksum or completeness verification."""

    error_code = "ARTIFACT_INTEGRITY_FAILED"


class SecretDetectedError(ArtifactError):
    """Raised when a secret, credential, or sensitive token is detected in artifact payload."""

    error_code = "SECRET_DETECTED"


class GuardError(ROCmHubError):
    """Base exception for Benchmark Guard and reproducibility verification failures."""

    error_code = "GUARD_ERROR"


class EnvironmentDriftError(GuardError):
    """Raised when the observed runtime environment diverges from the expected environment fingerprint."""

    error_code = "ENVIRONMENT_DRIFT"


class EvidenceInconsistentError(GuardError):
    """Raised when raw measurements contradict summary results or violate metric invariants."""

    error_code = "EVIDENCE_INCONSISTENT"


class SummaryMismatchError(GuardError):
    """Raised when recomputed summary metrics diverge from recorded values in BenchmarkResult."""

    error_code = "SUMMARY_MISMATCH"


class HardwareHealthError(GuardError):
    """Raised when hardware health telemetry indicates throttling or uncorrectable hardware degradation."""

    error_code = "HARDWARE_HEALTH_ERROR"


class NotImplementedFeatureError(ROCmHubError):
    """Raised when a feature planned for a future phase is requested."""

    error_code = "NOT_IMPLEMENTED"


class ForgeError(ROCmHubError):
    """Base exception for all Model Forge operations."""

    error_code = "FORGE_ERROR"


class UnsupportedModelArchitectureError(ForgeError):
    """Raised when a model architecture is unsupported by the selected recipe."""

    error_code = "UNSUPPORTED_MODEL_ARCHITECTURE"


class InsufficientDiskSpaceError(ForgeError):
    """Raised when required disk space exceeds available free disk space."""

    error_code = "INSUFFICIENT_DISK_SPACE"


class ModelMaterializationError(ForgeError):
    """Raised when materializing (downloading or linking) model weights/config fails."""

    error_code = "MODEL_MATERIALIZATION_FAILED"


class BuildExecutionError(ForgeError):
    """Raised when executing a recipe build step fails."""

    error_code = "BUILD_EXECUTION_FAILED"


class BuildConflictError(ForgeError):
    """Raised when a build output directory already exists and conflict handling fails."""

    error_code = "BUILD_CONFLICT"


class EngineerError(ROCmHubError):
    """Base exception for all Autonomous AI Engineer operations."""

    error_code = "ENGINEER_ERROR"


class BudgetExceededError(EngineerError):
    """Raised when engineer budget (time, disk, or attempts) is exhausted."""

    error_code = "BUDGET_EXCEEDED"


class RepeatedFailureError(EngineerError):
    """Raised when repeated failures in the same state are detected."""

    error_code = "REPEATED_FAILURE_LOOP"


class SecurityBoundaryError(EngineerError):
    """Raised when an action violates security sandboxing or path safety rules."""

    error_code = "SECURITY_BOUNDARY_VIOLATION"


class LLMProviderError(EngineerError):
    """Raised when the LLM provider fails, times out, or returns malformed output."""

    error_code = "LLM_PROVIDER_ERROR"


class InvalidToolCallError(EngineerError):
    """Raised when an agent attempts to invoke an unknown or improperly parameterized tool."""

    error_code = "INVALID_TOOL_CALL"

