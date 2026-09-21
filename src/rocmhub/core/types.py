"""Core data models and schemas for ROCmHub."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CURRENT_SCHEMA_VERSION = "1.0.0"


def _utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


class ExecutionStatus(str, Enum):
    """Execution status of an experiment or benchmark."""

    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    NOT_MEASURED = "NOT_MEASURED"


class ValidationVerdict(str, Enum):
    """Top-level verdict of a model correctness and quality validation run."""

    PASS = "PASS"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"
    NOT_MEASURED = "NOT_MEASURED"


class ValidationMode(str, Enum):
    """Execution mode of the validation suite."""

    SELF_VALIDATION = "SELF_VALIDATION"
    COMPARISON = "COMPARISON"


class ArtifactStatus(str, Enum):
    """Integrity and completeness status of a materialized artifact bundle."""

    COMPLETE = "COMPLETE"
    INCOMPLETE = "INCOMPLETE"
    INVALID = "INVALID"


class GuardVerdict(str, Enum):
    """Verdict of the Benchmark Guard determining evidence validity and reproducibility."""

    PASS = "PASS"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"
    NOT_MEASURED = "NOT_MEASURED"


class ExperimentRole(str, Enum):
    """Role of an experiment in a comparative evaluation."""

    BASELINE = "BASELINE"
    CANDIDATE = "CANDIDATE"
    REFERENCE = "REFERENCE"


class ModelSpec(BaseModel):
    """Specification of the AI model under test."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(default=CURRENT_SCHEMA_VERSION, description="Schema version of ModelSpec")
    model_id: str = Field(..., description="Model identifier, e.g. 'Qwen/Qwen2.5-0.5B-Instruct'")
    source: str = Field(default="huggingface", description="Model repository source")
    requested_revision: str = Field(default="main", description="User-requested revision, branch, or tag")
    commit_sha: str = Field(
        ...,
        description="Immutable resolved Git commit SHA (40-char hex string) representing exact model snapshot",
    )
    architecture: Optional[str] = Field(default=None, description="Model architecture class, e.g. 'Qwen2ForCausalLM'")
    parameter_count: Optional[int] = Field(default=None, description="Total parameter count")
    context_length: Optional[int] = Field(default=None, description="Maximum supported context length in tokens")
    default_dtype: Optional[str] = Field(default=None, description="Default model weight precision, e.g. 'bfloat16'")
    weights_format: Optional[str] = Field(default=None, description="Format of weights, e.g. 'safetensors'")
    remote_code_required: bool = Field(
        default=False,
        description="True if model architecture requires executing custom remote repository code",
    )

    @field_validator("commit_sha")
    @classmethod
    def validate_commit_sha(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("commit_sha must not be empty")
        # If it is a full 40-character SHA, verify hex characters
        if len(v) == 40 and not re.fullmatch(r"[0-9a-fA-F]{40}", v):
            raise ValueError(f"commit_sha of length 40 must contain only hexadecimal characters, got '{v}'")
        return v.lower()

    @field_validator("parameter_count")
    @classmethod
    def validate_parameter_count(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v < 0:
            raise ValueError("parameter_count cannot be negative")
        return v

    @field_validator("context_length")
    @classmethod
    def validate_context_length(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v <= 0:
            raise ValueError("context_length must be greater than zero")
        return v


class HardwareSpec(BaseModel):
    """Hardware specification of the host execution environment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(default=CURRENT_SCHEMA_VERSION, description="Schema version of HardwareSpec")
    gpu_present: bool = Field(default=False, description="True if a discrete/integrated GPU is detected")
    gpu_vendor: Optional[str] = Field(default=None, description="GPU vendor, e.g. 'AMD', 'NVIDIA', or None")
    device_id: Optional[int] = Field(default=None, description="Device index / logical ID, e.g. 0")
    device_name: Optional[str] = Field(
        default=None,
        description="Product name, e.g. 'AMD Radeon RX 7900 XTX' or 'AMD Instinct MI300X'",
    )
    family: Optional[str] = Field(
        default=None,
        description="Architectural family, e.g. 'Radeon', 'Instinct', or None",
    )
    gfx_target: Optional[str] = Field(
        default=None,
        description="Target GPU instruction architecture as open string, e.g. 'gfx1100', 'gfx942', 'unknown'",
    )
    vram_total_mb: Optional[int] = Field(default=None, description="Total VRAM in megabytes")
    compute_units: Optional[int] = Field(default=None, description="Number of compute units (CU), if reported")
    bus_id: Optional[str] = Field(default=None, description="PCI bus ID, e.g. '0000:03:00.0'")

    @field_validator("vram_total_mb", "compute_units")
    @classmethod
    def validate_non_negative(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v < 0:
            raise ValueError("Hardware memory and compute units cannot be negative")
        return v


class EnvironmentSpec(BaseModel):
    """Software environment and driver specification."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(default=CURRENT_SCHEMA_VERSION, description="Schema version of EnvironmentSpec")
    os: str = Field(..., description="Operating system release, e.g. 'Linux 6.8.0-40-generic' or 'Darwin 24.0.0'")
    kernel: Optional[str] = Field(default=None, description="Operating system kernel release, e.g. '6.8.0-40-generic'")
    architecture: Optional[str] = Field(default=None, description="Host CPU architecture, e.g. 'x86_64' or 'arm64'")
    python_version: str = Field(..., description="Python interpreter version, e.g. '3.11.9'")
    rocm_version: Optional[str] = Field(default=None, description="Installed ROCm version, e.g. '6.2.0', or None")
    hip_version: Optional[str] = Field(default=None, description="Installed HIP runtime version, or None")
    torch_version: str = Field(..., description="PyTorch version string, e.g. '2.4.0+rocm6.2', or 'not_installed'")
    torch_hip_available: bool = Field(
        default=False,
        description="True if PyTorch has functional ROCm/HIP backend (torch.cuda.is_available() on ROCm PyTorch)",
    )
    env_vars: Dict[str, str] = Field(
        default_factory=dict,
        description="Recorded environment variables (e.g. HSA_OVERRIDE_GFX_VERSION, HIP_VISIBLE_DEVICES)",
    )


class DetectionReport(BaseModel):
    """Observation report from system hardware and environment discovery."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(default=CURRENT_SCHEMA_VERSION, description="Schema version of DetectionReport")
    environment: EnvironmentSpec = Field(..., description="Observed software environment")
    gpus: List[HardwareSpec] = Field(default_factory=list, description="List of detected GPUs (0 or more)")
    provenance: Dict[str, str] = Field(
        default_factory=dict,
        description="Source provenance for discovered fields, e.g. {'gfx_target[0]': 'rocminfo'}",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Non-fatal diagnostic warnings collected during hardware observation",
    )


class EvaluationVerdict(str, Enum):
    """Overall or per-device verdict on whether baseline execution can be attempted."""

    READY = "READY"
    BLOCKED = "BLOCKED"
    UNKNOWN = "UNKNOWN"
    NO_ACCELERATOR = "NO_ACCELERATOR"


class EvaluationSeverity(str, Enum):
    """Severity classification for an evaluation reason."""

    OK = "ok"
    INFO = "info"
    WARNING = "warning"
    BLOCKER = "blocker"


class EvaluationReason(BaseModel):
    """Structured rationale explaining a capability finding."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str = Field(..., description="Machine-readable reason code, e.g. 'NO_AMD_GPU', 'ROCM_DETECTED'")
    severity: EvaluationSeverity = Field(..., description="Severity: ok, info, warning, blocker")
    message: str = Field(..., description="Human-readable explanation")
    evidence: Dict[str, Any] = Field(default_factory=dict, description="Concrete factual evidence supporting reason")


class DeviceCapabilityAssessment(BaseModel):
    """Capability evaluation outcome for a specific GPU device."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    device_id: Optional[int] = Field(default=None, description="Logical GPU device index")
    device_name: Optional[str] = Field(default=None, description="Product name of the device")
    gfx_target: Optional[str] = Field(default=None, description="Discovered target instruction architecture")
    verdict: EvaluationVerdict = Field(..., description="Verdict for this device: READY, BLOCKED, UNKNOWN, NO_ACCELERATOR")
    reasons: List[EvaluationReason] = Field(default_factory=list, description="Reasons for this device's verdict")
    warnings: List[str] = Field(default_factory=list, description="Device-specific non-fatal warnings")


class SystemCapabilities(BaseModel):
    """Conservative capability evaluation flags for baseline assessment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    amd_gpu_present: bool = Field(default=False, description="At least one AMD GPU is detected")
    rocm_detected: bool = Field(default=False, description="ROCm runtime version detected")
    hip_detected: bool = Field(default=False, description="HIP runtime version detected")
    torch_available: bool = Field(default=False, description="PyTorch is importable in the current environment")
    torch_hip_available: bool = Field(default=False, description="PyTorch has functional HIP backend support")
    model_metadata_complete: bool = Field(default=False, description="Required model metadata fields are present")
    remote_code_required: bool = Field(default=False, description="Model requires executing custom remote code")
    baseline_runtime_candidate: Optional[str] = Field(
        default=None,
        description="Candidate runtime adapter for baseline execution attempt, e.g. 'pytorch_transformers_hip'",
    )


class CapabilityReport(BaseModel):
    """Preflight capability report linking ModelSpec and DetectionReport."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(default=CURRENT_SCHEMA_VERSION, description="Schema version of CapabilityReport")
    evaluated_at_utc: str = Field(default_factory=_utc_now_iso, description="ISO-8601 UTC timestamp of evaluation")
    model: ModelSpec = Field(..., description="Model specification evaluated")
    environment: EnvironmentSpec = Field(..., description="Observed software environment")
    hardware: List[HardwareSpec] = Field(default_factory=list, description="Observed GPU hardware devices")
    device_assessments: List[DeviceCapabilityAssessment] = Field(
        default_factory=list,
        description="Per-device capability evaluation outcomes",
    )
    verdict: EvaluationVerdict = Field(..., description="Top-level system verdict")
    reasons: List[EvaluationReason] = Field(default_factory=list, description="Top-level evaluation reasons")
    warnings: List[str] = Field(default_factory=list, description="Non-fatal warnings and advisories")
    capabilities: SystemCapabilities = Field(..., description="Evaluated baseline capabilities")


class RunResult(BaseModel):
    """Execution outcome of a model inference run.

    Distinction:
    - RunResult answers: 'Did the model actually execute inference?'
    - BenchmarkResult answers: 'How fast did it execute?'
    - Benchmark metrics (TTFT, ITL, VRAM) are NOT part of RunResult.
    - Real SUCCESS status is permitted ONLY when weights were loaded, model was placed
      on target device, forward generation was executed, and tokens were returned.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(default=CURRENT_SCHEMA_VERSION, description="Schema version of RunResult")
    status: ExecutionStatus = Field(..., description="Inference execution status: SUCCESS, FAILED, SKIPPED, NOT_MEASURED")
    runtime_name: str = Field(..., description="Runtime adapter identifier, e.g. 'pytorch_transformers_hip'")
    model_id: str = Field(..., description="Target model repository or ID")
    model_revision: str = Field(..., description="Immutable resolved Git commit SHA of the executed model")
    device_id: Optional[int] = Field(default=None, description="Logical accelerator device index (e.g. 0)")
    precision: str = Field(..., description="Precision used, e.g. 'fp16', 'bf16', 'fp32'")
    prompt: str = Field(..., description="Input prompt text submitted to the model")
    generated_text: Optional[str] = Field(default=None, description="Output text generated by the model (excluding prompt)")
    input_tokens: Optional[int] = Field(default=None, description="Actual count of prompt tokens encoded by tokenizer")
    generated_tokens: Optional[int] = Field(default=None, description="Actual count of newly generated tokens")
    started_at_utc: str = Field(default_factory=_utc_now_iso, description="ISO-8601 UTC timestamp of execution start")
    finished_at_utc: str = Field(default_factory=_utc_now_iso, description="ISO-8601 UTC timestamp of execution completion")
    error: Optional[str] = Field(default=None, description="Error message if execution failed or was skipped")
    generation_params: Dict[str, Any] = Field(default_factory=dict, description="Deterministic generation parameters")

    @model_validator(mode="after")
    def validate_execution_integrity(self) -> RunResult:
        """Enforce execution integrity: SUCCESS requires real output; SKIPPED/NOT_MEASURED forbids fake output."""
        if self.status == ExecutionStatus.SUCCESS:
            if self.generated_text is None:
                raise ValueError("SUCCESS status requires non-null generated_text")
            if self.input_tokens is None or self.input_tokens < 0:
                raise ValueError("SUCCESS status requires non-negative input_tokens")
            if self.generated_tokens is None or self.generated_tokens < 0:
                raise ValueError("SUCCESS status requires non-negative generated_tokens")

        if self.status in (ExecutionStatus.SKIPPED, ExecutionStatus.NOT_MEASURED):
            if self.generated_text is not None:
                raise ValueError(
                    f"Status '{self.status.value}' must not have generated_text (no synthetic output permitted)"
                )
            if self.input_tokens is not None or self.generated_tokens is not None:
                raise ValueError(
                    f"Status '{self.status.value}' must not have token counts (no synthetic output permitted)"
                )

        return self


class BenchmarkResult(BaseModel):
    """Performance measurements and execution status of a benchmark run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(default=CURRENT_SCHEMA_VERSION, description="Schema version of BenchmarkResult")
    status: ExecutionStatus = Field(..., description="Explicit execution outcome")
    error_message: Optional[str] = Field(default=None, description="Error details if execution failed")

    # Metrics with explicit units in names (Nullable for non-inference or diagnostic runs)
    ttft_ms: Optional[float] = Field(
        default=None,
        description="Time To First Token in milliseconds. Null if not measured.",
    )
    itl_ms_mean: Optional[float] = Field(
        default=None,
        description="Mean Inter-Token Latency in milliseconds. Null if not measured.",
    )
    itl_ms_p50: Optional[float] = Field(
        default=None,
        description="50th percentile (median) Inter-Token Latency in milliseconds. Null if not measured.",
    )
    itl_ms_p90: Optional[float] = Field(
        default=None,
        description="90th percentile Inter-Token Latency in milliseconds. Null if not measured.",
    )
    itl_ms_p99: Optional[float] = Field(
        default=None,
        description="99th percentile Inter-Token Latency in milliseconds. Null if not measured.",
    )
    throughput_tokens_per_sec: Optional[float] = Field(
        default=None,
        description="Tokens generated per second (throughput). Null if not measured.",
    )
    peak_vram_used_mb: Optional[float] = Field(
        default=None,
        description="Peak VRAM allocated/reserved during benchmark in MB. Null if not measured.",
    )
    total_latency_ms: Optional[float] = Field(
        default=None,
        description="Total elapsed benchmark time in milliseconds. Null if not measured.",
    )
    generated_tokens_count: Optional[int] = Field(
        default=None,
        description="Total number of generated tokens. Null if not measured.",
    )
    raw_latencies_ms: Optional[List[float]] = Field(
        default=None,
        description="Optional raw per-token latency sequence for detailed distribution profiling.",
    )
    # Workload metadata (optional / nullable)
    model_id: Optional[str] = Field(default=None, description="Target model repository or ID")
    model_revision: Optional[str] = Field(default=None, description="Immutable Git commit SHA of the benchmarked model")
    device_id: Optional[int] = Field(default=None, description="Logical accelerator device index")
    runtime_name: Optional[str] = Field(default=None, description="Runtime adapter identifier, e.g. 'pytorch_transformers_hip'")
    precision: Optional[str] = Field(default=None, description="Precision used, e.g. 'fp16', 'bf16', 'fp32'")

    # Run execution counts
    warmup_runs: Optional[int] = Field(default=None, description="Count of executed warmup iterations")
    measurement_runs_requested: Optional[int] = Field(default=None, description="Count of requested measurement iterations")
    measurement_runs_completed: Optional[int] = Field(default=None, description="Count of successfully completed measurement iterations")
    failed_runs: Optional[int] = Field(default=None, description="Count of failed measurement iterations")
    raw_measurements_reference: Optional[str] = Field(default=None, description="Optional path or URI to detailed raw run measurements")

    @model_validator(mode="after")
    def validate_metrics_consistency(self) -> BenchmarkResult:
        """Enforce strict integrity: Diagnostic/not-measured runs must NOT contain fake metrics."""
        measured_fields = [
            ("ttft_ms", self.ttft_ms),
            ("itl_ms_mean", self.itl_ms_mean),
            ("itl_ms_p50", self.itl_ms_p50),
            ("itl_ms_p90", self.itl_ms_p90),
            ("itl_ms_p99", self.itl_ms_p99),
            ("throughput_tokens_per_sec", self.throughput_tokens_per_sec),
            ("peak_vram_used_mb", self.peak_vram_used_mb),
            ("total_latency_ms", self.total_latency_ms),
            ("generated_tokens_count", self.generated_tokens_count),
            ("raw_latencies_ms", self.raw_latencies_ms),
        ]

        if self.status in (ExecutionStatus.NOT_MEASURED, ExecutionStatus.SKIPPED):
            non_null_fields = [name for name, val in measured_fields if val is not None]
            if non_null_fields:
                raise ValueError(
                    f"Diagnostic / non-measured results must not contain benchmark metrics. "
                    f"Found non-null values for: {', '.join(non_null_fields)} with status '{self.status.value}'."
                )

        if self.status == ExecutionStatus.SUCCESS:
            for name, val in measured_fields:
                if val is not None and not isinstance(val, list) and val < 0:
                    raise ValueError(f"Metric '{name}' cannot be negative, got {val}")
        return self


class ExperimentSpec(BaseModel):
    """Specification of an experiment run, linking inputs, environment, and configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(default=CURRENT_SCHEMA_VERSION, description="Schema version of ExperimentSpec")
    experiment_id: str = Field(..., description="Unique identifier of this experiment (e.g. UUID)")
    model: ModelSpec = Field(..., description="Target model specification")
    hardware: HardwareSpec = Field(..., description="Target hardware specification")
    environment: EnvironmentSpec = Field(..., description="Software environment specification")
    runtime_name: str = Field(..., description="Runtime adapter identifier, e.g. 'hf-transformers'")
    precision: str = Field(..., description="Floating-point precision, e.g. 'fp16', 'bf16', 'fp32'")
    benchmark_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters passed to benchmark harness (prompt_tokens, max_new_tokens, etc.)",
    )
    created_at_utc: str = Field(
        default_factory=_utc_now_iso,
        description="Creation timestamp in UTC (ISO-8601)",
    )


class ValidationCase(BaseModel):
    """Specification of an individual test case in the validation suite."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(..., description="Stable unique identifier of the test case, e.g. 'basic_completion_001'")
    prompt: str = Field(..., description="Input text prompt submitted to the model")
    max_new_tokens: int = Field(default=16, description="Token limit for this case")
    critical: bool = Field(default=False, description="True if failure in this case strictly blocks PASS verdict")
    description: Optional[str] = Field(default=None, description="Human-readable description of case intent")
    expected_pattern: Optional[str] = Field(default=None, description="Optional regex pattern to test against output")


class ValidationRunResult(BaseModel):
    """Execution evidence from running a single validation test case."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(default=CURRENT_SCHEMA_VERSION, description="Schema version of ValidationRunResult")
    case_id: str = Field(..., description="Identifier of the executed validation test case")
    status: ExecutionStatus = Field(..., description="Execution outcome: SUCCESS, FAILED, SKIPPED, NOT_MEASURED")
    input_tokens: Optional[int] = Field(default=None, description="Count of prompt tokens encoded")
    generated_tokens: Optional[int] = Field(default=None, description="Count of newly generated tokens")
    generated_text: Optional[str] = Field(default=None, description="Generated output text (excluding prompt)")
    error: Optional[str] = Field(default=None, description="Error message if run failed or was skipped")
    run_duration_ms: Optional[float] = Field(default=None, description="Execution elapsed time in milliseconds")

    @model_validator(mode="after")
    def validate_run_integrity(self) -> ValidationRunResult:
        """Enforce validation execution integrity."""
        if self.status == ExecutionStatus.SUCCESS:
            if self.generated_text is None:
                raise ValueError("SUCCESS status requires non-null generated_text")
            if self.generated_tokens is None or self.generated_tokens < 0:
                raise ValueError("SUCCESS status requires non-negative generated_tokens")
        if self.status in (ExecutionStatus.SKIPPED, ExecutionStatus.NOT_MEASURED):
            if self.generated_text is not None:
                raise ValueError(f"Status '{self.status.value}' must not have generated_text")
            if self.generated_tokens is not None or self.input_tokens is not None:
                raise ValueError(f"Status '{self.status.value}' must not have token counts")
        return self


class ValidationReport(BaseModel):
    """Consolidated assessment report of model correctness and quality validation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(default=CURRENT_SCHEMA_VERSION, description="Schema version of ValidationReport")
    mode: ValidationMode = Field(..., description="Validation suite mode: SELF_VALIDATION or COMPARISON")
    model_id: str = Field(..., description="Target model repository or ID")
    baseline_revision: str = Field(..., description="Immutable Git commit SHA of the baseline model snapshot")
    candidate_revision: Optional[str] = Field(
        default=None,
        description="Immutable Git commit SHA of candidate model (None in SELF_VALIDATION mode)",
    )
    verdict: ValidationVerdict = Field(..., description="Top-level validation verdict: PASS, FAIL, INCONCLUSIVE, NOT_MEASURED")

    # Correctness and Quality outcomes
    correctness_passed: Optional[bool] = Field(
        default=None,
        description="True if all correctness gates passed; None if not measured",
    )
    quality_measured: bool = Field(
        default=False,
        description="True if comparative quality metrics were formally evaluated",
    )
    qrr_percent: Optional[float] = Field(
        default=None,
        description="Quality Retention Rate percentage (candidate_score / baseline_score * 100). None if not measured.",
    )

    # Case execution statistics
    cases_total: int = Field(default=0, description="Total count of test cases in suite")
    cases_completed: int = Field(default=0, description="Count of successfully executed test cases")
    cases_failed: int = Field(default=0, description="Count of failed test cases")
    critical_cases_failed: int = Field(default=0, description="Count of failed critical test cases")

    # Evidence details
    case_results: List[ValidationRunResult] = Field(default_factory=list, description="Per-case execution evidence")
    reasons: List[str] = Field(default_factory=list, description="Structured explanations for verdict")
    warnings: List[str] = Field(default_factory=list, description="Non-fatal warnings or advisories")
    created_at_utc: str = Field(default_factory=_utc_now_iso, description="ISO-8601 UTC timestamp")

    @model_validator(mode="after")
    def validate_report_integrity(self) -> ValidationReport:
        """Enforce validation verdict consistency."""
        if self.verdict == ValidationVerdict.PASS:
            if self.critical_cases_failed > 0:
                raise ValueError("Verdict cannot be PASS when critical cases failed")
            if self.correctness_passed is not True:
                raise ValueError("Verdict cannot be PASS when correctness did not pass")

        if self.verdict == ValidationVerdict.NOT_MEASURED:
            if self.correctness_passed is not None:
                raise ValueError("NOT_MEASURED verdict must have correctness_passed as None")
            if self.quality_measured is True:
                raise ValueError("NOT_MEASURED verdict cannot have quality_measured as True")
            if self.qrr_percent is not None:
                raise ValueError("NOT_MEASURED verdict must have qrr_percent as None")

        return self


class ArtifactFileEntry(BaseModel):
    """Inventory entry representing an individual file contained in an artifact bundle."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(..., description="Relative file path within the artifact bundle, e.g. 'model.json'")
    sha256: str = Field(..., description="SHA-256 hex digest of the canonical file content")
    size_bytes: int = Field(..., description="File size in bytes")


class ReproductionMetadata(BaseModel):
    """Declarative parameters and specifications required to reproduce the experiment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(default=CURRENT_SCHEMA_VERSION, description="Schema version of ReproductionMetadata")
    model_id: str = Field(..., description="Target model identifier")
    requested_revision: str = Field(default="main", description="Requested model revision")
    immutable_revision: str = Field(..., description="Immutable resolved Git commit SHA")
    runtime: str = Field(..., description="Execution runtime adapter identifier")
    precision: str = Field(..., description="Inference precision (e.g. 'fp16', 'bf16', 'fp32')")
    device_requirements: Dict[str, Any] = Field(default_factory=dict, description="Target device parameters")
    benchmark_config: Dict[str, Any] = Field(default_factory=dict, description="Benchmark workload configuration")
    validation_config: Dict[str, Any] = Field(default_factory=dict, description="Validation suite configuration")
    environment_requirements: Dict[str, Any] = Field(default_factory=dict, description="Environment constraints")


class ArtifactVerificationResult(BaseModel):
    """Detailed outcome of artifact bundle integrity and manifest verification."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(default=CURRENT_SCHEMA_VERSION, description="Schema version of ArtifactVerificationResult")
    valid: bool = Field(..., description="True if artifact bundle is completely intact, unaltered, and valid")
    artifact_id: str = Field(..., description="Artifact identifier under verification")
    manifest_valid: bool = Field(..., description="True if manifest.json is present, parses cleanly, and passes schema validation")
    checksums_valid: bool = Field(..., description="True if checksums.json matches all file hashes including manifest.json")
    missing_files: List[str] = Field(default_factory=list, description="Files declared in inventory but missing from disk")
    modified_files: List[str] = Field(default_factory=list, description="Files whose computed SHA-256 does not match declared checksum")
    unexpected_files: List[str] = Field(default_factory=list, description="Files present on disk but not declared in inventory")
    errors: List[str] = Field(default_factory=list, description="Descriptive error messages explaining failure reasons")


class ArtifactManifest(BaseModel):
    """Top-level reproducible artifact manifest bundling experiment inputs, evidence, and verification inventories."""

    model_config = ConfigDict(extra="forbid", frozen=False)

    schema_version: str = Field(default=CURRENT_SCHEMA_VERSION, description="Schema version of ArtifactManifest")
    artifact_id: str = Field(..., description="Content-derived unique artifact identifier")
    experiment_id: str = Field(..., description="Deterministic canonical identity of the experiment configuration")
    status: ArtifactStatus = Field(default=ArtifactStatus.COMPLETE, description="Structural completeness status of bundle")
    created_at_utc: str = Field(
        default_factory=_utc_now_iso,
        description="Creation timestamp in UTC (ISO-8601)",
    )

    # Component summaries
    model: Dict[str, Any] = Field(default_factory=dict, description="Model provenance summary")
    hardware: Dict[str, Any] = Field(default_factory=dict, description="Hardware summary")
    environment: Dict[str, Any] = Field(default_factory=dict, description="Software environment summary")

    # Execution parameters & outcomes
    runtime: str = Field(default="pytorch_transformers_hip", description="Runtime adapter identifier")
    precision: str = Field(default="fp16", description="Floating-point precision")

    capability_verdict: EvaluationVerdict = Field(default=EvaluationVerdict.NO_ACCELERATOR, description="Preflight capability evaluation verdict")
    execution_status: ExecutionStatus = Field(default=ExecutionStatus.SKIPPED, description="Baseline execution status")
    benchmark_status: ExecutionStatus = Field(default=ExecutionStatus.SKIPPED, description="Benchmark execution status")
    validation_verdict: ValidationVerdict = Field(default=ValidationVerdict.NOT_MEASURED, description="Validation suite verdict")

    # Inventory of constituent bundle files (path -> ArtifactFileEntry)
    files: Dict[str, ArtifactFileEntry] = Field(default_factory=dict, description="Inventory mapping relative paths to file checksums and sizes")

    # Reproduction reference / parameters
    reproduction: Dict[str, Any] = Field(default_factory=dict, description="Reproduction specification summary")

    manifest_checksum: Optional[str] = Field(
        default=None,
        description="Detached SHA256 checksum of canonical manifest JSON content (excluding manifest_checksum field)",
    )

    # Legacy fields for backward compatibility with early unit tests
    experiment: Optional[ExperimentSpec] = Field(
        default=None,
        description="Legacy experiment spec field (optional)",
    )
    result: Optional[BenchmarkResult] = Field(
        default=None,
        description="Legacy benchmark result field (optional)",
    )
    reproduce_command: Optional[str] = Field(
        default=None,
        description="Optional shell command string to reproduce this run",
    )
    manifest_id: Optional[str] = Field(
        default=None,
        description="Legacy alias for artifact_id",
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "artifact_id" not in data and "manifest_id" in data:
                data["artifact_id"] = data["manifest_id"]
            elif "manifest_id" not in data and "artifact_id" in data:
                data["manifest_id"] = data["artifact_id"]
            if "experiment_id" not in data:
                if "experiment" in data and isinstance(data["experiment"], (dict, ExperimentSpec)):
                    exp_id = (
                        data["experiment"].experiment_id
                        if isinstance(data["experiment"], ExperimentSpec)
                        else data["experiment"].get("experiment_id", "exp-legacy")
                    )
                    data["experiment_id"] = exp_id
                else:
                    data["experiment_id"] = "exp-default"
            if "experiment" in data and isinstance(data["experiment"], (dict, ExperimentSpec)):
                exp = data["experiment"]
                if isinstance(exp, ExperimentSpec):
                    if not data.get("model"):
                        data["model"] = exp.model.model_dump(mode="json")
                    if not data.get("runtime"):
                        data["runtime"] = exp.runtime_name
                    if not data.get("precision"):
                        data["precision"] = exp.precision
            if "result" in data and isinstance(data["result"], (dict, BenchmarkResult)):
                res = data["result"]
                if isinstance(res, BenchmarkResult):
                    if "benchmark_status" not in data:
                        data["benchmark_status"] = res.status
        return data

    @model_validator(mode="after")
    def validate_no_premature_certification(self) -> ArtifactManifest:
        """Enforce that artifact never claims certification or verification."""
        forbidden_keys = {"verified", "certified", "rocmhub_verified", "badge"}
        for k in forbidden_keys:
            if k in self.__dict__:
                raise ValueError(f"Prohibited certification field '{k}' found in ArtifactManifest.")
        return self

    def compute_canonical_checksum(self) -> str:
        """Compute the SHA256 checksum of this manifest in canonical JSON format, excluding manifest_checksum."""
        data = self.model_dump(mode="json")
        data.pop("manifest_checksum", None)
        canonical_json = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    def sign_manifest(self) -> ArtifactManifest:
        """Calculate and set the manifest_checksum field."""
        self.manifest_checksum = self.compute_canonical_checksum()
        return self

    def verify_checksum(self) -> bool:
        """Verify whether the current manifest matches its stored manifest_checksum."""
        if not self.manifest_checksum:
            return False
        return self.compute_canonical_checksum() == self.manifest_checksum


class HardwareHealthSnapshot(BaseModel):
    """Snapshot of hardware health and telemetry observed during execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    device_id: Optional[int] = Field(default=None, description="Target GPU device index")
    temperature_c: Optional[float] = Field(default=None, description="GPU temperature in degrees Celsius")
    power_w: Optional[float] = Field(default=None, description="Instantaneous power consumption in Watts")
    gpu_utilization_percent: Optional[float] = Field(default=None, description="GPU compute utilization percentage")
    memory_utilization_percent: Optional[float] = Field(default=None, description="VRAM memory utilization percentage")
    clock_mhz: Optional[int] = Field(default=None, description="Engine/shader clock frequency in MHz")
    memory_clock_mhz: Optional[int] = Field(default=None, description="Memory clock frequency in MHz")
    ecc_errors: Optional[int] = Field(default=None, description="Uncorrectable ECC error count")
    throttling_detected: Optional[bool] = Field(default=None, description="Whether thermal or power throttling was observed")


class GuardPolicy(BaseModel):
    """Configurable thresholds and rules governing the Benchmark Guard reproducibility evaluation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_version: str = Field(default="1.0.0", description="Policy version identifier")
    min_measurement_runs: int = Field(default=5, description="Minimum number of successful measurement runs required")
    max_relative_mad_ttft: float = Field(default=0.15, description="Maximum allowed relative MAD for TTFT (15%)")
    max_relative_mad_throughput: float = Field(default=0.10, description="Maximum allowed relative MAD for throughput (10%)")
    max_relative_mad_latency: float = Field(default=0.15, description="Maximum allowed relative MAD for total latency (15%)")
    require_environment_match: bool = Field(default=True, description="Enforce strict matching of expected environment fingerprint")
    require_complete_benchmark: bool = Field(default=True, description="Require zero failed measurement runs")
    strict_telemetry: bool = Field(default=False, description="Require telemetry availability (rejects runs without telemetry if True)")
    max_tolerated_ecc_errors: int = Field(default=0, description="Maximum allowable ECC errors before flagging FAIL")
    summary_recomputation_tolerance: float = Field(default=1e-3, description="Relative tolerance for summary recomputation checks")


class EnvironmentFingerprint(BaseModel):
    """Deterministic, canonical hardware and software environment signature."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(default=CURRENT_SCHEMA_VERSION, description="Schema version of fingerprint")
    fingerprint_hash: str = Field(..., description="Canonical SHA-256 digest of normalized environment attributes")
    attributes: Dict[str, Any] = Field(..., description="Normalized environment attributes excluding ephemeral runtime noise")


class ReferenceMeasurement(BaseModel):
    """Calibration or baseline reference measurement recorded before/after evaluation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reference_id: str = Field(..., description="Reference benchmark or hardware check identifier")
    metric_name: str = Field(..., description="Name of measured reference metric, e.g. 'kernel_gemm_tflops'")
    value: float = Field(..., description="Measured numerical value")
    timestamp_utc: str = Field(default_factory=_utc_now_iso, description="ISO-8601 UTC timestamp")


class ReferenceComparison(BaseModel):
    """Comparison of reference metrics across experiment iterations to detect hardware drift."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reference_id: str = Field(..., description="Reference benchmark identifier")
    metric_name: str = Field(..., description="Name of reference metric")
    before_value: float = Field(..., description="Reference measurement before experiment")
    after_value: float = Field(..., description="Reference measurement after experiment")
    relative_change: float = Field(..., description="Relative variation |after - before| / before")
    stable: bool = Field(..., description="True if relative variation is within acceptable bounds")


class ReproducibilityReport(BaseModel):
    """Independent audit report deciding whether benchmark evidence is valid, truthful, and reproducible."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(default=CURRENT_SCHEMA_VERSION, description="Schema version of report")
    artifact_id: str = Field(..., description="Artifact bundle identifier evaluated")
    experiment_id: str = Field(..., description="Experiment configuration identifier")
    policy_version: str = Field(default="1.0.0", description="Version of GuardPolicy applied")

    verdict: GuardVerdict = Field(..., description="Guard verdict: PASS, FAIL, INCONCLUSIVE, or NOT_MEASURED")

    environment_consistent: bool = Field(..., description="True if observed environment matches expected fingerprint")
    hardware_consistent: bool = Field(..., description="True if no throttling or unrecoverable hardware errors were detected")
    benchmark_complete: bool = Field(..., description="True if all requested measurement runs completed without error")
    reference_stable: Optional[bool] = Field(
        default=None,
        description="True if before/after reference checks were stable; None if reference measurements are absent",
    )

    measurement_runs: int = Field(..., description="Total measurement runs evaluated")
    valid_runs: int = Field(..., description="Successful measurement runs analyzed")

    ttft_variability: Optional[float] = Field(default=None, description="Relative MAD for TTFT across valid runs")
    throughput_variability: Optional[float] = Field(default=None, description="Relative MAD for throughput across valid runs")
    itl_variability: Optional[float] = Field(default=None, description="Relative MAD for total latency / ITL across valid runs")

    reasons: List[str] = Field(default_factory=list, description="Structured reason codes explaining verdict")
    warnings: List[str] = Field(default_factory=list, description="Non-fatal warnings or advisories")
    created_at_utc: str = Field(default_factory=_utc_now_iso, description="Evaluation timestamp in UTC")

    @model_validator(mode="after")
    def validate_guard_report_invariants(self) -> ReproducibilityReport:
        """Enforce guard verdict integrity."""
        if self.verdict == GuardVerdict.PASS:
            if not self.environment_consistent:
                raise ValueError("PASS verdict requires environment_consistent=True")
            if not self.hardware_consistent:
                raise ValueError("PASS verdict requires hardware_consistent=True")
            if not self.benchmark_complete:
                raise ValueError("PASS verdict requires benchmark_complete=True")
            if self.valid_runs < 1:
                raise ValueError("PASS verdict requires at least one valid measurement run")
        elif self.verdict == GuardVerdict.NOT_MEASURED:
            if self.ttft_variability is not None or self.throughput_variability is not None:
                raise ValueError("NOT_MEASURED verdict cannot have variability metrics recorded")
        return self
