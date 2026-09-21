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


class ArtifactManifest(BaseModel):
    """Top-level reproducible artifact manifest bundling experiment inputs and benchmark results."""

    model_config = ConfigDict(extra="forbid", frozen=False)

    schema_version: str = Field(default=CURRENT_SCHEMA_VERSION, description="Schema version of ArtifactManifest")
    manifest_id: str = Field(..., description="Artifact manifest identifier")
    created_at_utc: str = Field(
        default_factory=_utc_now_iso,
        description="Creation timestamp in UTC (ISO-8601)",
    )
    experiment: ExperimentSpec = Field(..., description="Experiment specification and provenance")
    result: BenchmarkResult = Field(..., description="Benchmark results and execution status")
    reproduce_command: str = Field(..., description="Command to reproduce this run")
    manifest_checksum: Optional[str] = Field(
        default=None,
        description="SHA256 checksum of canonical manifest JSON content (excluding manifest_checksum field)",
    )

    def compute_canonical_checksum(self) -> str:
        """Compute the SHA256 checksum of this manifest in canonical JSON format, excluding manifest_checksum."""
        # Convert to dictionary and exclude manifest_checksum
        data = self.model_dump(mode="json")
        data.pop("manifest_checksum", None)
        # Produce canonical deterministic JSON string with sorted keys
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
