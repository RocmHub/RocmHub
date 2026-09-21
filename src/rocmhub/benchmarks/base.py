"""Base specifications and data contracts for ROCmHub benchmark harness."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rocmhub.core.errors import InvalidBenchmarkConfigError
from rocmhub.core.types import CURRENT_SCHEMA_VERSION, ExecutionStatus


class BenchmarkConfig(BaseModel):
    """Workload and hardware configuration for benchmark executions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(default=CURRENT_SCHEMA_VERSION, description="Schema version of BenchmarkConfig")
    prompt: str = Field(default="Hello, ROCm!", description="Input text prompt for generation")
    max_new_tokens: int = Field(default=16, description="Maximum count of newly generated tokens")
    warmup_runs: int = Field(default=2, description="Count of untimed warmup runs before measurements")
    measurement_runs: int = Field(default=5, description="Count of timed measurement runs")
    precision: str = Field(default="fp16", description="Floating-point precision: 'fp32', 'fp16', 'bf16'")
    device_id: int = Field(default=0, description="Target accelerator device index")
    seed: Optional[int] = Field(default=None, description="Optional random seed for reproducibility")
    synchronize_device: bool = Field(
        default=True,
        description="Whether to synchronize accelerator device before and after measured execution",
    )
    collect_memory: bool = Field(
        default=True,
        description="Whether to record allocator peak memory usage for measured runs",
    )

    @field_validator("warmup_runs")
    @classmethod
    def validate_warmup_runs(cls, v: int) -> int:
        if v < 0:
            raise InvalidBenchmarkConfigError(
                f"warmup_runs must be non-negative (got {v})",
                details={"warmup_runs": v},
            )
        return v

    @field_validator("measurement_runs")
    @classmethod
    def validate_measurement_runs(cls, v: int) -> int:
        if v < 1:
            raise InvalidBenchmarkConfigError(
                f"measurement_runs must be at least 1 (got {v})",
                details={"measurement_runs": v},
            )
        return v

    @field_validator("max_new_tokens")
    @classmethod
    def validate_max_new_tokens(cls, v: int) -> int:
        if v <= 0:
            raise InvalidBenchmarkConfigError(
                f"max_new_tokens must be strictly positive (got {v})",
                details={"max_new_tokens": v},
            )
        return v

    @field_validator("device_id")
    @classmethod
    def validate_device_id(cls, v: int) -> int:
        if v < 0:
            raise InvalidBenchmarkConfigError(
                f"device_id must be non-negative (got {v})",
                details={"device_id": v},
            )
        return v

    @field_validator("precision")
    @classmethod
    def validate_precision(cls, v: str) -> str:
        cleaned = v.strip().lower()
        valid = {"fp32", "float32", "fp16", "float16", "bf16", "bfloat16"}
        if cleaned not in valid:
            raise InvalidBenchmarkConfigError(
                f"Unsupported precision '{v}'. Supported: 'fp32', 'fp16', 'bf16'.",
                details={"precision": v},
            )
        return cleaned


class BenchmarkRunMeasurement(BaseModel):
    """Raw timing and resource measurement recorded for a single benchmark run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(default=CURRENT_SCHEMA_VERSION, description="Schema version of measurement")
    run_index: int = Field(..., description="Zero-based iteration index of the run")
    is_warmup: bool = Field(default=False, description="True if this was an untimed warmup cycle")
    status: ExecutionStatus = Field(..., description="Execution status: SUCCESS or FAILED")
    input_tokens: Optional[int] = Field(default=None, description="Actual count of input prompt tokens")
    generated_tokens: Optional[int] = Field(default=None, description="Actual count of newly decoded tokens")

    # High-resolution nanosecond timestamps (from monotonic time.perf_counter_ns)
    started_at_ns: int = Field(..., description="Monotonic nanosecond timestamp of generation start")
    first_token_at_ns: Optional[int] = Field(
        default=None,
        description="Monotonic nanosecond timestamp of first generated token emission",
    )
    finished_at_ns: int = Field(..., description="Monotonic nanosecond timestamp of generation completion")

    # Calculated run-level metrics (in milliseconds)
    ttft_ms: Optional[float] = Field(
        default=None,
        description="Time To First Token in ms (first_token_at_ns - started_at_ns) / 1e6",
    )
    inter_token_latencies_ms: List[float] = Field(
        default_factory=list,
        description="Consecutive inter-token latency deltas (t_{i+1} - t_i) in ms. Excludes TTFT.",
    )
    total_latency_ms: float = Field(
        ...,
        description="Total duration from generation request to return in ms (finished_at_ns - started_at_ns) / 1e6",
    )
    peak_vram_used_mb: Optional[float] = Field(
        default=None,
        description="Peak allocator memory allocated during this run in MB",
    )
    error: Optional[str] = Field(default=None, description="Error message if run failed")

    @model_validator(mode="after")
    def validate_measurement_consistency(self) -> BenchmarkRunMeasurement:
        """Enforce measurement integrity between status and recorded values."""
        if self.status == ExecutionStatus.SUCCESS:
            if self.generated_tokens is not None and self.generated_tokens > 0:
                if self.first_token_at_ns is None:
                    raise ValueError("SUCCESS run with generated tokens must have first_token_at_ns recorded")
                if self.ttft_ms is None:
                    raise ValueError("SUCCESS run with generated tokens must have ttft_ms calculated")
        return self
