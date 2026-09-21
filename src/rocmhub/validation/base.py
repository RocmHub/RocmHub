"""Base configuration and definitions for ROCmHub correctness and quality validation."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from rocmhub.core.errors import InvalidValidationConfigError
from rocmhub.core.types import CURRENT_SCHEMA_VERSION, ValidationCase, ValidationMode

DEFAULT_VALIDATION_CASES: List[ValidationCase] = [
    ValidationCase(
        case_id="basic_completion_001",
        prompt="The capital of France is",
        max_new_tokens=16,
        critical=True,
        description="Factual completion test verifying basic causal decoding without corruption.",
    ),
    ValidationCase(
        case_id="instruction_following_001",
        prompt="Translate the word 'apple' to French:",
        max_new_tokens=16,
        critical=True,
        description="Minimal instruction-following sanity check.",
    ),
    ValidationCase(
        case_id="deterministic_generation_001",
        prompt="Count from 1 to 5: 1, 2,",
        max_new_tokens=16,
        critical=False,
        description="Deterministic generation verification under fixed seed.",
    ),
]


class ValidationConfig(BaseModel):
    """Configuration for validation suite execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(default=CURRENT_SCHEMA_VERSION, description="Schema version of ValidationConfig")
    cases: List[ValidationCase] = Field(
        default_factory=lambda: list(DEFAULT_VALIDATION_CASES),
        description="List of validation test cases to execute",
    )
    max_new_tokens: int = Field(default=16, description="Default max new tokens per case")
    seed: Optional[int] = Field(default=42, description="Random seed for deterministic inference")
    deterministic: bool = Field(default=True, description="Enforce greedy deterministic generation (temperature=0)")
    quality_threshold: float = Field(
        default=0.95,
        description="Minimum Quality Retention Rate (QRR) or score ratio to pass comparison gate (0.0 - 1.0)",
    )
    critical_regression_tolerance: float = Field(
        default=0.0,
        description="Maximum allowed failure rate on critical cases (default 0.0: zero tolerance)",
    )
    device_id: int = Field(default=0, description="Target accelerator device index")
    precision: str = Field(default="fp16", description="Model weight precision ('fp32', 'fp16', 'bf16')")
    mode: ValidationMode = Field(
        default=ValidationMode.SELF_VALIDATION,
        description="Validation mode: SELF_VALIDATION or COMPARISON",
    )

    @field_validator("cases")
    @classmethod
    def validate_cases(cls, v: List[ValidationCase]) -> List[ValidationCase]:
        if not v:
            raise InvalidValidationConfigError("Validation cases list must not be empty")
        return v

    @field_validator("max_new_tokens")
    @classmethod
    def validate_max_new_tokens(cls, v: int) -> int:
        if v <= 0:
            raise InvalidValidationConfigError(
                f"max_new_tokens must be strictly positive (got {v})",
                details={"max_new_tokens": v},
            )
        return v

    @field_validator("device_id")
    @classmethod
    def validate_device_id(cls, v: int) -> int:
        if v < 0:
            raise InvalidValidationConfigError(
                f"device_id must be non-negative (got {v})",
                details={"device_id": v},
            )
        return v

    @field_validator("quality_threshold")
    @classmethod
    def validate_quality_threshold(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise InvalidValidationConfigError(
                f"quality_threshold must be between 0.0 and 1.0 inclusive (got {v})",
                details={"quality_threshold": v},
            )
        return v

    @field_validator("critical_regression_tolerance")
    @classmethod
    def validate_critical_tolerance(cls, v: float) -> float:
        if v < 0.0 or v > 1.0:
            raise InvalidValidationConfigError(
                f"critical_regression_tolerance must be between 0.0 and 1.0 (got {v})",
                details={"critical_regression_tolerance": v},
            )
        return v

    @field_validator("precision")
    @classmethod
    def validate_precision(cls, v: str) -> str:
        cleaned = v.strip().lower()
        valid = {"fp32", "float32", "fp16", "float16", "bf16", "bfloat16"}
        if cleaned not in valid:
            raise InvalidValidationConfigError(
                f"Unsupported precision '{v}'. Supported: 'fp32', 'fp16', 'bf16'.",
                details={"precision": v},
            )
        return cleaned
