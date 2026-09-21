"""Domain-specific exceptions for ROCmHub."""

from __future__ import annotations

from typing import Any, Dict, Optional


class ROCmHubError(Exception):
    """Base exception for all ROCmHub errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} (details: {self.details})"
        return self.message


class ConfigurationError(ROCmHubError):
    """Raised when configuration or runtime arguments are invalid."""


class SchemaValidationError(ROCmHubError):
    """Raised when data model invariants or schema validations fail."""


class HardwareDetectionError(ROCmHubError):
    """Raised when hardware probing encounters an unrecoverable failure."""


class ModelResolutionError(ROCmHubError):
    """Raised when model source cannot be resolved or commit SHA cannot be fetched."""


class ModelInspectionError(ROCmHubError):
    """Raised when static inspection of model config or weights fails."""


class RunnerExecutionError(ROCmHubError):
    """Raised when inference runner fails during setup, warmup, or generation."""


class BenchmarkError(ROCmHubError):
    """Raised when benchmark harness fails to collect or compute metrics."""


class ArtifactPackagingError(ROCmHubError):
    """Raised when artifact manifest, packaging, or checksum validation fails."""


class NotImplementedFeatureError(ROCmHubError):
    """Raised when a feature planned for a future phase is requested."""
