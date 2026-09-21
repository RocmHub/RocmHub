"""Abstract contract and protocol for model execution runners."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from rocmhub.core.types import CapabilityReport, ModelSpec, RunResult


@runtime_checkable
class BaseRunner(Protocol):
    """Protocol for model execution runtime adapters.

    Responsibilities:
    - Responsible ONLY for loading model weights onto the target accelerator,
      executing deterministic inference, and releasing device memory.
    - Does NOT decide capability policies or matrix matching (handled by CapabilityEvaluator).
    - Does NOT perform benchmark statistical latency analysis (handled by BenchmarkHarness).
    - Does NOT assign badges, select quantizations, or build manifests.

    Lifecycle:
      load() -> generate() -> unload()
    """

    @property
    def runner_name(self) -> str:
        """Name of the runtime adapter, e.g. 'pytorch_transformers_hip'."""
        ...

    def supports(self, capability_report: CapabilityReport, device_id: int) -> bool:
        """Evaluate whether this runner is permitted to execute on target device.

        Args:
            capability_report: Preflight capability report from CapabilityEvaluator.
            device_id: Target GPU device index.

        Returns:
            True if preflight verdict and device assessment permit execution.
        """
        ...

    def load(self, model: ModelSpec, device_id: int, precision: str = "fp16") -> None:
        """Load model weights and tokenizer onto the designated target device.

        Args:
            model: Target model specification with immutable commit SHA.
            device_id: Concrete accelerator index (e.g. 0 for cuda:0 / hip:0).
            precision: Floating point precision ('fp32', 'fp16', 'bf16').

        Raises:
            ModelLoadError: If weights or tokenizer fail to load.
            DeviceNotAvailableError: If target device does not exist or is unavailable.
            UnsupportedPrecisionError: If requested precision is unsupported.
            UnsupportedModelTypeError: If model architecture is unsupported.
        """
        ...

    def generate(self, prompt: str, max_new_tokens: int = 16, **kwargs: Any) -> RunResult:
        """Execute deterministic greedy generation on the loaded model.

        Args:
            prompt: Text prompt to submit.
            max_new_tokens: Maximum number of tokens to generate.
            **kwargs: Additional generation parameters (e.g. temperature=0, do_sample=False).

        Returns:
            RunResult confirming actual inference execution with token counts.

        Raises:
            RunnerNotReadyError: If called prior to load().
            GenerationError: If forward pass or decoding fails.
        """
        ...

    def unload(self) -> None:
        """Release device and host memory, clean references, and collect garbage."""
        ...
