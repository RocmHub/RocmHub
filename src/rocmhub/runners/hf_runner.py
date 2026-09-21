"""HuggingFaceRunner: PyTorch + Transformers baseline runtime adapter for AMD ROCm/HIP."""

from __future__ import annotations

import gc
from contextlib import nullcontext
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, Set

try:
    import torch  # type: ignore[import-not-found]
except ImportError:
    torch = None  # type: ignore[assignment]

from rocmhub.core.errors import (
    DeviceNotAvailableError,
    GenerationError,
    ModelLoadError,
    RunnerNotReadyError,
    UnsupportedModelTypeError,
    UnsupportedPrecisionError,
)
from rocmhub.core.types import (
    CapabilityReport,
    EvaluationVerdict,
    ExecutionStatus,
    ModelSpec,
    RunResult,
)


def _utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _resolve_torch_dtype(precision_name: str) -> Any:
    """Map precision string to torch.dtype if torch is available."""
    if torch is None:
        return precision_name
    mapping = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    return mapping.get(precision_name, precision_name)


class HuggingFaceRunner:
    """Baseline PyTorch + Transformers execution runner for AMD ROCm/HIP accelerators.

    Adheres strictly to the BaseRunner contract:
    - Lifecycle: load() -> generate() -> unload().
    - Uses immutable commit SHA from ModelSpec (`revision=model.commit_sha`).
    - Explicitly sets `trust_remote_code=False`.
    - Targets a concrete single accelerator (`cuda:device_id`); never uses `device_map="auto"`.
    - Executes deterministic greedy decoding (temperature=0, do_sample=False).
    - Accurately counts input and generated tokens via the tokenizer.
    - Reclaims memory in unload().
    """

    RUNNER_NAME = "pytorch_transformers_hip"

    # Supported baseline precision types
    SUPPORTED_PRECISIONS: Set[str] = {"fp32", "fp16", "bf16"}

    def __init__(
        self,
        model_loader: Optional[Callable[..., Any]] = None,
        tokenizer_loader: Optional[Callable[..., Any]] = None,
        device_override: Optional[str] = None,
    ) -> None:
        """Initialize runner with optional dependency injection for testability.

        Args:
            model_loader: Callable to load causal model (defaults to AutoModelForCausalLM.from_pretrained).
            tokenizer_loader: Callable to load tokenizer (defaults to AutoTokenizer.from_pretrained).
            device_override: Optional device string override for test-only isolation (e.g. 'cpu').
        """
        self._model_loader = model_loader
        self._tokenizer_loader = tokenizer_loader
        self._device_override = device_override

        # Lifecycle state
        self._is_loaded = False
        self._loaded_model: Optional[Any] = None
        self._loaded_tokenizer: Optional[Any] = None
        self._loaded_model_spec: Optional[ModelSpec] = None
        self._device_id: Optional[int] = None
        self._target_device: Optional[Any] = None
        self._precision: Optional[str] = None

    @property
    def runner_name(self) -> str:
        """Name of the runtime adapter."""
        return self.RUNNER_NAME

    @property
    def is_loaded(self) -> bool:
        """True if a model is currently loaded and ready for generation."""
        return self._is_loaded

    def supports(self, capability_report: CapabilityReport, device_id: int) -> bool:
        """Evaluate whether this runner is permitted to execute on target device.

        Args:
            capability_report: Preflight report from CapabilityEvaluator.
            device_id: Logical GPU index.

        Returns:
            True if system verdict is READY and designated device is READY.
        """
        if capability_report.verdict != EvaluationVerdict.READY:
            return False

        if capability_report.capabilities.baseline_runtime_candidate != self.RUNNER_NAME:
            return False

        for assessment in capability_report.device_assessments:
            if assessment.device_id == device_id:
                return assessment.verdict == EvaluationVerdict.READY

        return False

    def _is_causal_lm_architecture(self, architecture: Optional[str]) -> bool:
        """Verify whether architecture matches supported causal language modeling classes."""
        if not architecture:
            return False
        arch_lower = architecture.lower()
        return (
            architecture.endswith("ForCausalLM")
            or architecture.endswith("LMHeadModel")
            or "causallm" in arch_lower
        )

    def load(self, model: ModelSpec, device_id: int, precision: str = "fp16") -> None:
        """Load model weights and tokenizer onto specific target device.

        Args:
            model: Target model specification with immutable commit SHA.
            device_id: Concrete accelerator index (e.g. 0).
            precision: Floating point precision ('fp32', 'fp16', 'bf16').

        Raises:
            UnsupportedModelTypeError: If architecture is not a supported causal LM.
            UnsupportedPrecisionError: If precision is not in fp32, fp16, bf16.
            DeviceNotAvailableError: If target device is invalid or unavailable.
            ModelLoadError: If loading model weights or tokenizer fails.
        """
        # 1. Model architecture verification
        if not self._is_causal_lm_architecture(model.architecture):
            raise UnsupportedModelTypeError(
                f"Model architecture '{model.architecture or 'indeterminate'}' is not a supported causal language model.",
                details={"model_id": model.model_id, "architecture": model.architecture},
            )

        # 2. Precision verification
        prec_key = precision.lower().strip()
        if prec_key not in self.SUPPORTED_PRECISIONS:
            raise UnsupportedPrecisionError(
                f"Precision '{precision}' is not supported by baseline runner. Supported: {sorted(self.SUPPORTED_PRECISIONS)}",
                details={"requested_precision": precision, "supported": sorted(self.SUPPORTED_PRECISIONS)},
            )
        torch_dtype = _resolve_torch_dtype(prec_key)

        # 3. Device target determination
        if self._device_override is not None:
            device: Any = self._device_override
            if torch is not None and hasattr(torch, "device"):
                try:
                    device = torch.device(self._device_override)
                except Exception:
                    device = self._device_override
        else:
            if torch is None:
                raise DeviceNotAvailableError(
                    "PyTorch is not installed in the current environment.",
                    details={"runner": self.RUNNER_NAME},
                )
            if not torch.cuda.is_available():
                raise DeviceNotAvailableError(
                    f"Target AMD GPU device 'cuda:{device_id}' is not available (PyTorch ROCm/HIP backend not available).",
                    details={"device_id": device_id, "torch_version": getattr(torch, "__version__", "unknown")},
                )
            device_count = torch.cuda.device_count()
            if device_id < 0 or device_id >= device_count:
                raise DeviceNotAvailableError(
                    f"Device index {device_id} is invalid or out of range. Detected devices: {device_count}.",
                    details={"device_id": device_id, "available_devices": device_count},
                )
            device = torch.device(f"cuda:{device_id}")

        # 4. Resolve loaders (lazy import if not injected)
        if self._tokenizer_loader is None:
            try:
                from transformers import AutoTokenizer  # type: ignore[import-not-found,import-untyped]
                tok_loader = AutoTokenizer.from_pretrained
            except ImportError as exc:
                raise ModelLoadError(
                    f"Transformers library is not installed: {exc}",
                    details={"model_id": model.model_id, "cause": str(exc)},
                ) from exc
        else:
            tok_loader = self._tokenizer_loader

        if self._model_loader is None:
            try:
                from transformers import AutoModelForCausalLM  # type: ignore[import-not-found,import-untyped]
                mod_loader = AutoModelForCausalLM.from_pretrained
            except ImportError as exc:
                raise ModelLoadError(
                    f"Transformers library is not installed: {exc}",
                    details={"model_id": model.model_id, "cause": str(exc)},
                ) from exc
        else:
            mod_loader = self._model_loader

        # 5. Load tokenizer and model using immutable commit SHA and trust_remote_code=False
        try:
            tokenizer = tok_loader(
                model.model_id,
                revision=model.commit_sha,
                trust_remote_code=False,
            )
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load tokenizer for model '{model.model_id}' (revision: {model.commit_sha}): {exc}",
                details={"model_id": model.model_id, "revision": model.commit_sha, "cause": str(exc)},
            ) from exc

        try:
            # Explicit device placement: load weights and move to single concrete device without device_map="auto"
            load_kwargs: Dict[str, Any] = {
                "revision": model.commit_sha,
                "trust_remote_code": False,
            }
            if torch_dtype is not None:
                load_kwargs["torch_dtype"] = torch_dtype

            model_obj = mod_loader(model.model_id, **load_kwargs)
            if hasattr(model_obj, "to"):
                model_obj = model_obj.to(device)
            if hasattr(model_obj, "eval"):
                model_obj.eval()
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load weights for model '{model.model_id}' (revision: {model.commit_sha}) onto device '{device}': {exc}",
                details={
                    "model_id": model.model_id,
                    "revision": model.commit_sha,
                    "device": str(device),
                    "precision": prec_key,
                    "cause": str(exc),
                },
            ) from exc

        # 6. Update runner state
        self._loaded_model = model_obj
        self._loaded_tokenizer = tokenizer
        self._loaded_model_spec = model
        self._device_id = device_id
        self._target_device = device
        self._precision = prec_key
        self._is_loaded = True

    def generate(self, prompt: str, max_new_tokens: int = 16, **kwargs: Any) -> RunResult:
        """Execute deterministic greedy inference on the loaded model.

        Args:
            prompt: Input text prompt.
            max_new_tokens: Maximum number of new tokens to generate.
            **kwargs: Extra generation parameters.

        Returns:
            RunResult with factual token counts and generated output.

        Raises:
            RunnerNotReadyError: If called before load().
            GenerationError: If forward pass or decoding fails.
        """
        if (
            not self._is_loaded
            or self._loaded_model is None
            or self._loaded_tokenizer is None
            or self._loaded_model_spec is None
            or self._target_device is None
        ):
            raise RunnerNotReadyError(
                "Runner is not loaded. Call load() before executing generate().",
                details={"runner_name": self.RUNNER_NAME},
            )

        started_at = _utc_now_iso()

        try:
            # 1. Tokenize prompt and place tensors on target device
            inputs = self._loaded_tokenizer(prompt, return_tensors="pt")
            if isinstance(inputs, dict):
                inputs = {
                    k: (v.to(self._target_device) if hasattr(v, "to") else v)
                    for k, v in inputs.items()
                }
            input_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs
            input_tokens_count = int(input_ids.shape[-1])

            # 2. Deterministic greedy decoding
            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
            }
            # Remove conflicting sampling params if present
            generation_kwargs.update({k: v for k, v in kwargs.items() if k not in ("max_new_tokens", "do_sample")})

            inference_ctx = torch.inference_mode() if (torch is not None and hasattr(torch, "inference_mode")) else nullcontext()
            with inference_ctx:
                if isinstance(inputs, dict):
                    output_ids = self._loaded_model.generate(
                        **inputs,
                        **generation_kwargs,
                    )
                else:
                    output_ids = self._loaded_model.generate(
                        input_ids=input_ids,
                        **generation_kwargs,
                    )

            # 3. Factual token count derivation (generated tokens exclude input tokens)
            total_tokens = int(output_ids.shape[-1])
            generated_tokens_count = total_tokens - input_tokens_count

            # 4. Decode generated tokens slice
            new_tokens_slice = output_ids[0, input_tokens_count:]
            generated_text = self._loaded_tokenizer.decode(new_tokens_slice, skip_special_tokens=True)

            finished_at = _utc_now_iso()

            return RunResult(
                status=ExecutionStatus.SUCCESS,
                runtime_name=self.RUNNER_NAME,
                model_id=self._loaded_model_spec.model_id,
                model_revision=self._loaded_model_spec.commit_sha,
                device_id=self._device_id,
                precision=self._precision or "fp16",
                prompt=prompt,
                generated_text=generated_text,
                input_tokens=input_tokens_count,
                generated_tokens=generated_tokens_count,
                started_at_utc=started_at,
                finished_at_utc=finished_at,
                generation_params={"max_new_tokens": max_new_tokens, "do_sample": False},
            )

        except Exception as exc:
            raise GenerationError(
                f"Inference generation failed: {exc}",
                details={
                    "model_id": self._loaded_model_spec.model_id,
                    "device": str(self._target_device),
                    "prompt_length": len(prompt),
                    "cause": str(exc),
                },
            ) from exc

    def unload(self) -> None:
        """Release device and host memory, clean references, and collect garbage."""
        self._loaded_model = None
        self._loaded_tokenizer = None
        self._loaded_model_spec = None
        self._device_id = None
        self._target_device = None
        self._precision = None
        self._is_loaded = False

        # Force Python garbage collection
        gc.collect()

        # Clear accelerator device cache if available
        if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
