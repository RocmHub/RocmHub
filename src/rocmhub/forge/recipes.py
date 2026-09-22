"""Build recipes for ROCmHub Model Forge."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from rocmhub.core.errors import (
    UnsupportedModelArchitectureError,
    UnsupportedPrecisionError,
)
from rocmhub.core.types import ModelSpec
from rocmhub.forge.base import BuildStepSpec

SUPPORTED_PRECISIONS = ["fp16", "bf16", "fp32"]

STANDARD_CAUSAL_LM_ARCHITECTURES = {
    "qwen2forcausallm",
    "llamaforcausallm",
    "mistralforcausallm",
    "gemmaforcausallm",
    "gemma2forcausallm",
    "phiforcausallm",
    "phi3forcausallm",
    "falconforcausallm",
    "bloomforcausallm",
    "gpt2lmheadmodel",
    "gptneoforcausallm",
    "gptneoxforcausallm",
    "optforcausallm",
}


@runtime_checkable
class ForgeRecipe(Protocol):
    """Protocol for model build recipes."""

    @property
    def recipe_id(self) -> str:
        """Unique identifier of the recipe."""
        ...

    @property
    def version(self) -> str:
        """Recipe version."""
        ...

    @property
    def runtime(self) -> str:
        """Target runtime identifier."""
        ...

    def matches(self, model_spec: ModelSpec) -> bool:
        """Return True if this recipe is applicable to the given model."""
        ...

    def validate(self, model_spec: ModelSpec, precision: str) -> None:
        """Validate that the model architecture and precision are supported."""
        ...

    def get_steps(self) -> List[BuildStepSpec]:
        """Return the ordered build steps defined by this recipe."""
        ...

    def generate_runtime_config(
        self,
        model_spec: ModelSpec,
        precision: str,
        target_gpu: Optional[str] = None,
        weights_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate runtime configuration dictionary."""
        ...

    def generate_launch_script(
        self,
        model_spec: ModelSpec,
        precision: str,
        target_gpu: Optional[str] = None,
        weights_path: Optional[str] = None,
    ) -> str:
        """Generate standalone executable run_inference.py content."""
        ...


class PyTorchTransformersHipRecipe:
    """Recipe for causal language models running on PyTorch + Transformers + ROCm/HIP."""

    recipe_id: str = "pytorch_transformers_hip"
    version: str = "1.0.0"
    runtime: str = "pytorch_transformers_hip"

    def is_causal_lm(self, model_spec: ModelSpec) -> bool:
        """Check whether the model architecture represents a causal language model."""
        if not model_spec.architecture:
            return False
        arch = model_spec.architecture.strip().lower()
        if arch in STANDARD_CAUSAL_LM_ARCHITECTURES:
            return True
        if arch.endswith("forcausallm") or arch.endswith("lmheadmodel"):
            return True
        return False

    def matches(self, model_spec: ModelSpec) -> bool:
        """Check if this recipe can build the given model."""
        return self.is_causal_lm(model_spec)

    def validate(self, model_spec: ModelSpec, precision: str) -> None:
        """Validate model architecture and precision."""
        if precision.lower() not in SUPPORTED_PRECISIONS:
            raise UnsupportedPrecisionError(
                f"Precision '{precision}' is not supported by recipe '{self.recipe_id}'. "
                f"Supported precisions: {SUPPORTED_PRECISIONS}",
                details={"precision": precision, "supported": SUPPORTED_PRECISIONS},
            )

        if not self.matches(model_spec):
            raise UnsupportedModelArchitectureError(
                f"Model architecture '{model_spec.architecture}' is not supported by recipe '{self.recipe_id}'. "
                "Only causal language models (text-generation) are currently supported.",
                details={
                    "model_id": model_spec.model_id,
                    "architecture": model_spec.architecture,
                },
            )

    def get_steps(self) -> List[BuildStepSpec]:
        """Return the standard sequence of build steps."""
        return [
            BuildStepSpec(
                name="check_prerequisites",
                description="Verify environment capabilities, disk space, and remote code safety.",
                required=True,
            ),
            BuildStepSpec(
                name="materialize_model",
                description="Acquire and cache model configuration and weights at immutable commit SHA.",
                required=True,
            ),
            BuildStepSpec(
                name="configure_runtime",
                description="Synthesize runtime configuration and target execution parameters.",
                required=True,
            ),
            BuildStepSpec(
                name="generate_launch_scripts",
                description="Generate reproducible standalone launch scripts (run_inference.py).",
                required=True,
            ),
            BuildStepSpec(
                name="verify_build",
                description="Validate build outputs, checksums, and manifest completeness.",
                required=True,
            ),
        ]

    def _precision_to_dtype_str(self, precision: str) -> str:
        p = precision.lower()
        if p == "fp16":
            return "float16"
        if p == "bf16":
            return "bfloat16"
        if p == "fp32":
            return "float32"
        return p

    def generate_runtime_config(
        self,
        model_spec: ModelSpec,
        precision: str,
        target_gpu: Optional[str] = None,
        weights_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate runtime configuration."""
        dtype_str = self._precision_to_dtype_str(precision)
        return {
            "schema_version": "1.0.0",
            "runtime": self.runtime,
            "recipe_id": self.recipe_id,
            "recipe_version": self.version,
            "model_id": model_spec.model_id,
            "revision": model_spec.commit_sha,
            "precision": precision.lower(),
            "torch_dtype": dtype_str,
            "target_gpu": target_gpu,
            "device": "cuda",
            "trust_remote_code": False,
            "weights_path": weights_path,
            "inference_parameters": {
                "max_new_tokens": 32,
                "temperature": 0.0,
                "top_p": 1.0,
                "do_sample": False,
            },
        }

    def generate_launch_script(
        self,
        model_spec: ModelSpec,
        precision: str,
        target_gpu: Optional[str] = None,
        weights_path: Optional[str] = None,
    ) -> str:
        """Generate standalone run_inference.py."""
        dtype_str = self._precision_to_dtype_str(precision)
        model_id = model_spec.model_id
        revision = model_spec.commit_sha

        return f'''#!/usr/bin/env python3
"""ROCmHub Generated Standalone Baseline Inference Script.

Model: {model_id}
Revision: {revision}
Recipe: {self.recipe_id} ({self.version})
Precision: {precision}
Target GPU: {target_gpu or "Auto-detected"}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run baseline inference on AMD ROCm.")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain the significance of open-source AI acceleration in one sentence.",
        help="Input text prompt for inference.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Maximum new tokens to generate.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on ('cuda' maps to HIP on ROCm).",
    )
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        sys.stderr.write(f"Missing dependencies: {{exc}}\\n")
        sys.stderr.write("Install with: pip install torch transformers\\n")
        return 1

    device_str = args.device
    if device_str == "cuda" and not torch.cuda.is_available():
        sys.stderr.write("WARNING: CUDA/HIP not available. Falling back to CPU.\\n")
        device_str = "cpu"

    model_location = r"""{weights_path or model_id}"""
    print(f"Loading model: {model_id} (revision: {revision[:8] if len('{revision}') >= 8 else '{revision}'})")
    print(f"Device: {{device_str}}, Precision: {precision}")

    dtype_map = {{
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }}
    torch_dtype = dtype_map.get("{dtype_str}", torch.float32)
    if device_str == "cpu" and torch_dtype == torch.float16:
        print("Note: CPU inference with float16 is typically unsupported or slow; using float32 on CPU.")
        torch_dtype = torch.float32

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        model_location,
        revision="{revision}",
        trust_remote_code=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_location,
        revision="{revision}",
        torch_dtype=torch_dtype,
        trust_remote_code=False,
    )
    model.to(device_str)
    model.eval()
    load_time = time.perf_counter() - t0
    print(f"Model loaded successfully in {{load_time:.2f}}s.")

    inputs = tokenizer(args.prompt, return_tensors="pt").to(device_str)
    print(f"Input prompt: {{args.prompt!r}}")
    print("Generating...")

    t1 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    gen_time = time.perf_counter() - t1

    new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
    output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    tokens_generated = len(new_tokens)
    tps = tokens_generated / gen_time if gen_time > 0 else 0.0

    print(f"Generated text: {{output_text!r}}")
    print(f"Tokens: {{tokens_generated}}, Generation time: {{gen_time:.3f}}s ({{tps:.1f}} tok/s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


_RECIPE_REGISTRY: Dict[str, ForgeRecipe] = {
    PyTorchTransformersHipRecipe.recipe_id: PyTorchTransformersHipRecipe(),
}


def get_recipe(recipe_id: str) -> ForgeRecipe:
    """Look up a recipe by identifier."""
    if recipe_id not in _RECIPE_REGISTRY:
        raise UnsupportedModelArchitectureError(
            f"No build recipe found with ID '{recipe_id}'. "
            f"Available recipes: {list(_RECIPE_REGISTRY.keys())}",
            details={"recipe_id": recipe_id, "available": list(_RECIPE_REGISTRY.keys())},
        )
    return _RECIPE_REGISTRY[recipe_id]


def find_recipe_for_model(model_spec: ModelSpec) -> ForgeRecipe:
    """Find a compatible recipe for the given model spec."""
    for recipe in _RECIPE_REGISTRY.values():
        if recipe.matches(model_spec):
            return recipe

    raise UnsupportedModelArchitectureError(
        f"No compatible build recipe found for model '{model_spec.model_id}' "
        f"with architecture: '{model_spec.architecture}'",
        details={
            "model_id": model_spec.model_id,
            "architecture": model_spec.architecture,
        },
    )
