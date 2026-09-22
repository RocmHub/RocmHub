"""ForgePlanner: generates deterministic build plans for model preparation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional, Union

from rocmhub.capabilities.evaluator import CapabilityEvaluator
from rocmhub.core.types import EvaluationVerdict, ModelSpec
from rocmhub.forge.base import ForgePlan
from rocmhub.forge.recipes import find_recipe_for_model, get_recipe
from rocmhub.hardware.detector import SystemObserver
from rocmhub.models.huggingface import HuggingFaceModelSource
from rocmhub.models.inspector import ModelInspector


class ForgePlanner:
    """Generates reproducible, deterministic build plans without downloading weights."""

    def __init__(
        self,
        model_inspector: Optional[ModelInspector] = None,
        system_observer: Optional[SystemObserver] = None,
        capability_evaluator: Optional[CapabilityEvaluator] = None,
    ) -> None:
        self._inspector = model_inspector or ModelInspector(source=HuggingFaceModelSource())
        self._observer = system_observer or SystemObserver()
        self._evaluator = capability_evaluator or CapabilityEvaluator()

    def estimate_required_disk_bytes(self, model_spec: ModelSpec, precision: str) -> int:
        """Estimate the required disk space in bytes for materialization and configs."""
        # 4 bytes for fp32, 2 bytes for fp16 / bf16
        bytes_per_param = 4 if precision.lower() == "fp32" else 2
        if model_spec.parameter_count and model_spec.parameter_count > 0:
            raw_weights = model_spec.parameter_count * bytes_per_param
            # 15% safety margin for tokenizer, config files, and build assets
            return int(raw_weights * 1.15)
        # Conservative fallback if parameter_count is missing (2 GB)
        return 2 * 1024 * 1024 * 1024

    def create_plan(
        self,
        model_id: str,
        revision: Optional[str] = None,
        precision: str = "fp16",
        target_gpu: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
        recipe_id: Optional[str] = None,
        model_spec: Optional[ModelSpec] = None,
    ) -> ForgePlan:
        """Create a deterministic ForgePlan for the requested model and configuration.

        This method inspects model metadata via HuggingFace Hub API to resolve
        the immutable commit SHA, but NEVER downloads model weights.
        """
        # 1. Resolve model metadata and immutable commit SHA
        if model_spec is None:
            model_spec = self._inspector.inspect(model_id=model_id, revision=revision or "main")

        # 2. Select and validate build recipe
        if recipe_id is not None:
            recipe = get_recipe(recipe_id)
        else:
            recipe = find_recipe_for_model(model_spec)

        recipe.validate(model_spec=model_spec, precision=precision)

        # 3. Target GPU detection and capability check
        detection_report = self._observer.observe()
        detected_target_gpu = target_gpu

        if detected_target_gpu is None:
            # Check if any AMD GPU is present in detection report
            for gpu in detection_report.gpus:
                if gpu.gpu_vendor and gpu.gpu_vendor.lower() == "amd":
                    detected_target_gpu = gpu.gfx_target or gpu.device_name
                    break

        cap_report = self._evaluator.evaluate(model_spec, detection_report)
        compatibility_confirmed = cap_report.verdict == EvaluationVerdict.READY

        # 4. Estimate required disk space
        estimated_disk_bytes = self.estimate_required_disk_bytes(model_spec, precision)

        # 5. Determine canonical output directory
        safe_model_slug = model_spec.model_id.replace("/", "--")
        short_sha = model_spec.commit_sha[:8]
        if output_dir is not None:
            canonical_output_dir = str(Path(output_dir).resolve())
        else:
            canonical_output_dir = str((Path("./builds") / f"{safe_model_slug}--{short_sha}").resolve())

        # 6. Generate deterministic plan ID
        hash_seed = f"{model_spec.model_id}:{model_spec.commit_sha}:{recipe.recipe_id}:{precision.lower()}:{detected_target_gpu or 'none'}"
        plan_hash = hashlib.sha256(hash_seed.encode("utf-8")).hexdigest()[:16]
        plan_id = f"plan_{plan_hash}"

        # 7. Get ordered steps from recipe
        steps = recipe.get_steps()

        return ForgePlan(
            plan_id=plan_id,
            model_id=model_spec.model_id,
            revision=model_spec.commit_sha,
            target_gpu=detected_target_gpu,
            precision=precision.lower(),
            recipe_id=recipe.recipe_id,
            recipe_version=recipe.version,
            output_dir=canonical_output_dir,
            estimated_disk_space_bytes=estimated_disk_bytes,
            steps=steps,
            compatibility_confirmed=compatibility_confirmed,
        )
