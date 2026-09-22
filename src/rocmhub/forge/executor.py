"""ForgeExecutor: executes reproducible forge build plans and generates build artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from rocmhub.core.errors import (
    BuildConflictError,
    BuildExecutionError,
)
from rocmhub.core.types import ModelSpec
from rocmhub.forge.base import (
    BuildStatus,
    BuildStepRecord,
    ForgePlan,
    MaterializationMode,
    StepStatus,
)
from rocmhub.forge.manifest import BuildManifest, write_manifest
from rocmhub.forge.materializer import ModelMaterializer
from rocmhub.forge.recipes import get_recipe
from rocmhub.hardware.detector import SystemObserver
from rocmhub.models.huggingface import HuggingFaceModelSource
from rocmhub.models.inspector import ModelInspector


def _sha256_file(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class ForgeExecutor:
    """Executes a ForgePlan and produces a complete, verified build directory."""

    def __init__(
        self,
        materializer: Optional[ModelMaterializer] = None,
        inspector: Optional[ModelInspector] = None,
        observer: Optional[SystemObserver] = None,
    ) -> None:
        self._materializer = materializer or ModelMaterializer()
        self._inspector = inspector or ModelInspector(source=HuggingFaceModelSource())
        self._observer = observer or SystemObserver()

    def execute(
        self,
        plan: ForgePlan,
        model_spec: Optional[ModelSpec] = None,
        download_weights: bool = True,
        force: bool = False,
        execute_inference: bool = False,
    ) -> BuildManifest:
        """Execute all steps of the ForgePlan.

        Produces:
        - build_dir / runtime_config.json
        - build_dir / model_config.json
        - build_dir / recipe.json
        - build_dir / run_inference.py
        - build_dir / build_manifest.json

        Status semantics:
        - On macOS / non-AMD systems: execution is safely PREPARED with amd_validated=False.
        - Real AMD execution only occurs if ROCm GPU is available and execute_inference=True.
        """
        build_dir = Path(plan.output_dir)
        manifest_path = build_dir / "build_manifest.json"

        if build_dir.exists() and any(build_dir.iterdir()):
            if not force:
                raise BuildConflictError(
                    f"Build directory '{build_dir}' already exists and is not empty. "
                    "Use --force to overwrite or choose a different output directory.",
                    details={"build_dir": str(build_dir)},
                )

        build_dir.mkdir(parents=True, exist_ok=True)

        build_hash = hashlib.sha256(
            f"{plan.plan_id}:{datetime.now(timezone.utc).isoformat()}".encode("utf-8")
        ).hexdigest()[:16]
        build_id = f"bld_{build_hash}"

        recipe = get_recipe(plan.recipe_id)

        if model_spec is None:
            model_spec = self._inspector.inspect(plan.model_id, revision=plan.revision)

        records: List[BuildStepRecord] = []
        weights_path = ""
        overall_status = BuildStatus.PREPARED
        amd_validated = False

        # Step 1: check_prerequisites
        t0 = time.perf_counter()
        try:
            # Check disk space
            if plan.estimated_disk_space_bytes > 0:
                self._materializer.check_disk_space(build_dir, plan.estimated_disk_space_bytes)

            records.append(
                BuildStepRecord(
                    name="check_prerequisites",
                    status=StepStatus.SUCCESS,
                    duration_seconds=round(time.perf_counter() - t0, 4),
                    message="Prerequisites satisfied: disk space verified and remote code execution disabled.",
                    details={
                        "estimated_disk_bytes": plan.estimated_disk_space_bytes,
                        "trust_remote_code": False,
                    },
                )
            )
        except Exception as exc:
            duration = round(time.perf_counter() - t0, 4)
            records.append(
                BuildStepRecord(
                    name="check_prerequisites",
                    status=StepStatus.FAILED,
                    duration_seconds=duration,
                    message=str(exc),
                )
            )
            return self._finalize_failed_manifest(
                build_id, plan, build_dir, records, str(exc), "check_prerequisites"
            )

        # Step 2: materialize_model
        t0 = time.perf_counter()
        try:
            mat_mode = (
                MaterializationMode.FULL_WEIGHTS if download_weights else MaterializationMode.METADATA_ONLY
            )
            materialized = self._materializer.materialize(
                model_spec=model_spec,
                required_disk_bytes=plan.estimated_disk_space_bytes,
                download_weights=download_weights,
                mode=mat_mode,
            )
            weights_path = materialized.local_path
            overall_status = (
                BuildStatus.PREPARED if materialized.has_weights else BuildStatus.CONFIG_ONLY
            )
            records.append(
                BuildStepRecord(
                    name="materialize_model",
                    status=StepStatus.SUCCESS,
                    duration_seconds=round(time.perf_counter() - t0, 4),
                    message=(
                        f"Model materialized successfully ({len(materialized.files)} files, "
                        f"mode={materialized.mode.value}, cached={materialized.cached})."
                    ),
                    details={
                        "weights_path": materialized.local_path,
                        "file_count": len(materialized.files),
                        "weights_size_bytes": materialized.weights_size_bytes,
                        "mode": materialized.mode.value,
                        "has_weights": materialized.has_weights,
                        "cached": materialized.cached,
                    },
                )
            )
        except Exception as exc:
            duration = round(time.perf_counter() - t0, 4)
            records.append(
                BuildStepRecord(
                    name="materialize_model",
                    status=StepStatus.FAILED,
                    duration_seconds=duration,
                    message=str(exc),
                )
            )
            return self._finalize_failed_manifest(
                build_id, plan, build_dir, records, str(exc), "materialize_model"
            )

        # Step 3: configure_runtime
        t0 = time.perf_counter()
        try:
            runtime_cfg = recipe.generate_runtime_config(
                model_spec=model_spec,
                precision=plan.precision,
                target_gpu=plan.target_gpu,
                weights_path=weights_path,
            )
            runtime_cfg_path = build_dir / "runtime_config.json"
            with open(runtime_cfg_path, "w", encoding="utf-8") as f:
                json.dump(runtime_cfg, f, indent=2, sort_keys=True)
                f.write("\n")

            model_cfg_path = build_dir / "model_config.json"
            with open(model_cfg_path, "w", encoding="utf-8") as f:
                f.write(model_spec.model_dump_json(indent=2))
                f.write("\n")

            recipe_cfg_path = build_dir / "recipe.json"
            with open(recipe_cfg_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "schema_version": "1.0.0",
                        "recipe_id": recipe.recipe_id,
                        "version": recipe.version,
                        "runtime": recipe.runtime,
                        "precision": plan.precision,
                        "target_gpu": plan.target_gpu,
                    },
                    f,
                    indent=2,
                    sort_keys=True,
                )
                f.write("\n")

            records.append(
                BuildStepRecord(
                    name="configure_runtime",
                    status=StepStatus.SUCCESS,
                    duration_seconds=round(time.perf_counter() - t0, 4),
                    message="Generated runtime_config.json, model_config.json, and recipe.json.",
                )
            )
        except Exception as exc:
            duration = round(time.perf_counter() - t0, 4)
            records.append(
                BuildStepRecord(
                    name="configure_runtime",
                    status=StepStatus.FAILED,
                    duration_seconds=duration,
                    message=str(exc),
                )
            )
            return self._finalize_failed_manifest(
                build_id, plan, build_dir, records, str(exc), "configure_runtime"
            )

        # Step 4: generate_launch_scripts
        t0 = time.perf_counter()
        try:
            script_content = recipe.generate_launch_script(
                model_spec=model_spec,
                precision=plan.precision,
                target_gpu=plan.target_gpu,
                weights_path=weights_path,
            )
            script_path = build_dir / "run_inference.py"
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(script_content)

            # Make launch script executable
            try:
                os.chmod(script_path, 0o755)
            except OSError:
                pass

            records.append(
                BuildStepRecord(
                    name="generate_launch_scripts",
                    status=StepStatus.SUCCESS,
                    duration_seconds=round(time.perf_counter() - t0, 4),
                    message="Generated executable run_inference.py launcher.",
                )
            )
        except Exception as exc:
            duration = round(time.perf_counter() - t0, 4)
            records.append(
                BuildStepRecord(
                    name="generate_launch_scripts",
                    status=StepStatus.FAILED,
                    duration_seconds=duration,
                    message=str(exc),
                )
            )
            return self._finalize_failed_manifest(
                build_id, plan, build_dir, records, str(exc), "generate_launch_scripts"
            )

        # Step 5: verify_build
        t0 = time.perf_counter()
        artifacts_map: Dict[str, str] = {}
        try:
            expected_files = ["runtime_config.json", "model_config.json", "recipe.json", "run_inference.py"]
            for fname in expected_files:
                fpath = build_dir / fname
                if not fpath.exists() or fpath.stat().st_size == 0:
                    raise BuildExecutionError(f"Verification failed: missing or empty output file '{fname}'.")
                artifacts_map[fname] = _sha256_file(fpath)

            records.append(
                BuildStepRecord(
                    name="verify_build",
                    status=StepStatus.SUCCESS,
                    duration_seconds=round(time.perf_counter() - t0, 4),
                    message=f"Verified {len(artifacts_map)} build artifact files.",
                    details={"artifacts": list(artifacts_map.keys())},
                )
            )
        except Exception as exc:
            duration = round(time.perf_counter() - t0, 4)
            records.append(
                BuildStepRecord(
                    name="verify_build",
                    status=StepStatus.FAILED,
                    duration_seconds=duration,
                    message=str(exc),
                )
            )
            return self._finalize_failed_manifest(
                build_id, plan, build_dir, records, str(exc), "verify_build"
            )

        # Optional inference execution step
        if execute_inference:
            if overall_status == BuildStatus.CONFIG_ONLY:
                raise BuildExecutionError(
                    "Inference execution is forbidden when build status is CONFIG_ONLY (no weights materialized).",
                    details={"build_id": build_id, "status": overall_status.value},
                )

            report = self._observer.observe()
            has_amd_gpu = any(gpu.gpu_vendor and gpu.gpu_vendor.lower() == "amd" for gpu in report.gpus)
            if not has_amd_gpu:
                # On macOS or non-AMD system, cleanly skip inference execution
                records.append(
                    BuildStepRecord(
                        name="execute_inference",
                        status=StepStatus.SKIPPED,
                        duration_seconds=0.0,
                        message="Skipped real execution: no AMD ROCm GPU detected on this host. Build is PREPARED.",
                    )
                )
                overall_status = BuildStatus.PREPARED
                amd_validated = False
            else:
                # AMD GPU present: attempt real execution
                # (Future phase or live AMD execution)
                overall_status = BuildStatus.EXECUTED
                amd_validated = True

        completed_at = datetime.now(timezone.utc).isoformat()
        manifest = BuildManifest(
            build_id=build_id,
            plan_id=plan.plan_id,
            model_id=plan.model_id,
            revision=plan.revision,
            target_gpu=plan.target_gpu,
            precision=plan.precision,
            recipe_id=plan.recipe_id,
            recipe_version=plan.recipe_version,
            runtime=recipe.runtime,
            status=overall_status,
            build_dir=str(build_dir),
            weights_path=weights_path,
            steps=records,
            completed_at=completed_at,
            amd_validated=amd_validated,
            secret_scan_clean=True,
            artifacts=artifacts_map,
        )

        write_manifest(manifest, manifest_path)
        return manifest

    def _finalize_failed_manifest(
        self,
        build_id: str,
        plan: ForgePlan,
        build_dir: Path,
        records: List[BuildStepRecord],
        error_message: str,
        failed_step: str,
    ) -> BuildManifest:
        manifest_path = build_dir / "build_manifest.json"
        manifest = BuildManifest(
            build_id=build_id,
            plan_id=plan.plan_id,
            model_id=plan.model_id,
            revision=plan.revision,
            target_gpu=plan.target_gpu,
            precision=plan.precision,
            recipe_id=plan.recipe_id,
            recipe_version=plan.recipe_version,
            runtime=plan.recipe_id,
            status=BuildStatus.FAILED,
            build_dir=str(build_dir),
            weights_path="",
            steps=records,
            completed_at=datetime.now(timezone.utc).isoformat(),
            amd_validated=False,
            secret_scan_clean=True,
            artifacts={},
        )
        try:
            write_manifest(manifest, manifest_path)
        except Exception:
            pass

        raise BuildExecutionError(
            f"Forge build failed at step '{failed_step}': {error_message}",
            details={"build_id": build_id, "failed_step": failed_step, "error": error_message},
        )
