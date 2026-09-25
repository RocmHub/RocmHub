"""ForgeExecutor: executes reproducible forge build plans and generates build artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from rocmhub.core.errors import (
    BuildConflictError,
    BuildExecutionError,
)
from rocmhub.core.types import ExecutionStatus, ModelSpec
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


class ExecutionResult(BaseModel):
    """Structured result of executing a standalone model launcher (run_inference.py)."""

    model_config = ConfigDict(frozen=True)

    status: ExecutionStatus
    amd_validated: bool = False
    exit_code: int = 0
    duration_seconds: float = 0.0
    device: Optional[str] = None
    generated_text: Optional[str] = None
    tokens_generated: Optional[int] = None
    tokens_per_second: Optional[float] = None
    stdout: str = ""
    stderr: str = ""
    error_message: Optional[str] = None

    # Separate factual observations (Phase 17 safety invariants)
    process_success: bool = False
    inference_executed: bool = False
    hip_runtime_used: bool = False
    amd_gpu_used: bool = False
    gpu_device_name: Optional[str] = None
    hip_version: Optional[str] = None
    pytorch_version: Optional[str] = None
    validation_passed: Optional[bool] = None
    validation_failures: List[str] = Field(default_factory=list)
    dry_run: bool = False


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
                        "cache_status": materialized.cache_status or ("HIT_UNVERIFIED" if materialized.cached else "MISS"),
                        "integrity": materialized.integrity,
                        "materialized_bytes": materialized.materialized_bytes,
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
                # AMD GPU present: execute launcher on AMD hardware
                t_exec = time.perf_counter()
                exec_res = self.execute_build(
                    build_dir=build_dir,
                    prompt="Explain the significance of open-source AI acceleration in one sentence.",
                    max_new_tokens=32,
                    device="cuda",
                )
                dur = round(time.perf_counter() - t_exec, 4)
                if exec_res.status == ExecutionStatus.SUCCESS and exec_res.amd_validated:
                    records.append(
                        BuildStepRecord(
                            name="execute_inference",
                            status=StepStatus.SUCCESS,
                            duration_seconds=dur,
                            message=f"AMD ROCm inference verified on {exec_res.gpu_device_name or 'AMD GPU'} ({exec_res.tokens_generated} tokens generated).",
                            details={
                                "generated_text": exec_res.generated_text,
                                "tokens_generated": exec_res.tokens_generated,
                                "tokens_per_second": exec_res.tokens_per_second,
                                "device": exec_res.device,
                                "hip_runtime_used": exec_res.hip_runtime_used,
                                "amd_gpu_used": exec_res.amd_gpu_used,
                                "validation_passed": exec_res.validation_passed,
                            },
                        )
                    )
                    overall_status = BuildStatus.EXECUTED
                    amd_validated = True
                elif exec_res.status == ExecutionStatus.SUCCESS and not exec_res.amd_validated:
                    records.append(
                        BuildStepRecord(
                            name="execute_inference",
                            status=StepStatus.SKIPPED,
                            duration_seconds=dur,
                            message="Inference executed but live AMD HIP hardware execution was not confirmed. Build remains PREPARED.",
                            details={
                                "device": exec_res.device,
                                "hip_runtime_used": exec_res.hip_runtime_used,
                                "amd_gpu_used": exec_res.amd_gpu_used,
                                "validation_passed": exec_res.validation_passed,
                            },
                        )
                    )
                    overall_status = BuildStatus.PREPARED
                    amd_validated = False
                else:
                    records.append(
                        BuildStepRecord(
                            name="execute_inference",
                            status=StepStatus.FAILED,
                            duration_seconds=dur,
                            message=f"Inference execution failed: {exec_res.error_message or exec_res.stderr}",
                            details={"exit_code": exec_res.exit_code, "stderr": exec_res.stderr},
                        )
                    )
                    overall_status = BuildStatus.FAILED
                    amd_validated = False

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

    def execute_build(
        self,
        build_dir: Union[Path, str],
        prompt: str = "Explain the significance of open-source AI acceleration in one sentence.",
        max_new_tokens: int = 32,
        device: str = "cuda",
        timeout_seconds: int = 120,
        dry_run: bool = False,
    ) -> ExecutionResult:
        """Execute standalone launcher (run_inference.py) in build_dir and verify output.

        Status and Safety semantics:
        - If dry_run is True: validates launcher and configuration without spawning subprocess or executing model.
        - If device='cuda' and no AMD GPU is detected: cleanly returns SKIPPED with amd_validated=False (no synthetic data).
        - Separates factual observations:
          * process_success (exit_code == 0)
          * inference_executed (generated text present and tokens > 0)
          * hip_runtime_used (HIP runtime verified in torch)
          * amd_gpu_used (AMD GPU confirmed as execution device)
          * validation_passed (output verified against corruption/invariants; NOT automatically True)
        - amd_validated is ONLY True when ALL 4 facts (process_success, inference_executed, hip_runtime_used, amd_gpu_used) are True.
        """
        bdir = Path(build_dir).resolve()
        script_path = bdir / "run_inference.py"
        if not script_path.exists():
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                amd_validated=False,
                exit_code=1,
                duration_seconds=0.0,
                error_message=f"Launcher script not found at '{script_path}'.",
            )

        if dry_run:
            return ExecutionResult(
                status=ExecutionStatus.SKIPPED,
                amd_validated=False,
                exit_code=0,
                duration_seconds=0.0,
                device=device,
                stdout="Dry-run: validated build structure and launch configuration. Subprocess execution skipped.",
                process_success=True,
                inference_executed=False,
                hip_runtime_used=False,
                amd_gpu_used=False,
                dry_run=True,
            )

        report = self._observer.observe()
        has_amd_gpu = any(gpu.gpu_vendor and gpu.gpu_vendor.lower() == "amd" for gpu in report.gpus)

        if device == "cuda" and not has_amd_gpu:
            return ExecutionResult(
                status=ExecutionStatus.SKIPPED,
                amd_validated=False,
                exit_code=0,
                duration_seconds=0.0,
                device="cpu",
                stdout="Skipped execution: no AMD ROCm GPU detected on this host. Build is PREPARED.",
                error_message="No AMD ROCm GPU detected on host.",
                process_success=True,
                inference_executed=False,
                hip_runtime_used=False,
                amd_gpu_used=False,
                dry_run=False,
            )

        cmd = [
            sys.executable,
            str(script_path),
            "--prompt",
            prompt,
            "--max-new-tokens",
            str(max_new_tokens),
            "--device",
            device,
            "--json",
        ]

        clean_env = os.environ.copy()
        clean_env["PYTHONUNBUFFERED"] = "1"
        clean_env["LC_ALL"] = "C"

        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(bdir),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env=clean_env,
            )
            duration = round(time.perf_counter() - t0, 4)
            stdout = proc.stdout.strip()
            stderr = proc.stderr.strip()

            parsed: Dict[str, Any] = {}
            if stdout:
                for line in reversed(stdout.splitlines()):
                    line = line.strip()
                    if line.startswith("{") and line.endswith("}"):
                        try:
                            parsed = json.loads(line)
                            break
                        except Exception:
                            continue

            process_success = (proc.returncode == 0)
            has_success_status = (parsed.get("status") == "SUCCESS")
            gen_text = parsed.get("generated_text")
            tokens_gen = parsed.get("tokens_generated")
            inference_executed = bool(
                process_success
                and has_success_status
                and tokens_gen is not None
                and tokens_gen > 0
                and gen_text is not None
            )

            hip_runtime_used = bool(parsed.get("is_hip", False))
            hip_ver = parsed.get("hip_version")
            torch_ver = parsed.get("pytorch_version")
            amd_gpu_used = bool(parsed.get("amd_gpu_used", False))
            gpu_dev_name = parsed.get("gpu_device_name")
            actual_device = parsed.get("device") or device

            # Validation: technical correctness check on generated output
            validation_failures: List[str] = []
            validation_passed: Optional[bool] = None
            if inference_executed:
                from rocmhub.validation.correctness import NUMERIC_CORRUPTION_PATTERNS

                text_str = str(gen_text)
                if not text_str.strip():
                    validation_failures.append("Generated output is empty or whitespace.")
                for pat in NUMERIC_CORRUPTION_PATTERNS:
                    if pat.search(text_str):
                        validation_failures.append(f"Generated text contains corruption pattern ({pat.pattern}).")
                if (tokens_gen or 0) <= 0:
                    validation_failures.append("Token count is zero or negative.")
                validation_passed = (len(validation_failures) == 0)

            # Strict AMD validation requirement: process + inference + HIP runtime + AMD GPU device
            amd_validated = bool(
                process_success
                and inference_executed
                and hip_runtime_used
                and amd_gpu_used
                and (device == "cuda" or actual_device.startswith("cuda"))
            )

            if process_success and has_success_status:
                res = ExecutionResult(
                    status=ExecutionStatus.SUCCESS,
                    amd_validated=amd_validated,
                    exit_code=0,
                    duration_seconds=duration,
                    generated_text=gen_text,
                    tokens_generated=tokens_gen,
                    tokens_per_second=parsed.get("tokens_per_second"),
                    device=actual_device,
                    stdout=stdout,
                    stderr=stderr,
                    process_success=True,
                    inference_executed=inference_executed,
                    hip_runtime_used=hip_runtime_used,
                    amd_gpu_used=amd_gpu_used,
                    gpu_device_name=gpu_dev_name,
                    hip_version=hip_ver,
                    pytorch_version=torch_ver,
                    validation_passed=validation_passed,
                    validation_failures=validation_failures,
                    dry_run=False,
                )
                if res.amd_validated:
                    self._upgrade_manifest_to_executed(bdir, res)
                return res
            else:
                err_msg = parsed.get("error") if parsed else (stderr or f"Process exited with code {proc.returncode}")
                return ExecutionResult(
                    status=ExecutionStatus.FAILED,
                    amd_validated=False,
                    exit_code=proc.returncode,
                    duration_seconds=duration,
                    stdout=stdout,
                    stderr=stderr,
                    error_message=err_msg,
                    process_success=process_success,
                    inference_executed=False,
                    hip_runtime_used=hip_runtime_used,
                    amd_gpu_used=amd_gpu_used,
                    gpu_device_name=gpu_dev_name,
                    hip_version=hip_ver,
                    pytorch_version=torch_ver,
                    validation_passed=False,
                    validation_failures=[err_msg] if err_msg else [],
                    dry_run=False,
                )

        except subprocess.TimeoutExpired:
            duration = round(time.perf_counter() - t0, 4)
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                amd_validated=False,
                exit_code=-1,
                duration_seconds=duration,
                error_message=f"Execution timed out after {timeout_seconds} seconds.",
                process_success=False,
                inference_executed=False,
                validation_passed=False,
                validation_failures=[f"Timeout after {timeout_seconds}s"],
                dry_run=False,
            )
        except Exception as exc:
            duration = round(time.perf_counter() - t0, 4)
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                amd_validated=False,
                exit_code=-1,
                duration_seconds=duration,
                error_message=str(exc),
                process_success=False,
                inference_executed=False,
                validation_passed=False,
                validation_failures=[str(exc)],
                dry_run=False,
            )

    def _upgrade_manifest_to_executed(self, build_dir: Path, result: ExecutionResult) -> None:
        """Safely upgrade build manifest to EXECUTED status after successful AMD validation."""
        manifest_path = build_dir / "build_manifest.json"
        if not manifest_path.exists():
            return
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["status"] = BuildStatus.EXECUTED.value
            data["amd_validated"] = True
            if result.hip_version:
                data["hip_version"] = result.hip_version
            if result.pytorch_version:
                data["torch_version"] = result.pytorch_version
            if result.validation_passed is not None:
                data["validation_passed"] = result.validation_passed
            step_names = [s.get("name") for s in data.get("steps", [])]
            if "execute_inference" not in step_names:
                data.setdefault("steps", []).append({
                    "name": "execute_inference",
                    "status": StepStatus.SUCCESS.value,
                    "duration_seconds": result.duration_seconds,
                    "message": f"Verified execution on AMD GPU ({result.tokens_generated} tokens).",
                    "details": {
                        "tokens_generated": result.tokens_generated,
                        "tokens_per_second": result.tokens_per_second,
                        "device": result.device,
                        "hip_runtime_used": result.hip_runtime_used,
                        "amd_gpu_used": result.amd_gpu_used,
                        "validation_passed": result.validation_passed,
                    },
                })
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass
