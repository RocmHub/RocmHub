"""Restricted Tool Registry for Autonomous AI Engineer (Phase 11).

Provides strictly typed, sandboxed interfaces to ROCmHub subsystems.
No shell access, no arbitrary code execution, and no unvalidated file operations.
"""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from rocmhub.capabilities.evaluator import CapabilityEvaluator
from rocmhub.core.errors import (
    InvalidToolCallError,
    SecurityBoundaryError,
)
from rocmhub.forge.base import ForgePlan, MaterializationMode
from rocmhub.forge.executor import ForgeExecutor
from rocmhub.forge.materializer import ModelMaterializer
from rocmhub.forge.planner import ForgePlanner
from rocmhub.hardware.detector import SystemObserver
from rocmhub.models.huggingface import HuggingFaceModelSource
from rocmhub.models.inspector import ModelInspector
from rocmhub.runners.hf_runner import HuggingFaceRunner

ALLOWED_TOOLS: Set[str] = {
    "inspect_model",
    "inspect_hardware",
    "check_capability",
    "create_forge_plan",
    "materialize_model",
    "execute_forge_build",
    "run_baseline",
    "run_benchmark",
    "read_build_errors",
    "save_engineer_report",
}

FORBIDDEN_PREFIXES = (
    "/etc",
    "/private/etc",
    "/bin",
    "/sbin",
    "/usr",
    "/System",
    "/dev",
    "/proc",
    "/sys",
)


def validate_safe_path(path_str: str, allowed_parents: Optional[List[Path]] = None) -> Path:
    """Validate that path does not escape sandbox, use traversal, or touch system directories."""
    if ".." in path_str:
        raise SecurityBoundaryError(
            f"Path traversal detected in '{path_str}': '..' is forbidden.",
            details={"path": path_str},
        )

    try:
        target = Path(path_str).resolve()
    except Exception as exc:
        raise SecurityBoundaryError(f"Failed to resolve path '{path_str}': {exc}") from exc

    # Check against forbidden system roots
    target_str = str(target)
    if any(target_str.startswith(prefix) for prefix in FORBIDDEN_PREFIXES):
        raise SecurityBoundaryError(
            f"Access to protected system path '{target}' is strictly forbidden.",
            details={"path": target_str},
        )

    if allowed_parents:
        resolved_parents = [p.resolve() for p in allowed_parents]
        if not any(target == p or p in target.parents for p in resolved_parents):
            raise SecurityBoundaryError(
                f"Path '{target}' is outside permitted directory boundaries.",
                details={"path": str(target), "allowed_parents": [str(p) for p in resolved_parents]},
            )

    return target


def sanitize_untrusted_text(text: Optional[str]) -> Optional[str]:
    """Sanitize untrusted text from model cards or user inputs against prompt injection."""
    if not text:
        return text
    # Strip dangerous instruction headers or script tags
    sanitized = re.sub(r"(?i)<script.*?>.*?</script>", "", text, flags=re.DOTALL)
    sanitized = re.sub(r"(?i)(ignore previous instructions|system prompt|admin override)", "[FILTERED]", sanitized)
    return sanitized.strip()


def default_allowed_dirs() -> List[Path]:
    """Default permitted parent directories for file creation."""
    dirs = [Path.cwd().resolve(), Path("/tmp").resolve()]
    try:
        tmp = Path(tempfile.gettempdir()).resolve()
        if tmp not in dirs:
            dirs.append(tmp)
    except Exception:
        pass
    return dirs


class ToolRegistry:
    """Sandboxed tool execution registry."""

    def __init__(
        self,
        inspector: Optional[ModelInspector] = None,
        observer: Optional[SystemObserver] = None,
        evaluator: Optional[CapabilityEvaluator] = None,
        planner: Optional[ForgePlanner] = None,
        executor: Optional[ForgeExecutor] = None,
        materializer: Optional[ModelMaterializer] = None,
        runner: Optional[HuggingFaceRunner] = None,
        allowed_dirs: Optional[List[Path]] = None,
    ) -> None:
        self._inspector = inspector or ModelInspector(source=HuggingFaceModelSource())
        self._observer = observer or SystemObserver()
        self._evaluator = evaluator or CapabilityEvaluator()
        self._planner = planner or ForgePlanner(model_inspector=self._inspector, system_observer=self._observer)
        self._materializer = materializer or ModelMaterializer()
        self._executor = executor or ForgeExecutor(
            inspector=self._inspector,
            observer=self._observer,
            materializer=self._materializer,
        )
        self._runner = runner or HuggingFaceRunner()
        self._allowed_dirs = allowed_dirs or default_allowed_dirs()

    def invoke(self, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke an allowed tool by name with arguments."""
        if tool_name not in ALLOWED_TOOLS:
            raise InvalidToolCallError(
                f"Tool '{tool_name}' is not in the allowed tool registry. Allowed: {sorted(ALLOWED_TOOLS)}",
                details={"tool_name": tool_name},
            )

        handler: Callable[[Dict[str, Any]], Dict[str, Any]] = getattr(self, f"_tool_{tool_name}")
        return handler(tool_args)

    def _tool_inspect_model(self, args: Dict[str, Any]) -> Dict[str, Any]:
        model_id = args.get("model_id")
        if not model_id:
            raise InvalidToolCallError("inspect_model requires 'model_id' argument.")
        revision = args.get("revision") or "main"

        spec = self._inspector.inspect(model_id=model_id, revision=revision)
        # Mark description as untrusted
        dump = spec.model_dump()
        if "description" in dump:
            dump["description"] = sanitize_untrusted_text(dump["description"])
        return {"status": "SUCCESS", "model_spec": dump}

    def _tool_inspect_hardware(self, args: Dict[str, Any]) -> Dict[str, Any]:
        report = self._observer.observe()
        return {"status": "SUCCESS", "detection_report": report.model_dump()}

    def _tool_check_capability(self, args: Dict[str, Any]) -> Dict[str, Any]:
        model_id = args.get("model_id")
        if not model_id:
            raise InvalidToolCallError("check_capability requires 'model_id'.")
        revision = str(args.get("revision") or "main")

        spec = self._inspector.inspect(model_id=model_id, revision=revision)
        detection = self._observer.observe()
        report = self._evaluator.evaluate(model=spec, detection=detection)
        return {"status": "SUCCESS", "capability_report": report.model_dump()}

    def _tool_create_forge_plan(self, args: Dict[str, Any]) -> Dict[str, Any]:
        model_id = args.get("model_id")
        if not model_id:
            raise InvalidToolCallError("create_forge_plan requires 'model_id'.")
        output_dir = args.get("output_dir")
        if output_dir:
            validate_safe_path(output_dir, self._allowed_dirs)

        plan = self._planner.create_plan(
            model_id=model_id,
            revision=args.get("revision"),
            precision=args.get("precision", "fp16"),
            target_gpu=args.get("target_gpu"),
            output_dir=output_dir,
            recipe_id=args.get("recipe_id"),
        )
        return {"status": "SUCCESS", "plan": plan.model_dump()}

    def _tool_materialize_model(self, args: Dict[str, Any]) -> Dict[str, Any]:
        model_id = args.get("model_id")
        if not model_id:
            raise InvalidToolCallError("materialize_model requires 'model_id'.")
        revision = str(args.get("revision") or "main")
        download_weights = bool(args.get("download_weights", False))
        mode_str = args.get("mode")
        mode = MaterializationMode(mode_str) if mode_str else None

        spec = self._inspector.inspect(model_id=model_id, revision=revision)
        materialized = self._materializer.materialize(
            model_spec=spec,
            download_weights=download_weights,
            mode=mode,
        )
        return {"status": "SUCCESS", "materialized": materialized.model_dump()}

    def _tool_execute_forge_build(self, args: Dict[str, Any]) -> Dict[str, Any]:
        plan_dict = args.get("plan")
        if not plan_dict:
            raise InvalidToolCallError("execute_forge_build requires 'plan'.")
        plan = ForgePlan.model_validate(plan_dict)

        validate_safe_path(plan.output_dir, self._allowed_dirs)

        download_weights = bool(args.get("download_weights", False))
        force = bool(args.get("force", False))
        execute_inference = bool(args.get("execute_inference", False))

        manifest = self._executor.execute(
            plan=plan,
            download_weights=download_weights,
            force=force,
            execute_inference=execute_inference,
        )
        return {"status": "SUCCESS", "manifest": manifest.model_dump()}

    def _tool_run_baseline(self, args: Dict[str, Any]) -> Dict[str, Any]:
        report = self._observer.observe()
        has_amd = any(gpu.gpu_vendor and gpu.gpu_vendor.lower() == "amd" for gpu in report.gpus)
        if not has_amd:
            return {
                "status": "SKIPPED",
                "message": "Host is macOS or non-AMD. Execution safely skipped with zero synthetic metrics.",
            }

        model_id = args.get("model_id")
        if not model_id:
            raise InvalidToolCallError("run_baseline requires 'model_id'.")
        revision = str(args.get("revision") or "main")
        device_id = args.get("device_id", 0)
        precision = args.get("precision", "fp16")

        spec = self._inspector.inspect(model_id=model_id, revision=revision)
        try:
            self._runner.load(model=spec, device_id=device_id, precision=precision)
            res = self._runner.generate(prompt="Hello AMD ROCm!", max_new_tokens=16)
            return {"status": "SUCCESS", "run_result": res.model_dump()}
        finally:
            self._runner.unload()

    def _tool_run_benchmark(self, args: Dict[str, Any]) -> Dict[str, Any]:
        report = self._observer.observe()
        has_amd = any(gpu.gpu_vendor and gpu.gpu_vendor.lower() == "amd" for gpu in report.gpus)
        if not has_amd:
            return {
                "status": "NOT_MEASURED",
                "message": "Host is macOS or non-AMD. Benchmark safely skipped.",
            }

        return {
            "status": "NOT_MEASURED",
            "message": "Benchmark harness ready for live AMD GPU execution.",
        }

    def _tool_read_build_errors(self, args: Dict[str, Any]) -> Dict[str, Any]:
        build_dir_str = args.get("build_dir")
        if not build_dir_str:
            raise InvalidToolCallError("read_build_errors requires 'build_dir'.")

        build_dir = validate_safe_path(build_dir_str, self._allowed_dirs)
        manifest_path = build_dir / "build_manifest.json"
        if not manifest_path.exists():
            return {
                "status": "NO_MANIFEST",
                "message": f"No build_manifest.json found in '{build_dir}'.",
                "failed_steps": [],
            }

        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            steps = data.get("steps", [])
            failed = [s for s in steps if s.get("status") == "FAILED"]
            return {
                "status": "SUCCESS",
                "build_status": data.get("status"),
                "failed_steps": failed,
                "step_count": len(steps),
            }
        except Exception as exc:
            return {"status": "ERROR", "message": f"Failed to read manifest: {exc}"}

    def _tool_save_engineer_report(self, args: Dict[str, Any]) -> Dict[str, Any]:
        summary = args.get("summary", "Session concluded.")
        return {"status": "SUCCESS", "summary": summary}
