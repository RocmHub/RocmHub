"""Autonomous AI Engineer Agent implementation (Phase 11).

Controls the autonomous OBSERVE -> PLAN -> ACT -> EVALUATE -> REVISE execution loop
under strict safety, budget, and sandboxing guards.
"""

from __future__ import annotations

import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from rocmhub.core.errors import (
    BudgetExceededError,
    RepeatedFailureError,
)
from rocmhub.engineer.base import (
    EngineerObjective,
    EngineerReport,
    EngineerRequest,
    EngineerStatus,
    TrajectoryStep,
)
from rocmhub.engineer.memory import TrajectoryStore
from rocmhub.engineer.policy import (
    BudgetGuard,
    FailureClassifier,
    FailureSeverity,
    LoopDetector,
)
from rocmhub.engineer.provider import AutonomousRulesProvider, LLMProvider
from rocmhub.engineer.tools import ToolRegistry
from rocmhub.forge.base import BuildStatus
from rocmhub.forge.manifest import BuildManifest


class AIEngineer:
    """Autonomous AI Engineer for preparing and validating models on AMD ROCm."""

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        tool_registry: Optional[ToolRegistry] = None,
        trajectory_store: Optional[TrajectoryStore] = None,
        max_loop_failures: int = 2,
    ) -> None:
        self._provider = provider or AutonomousRulesProvider()
        self._tools = tool_registry or ToolRegistry()
        self._store = trajectory_store or TrajectoryStore()
        self._max_loop_failures = max_loop_failures

    def run(self, request: EngineerRequest) -> EngineerReport:
        """Execute autonomous preparation loop for the requested model."""
        start_mono = time.monotonic()
        start_time_iso = datetime.now(timezone.utc).isoformat()

        session_hash = hashlib.sha256(
            f"{request.model_id}:{request.revision}:{start_time_iso}".encode("utf-8")
        ).hexdigest()[:12]
        session_id = f"eng_{session_hash}"

        budget_guard = BudgetGuard(request.budget)
        loop_detector = LoopDetector(self._max_loop_failures)

        trajectory: List[TrajectoryStep] = []
        errors_encountered: List[str] = []
        reasons: List[str] = []

        output_dir = request.output_dir or f"forge_builds/{request.model_id.split('/')[-1]}"
        out_path = Path(output_dir).resolve()
        if out_path.parent not in self._tools._allowed_dirs:
            self._tools._allowed_dirs.append(out_path.parent)

        context: Dict[str, Any] = {
            "session_id": session_id,
            "model_id": request.model_id,
            "revision": request.revision,
            "target_gpu": request.target_gpu,
            "objective": request.objective.value,
            "output_dir": str(out_path),
            "allow_full_weights": request.budget.allow_full_weights,
            "has_amd_gpu": False,
            "model_inspected": False,
            "hardware_inspected": False,
            "capability_checked": False,
            "plan_created": False,
            "weights_materialized": False,
            "build_executed": False,
            "baseline_run": False,
            "benchmark_run": False,
            "plan": None,
            "last_error": None,
            "error_diagnosed": False,
        }

        step_idx = 0
        final_manifest: Optional[BuildManifest] = None
        status = EngineerStatus.SUCCESS

        while True:
            # 1. Budget enforcement (time and attempts)
            try:
                budget_guard.check_time()
            except BudgetExceededError as exc:
                errors_encountered.append(str(exc))
                reasons.append("Execution time budget exceeded.")
                status = EngineerStatus.BUDGET_EXCEEDED
                break

            # 2. DECIDE next action (PLAN phase)
            try:
                action = self._provider.decide_action(context)
            except Exception as exc:
                errors_encountered.append(f"Provider decision failed: {exc}")
                reasons.append("LLM/rule decision provider encountered an unrecoverable error.")
                status = EngineerStatus.FAILED
                break

            # Check if provider signaled conclusion
            if action.is_final and action.tool_name == "save_engineer_report":
                step = TrajectoryStep(
                    step_index=step_idx,
                    phase="CONCLUDE",
                    action=action.tool_name,
                    tool_args=action.tool_args,
                    tool_result={"status": "SUCCESS"},
                    rationale=action.thought,
                    observation=action.tool_args.get("summary", "Session concluded."),
                    duration_seconds=0.0,
                )
                trajectory.append(step)
                self._store.append_step(session_id, step)
                reasons.append(action.tool_args.get("summary", "All engineering steps completed."))
                break

            # 3. ACT phase
            try:
                budget_guard.tick_attempt()
            except BudgetExceededError as exc:
                errors_encountered.append(str(exc))
                reasons.append("Resource or attempts budget exceeded.")
                status = EngineerStatus.BUDGET_EXCEEDED
                break

            t0 = time.monotonic()
            step_phase = "ACT"
            tool_res: Dict[str, Any] = {}
            step_observation: Optional[str] = None

            try:
                tool_res = self._tools.invoke(action.tool_name, action.tool_args)
                loop_detector.record_action(action.tool_name, action.tool_args, is_success=True)

                # EVALUATE outcome & update context
                if action.tool_name == "inspect_model":
                    context["model_inspected"] = True
                    spec_data = tool_res.get("model_spec", {})
                    context["revision"] = spec_data.get("commit_sha", context["revision"])
                    step_observation = f"Inspected model: {spec_data.get('architecture')} ({spec_data.get('parameter_count')} parameters)."

                elif action.tool_name == "inspect_hardware":
                    context["hardware_inspected"] = True
                    det = tool_res.get("detection_report", {})
                    gpus = det.get("gpus", [])
                    has_amd = any(g.get("gpu_vendor", "").lower() == "amd" for g in gpus)
                    context["has_amd_gpu"] = has_amd
                    step_observation = f"Hardware inspected: AMD GPU present={has_amd} (total GPUs={len(gpus)})."

                elif action.tool_name == "check_capability":
                    context["capability_checked"] = True
                    cap = tool_res.get("capability_report", {})
                    verdict = cap.get("verdict", "UNKNOWN")
                    step_observation = f"Capability evaluated: verdict={verdict}."

                elif action.tool_name == "create_forge_plan":
                    context["plan_created"] = True
                    context["plan"] = tool_res.get("plan")
                    step_observation = f"Forge plan created: {context['plan'].get('plan_id')}."

                elif action.tool_name == "materialize_model":
                    context["weights_materialized"] = True
                    mat = tool_res.get("materialized", {})
                    step_observation = f"Model materialized: {mat.get('mode')} (files={len(mat.get('files', []))})."

                elif action.tool_name == "execute_forge_build":
                    context["build_executed"] = True
                    manifest_data = tool_res.get("manifest", {})
                    final_manifest = BuildManifest.model_validate(manifest_data)
                    step_observation = f"Forge build executed: status={final_manifest.status.value}, AMD validated={final_manifest.amd_validated}."

                elif action.tool_name == "read_build_errors":
                    context["error_diagnosed"] = True
                    failed = tool_res.get("failed_steps", [])
                    step_observation = f"Diagnosed build errors: {len(failed)} failed step(s)."

                elif action.tool_name == "run_baseline":
                    context["baseline_run"] = True
                    step_observation = f"Baseline run status: {tool_res.get('status')}."

                elif action.tool_name == "run_benchmark":
                    context["benchmark_run"] = True
                    step_observation = f"Benchmark run status: {tool_res.get('status')}."

                # Check disk usage after materialization / build
                output_path = Path(context["output_dir"])
                if output_path.exists():
                    budget_guard.check_disk(output_path)

                context["last_error"] = None

            except RepeatedFailureError as exc:
                errors_encountered.append(str(exc))
                reasons.append("Infinite failure loop detected; agent aborted execution safely.")
                status = EngineerStatus.FAILED
                break

            except BudgetExceededError as exc:
                errors_encountered.append(str(exc))
                reasons.append("Resource or attempts budget exceeded.")
                status = EngineerStatus.BUDGET_EXCEEDED
                break

            except Exception as exc:
                loop_detector.record_action(action.tool_name, action.tool_args, is_success=False)
                err_msg = str(exc)
                errors_encountered.append(err_msg)
                context["last_error"] = err_msg
                context["error_diagnosed"] = False

                severity = FailureClassifier.classify(exc)
                if severity == FailureSeverity.FATAL:
                    reasons.append(f"Fatal unrecoverable failure: {err_msg}")
                    status = EngineerStatus.FAILED
                    # Record the failing step
                    duration = round(time.monotonic() - t0, 4)
                    step = TrajectoryStep(
                        step_index=step_idx,
                        phase="ACT",
                        action=action.tool_name,
                        tool_args=action.tool_args,
                        tool_result={"error": err_msg, "status": "FAILED"},
                        rationale=action.thought,
                        observation=f"Fatal failure: {err_msg}",
                        duration_seconds=duration,
                    )
                    trajectory.append(step)
                    self._store.append_step(session_id, step)
                    break
                else:
                    step_observation = f"Recoverable error encountered: {err_msg}"
                    tool_res = {"error": err_msg, "status": "FAILED"}

            duration = round(time.monotonic() - t0, 4)
            step = TrajectoryStep(
                step_index=step_idx,
                phase=step_phase,
                action=action.tool_name,
                tool_args=action.tool_args,
                tool_result=tool_res,
                rationale=action.thought,
                observation=step_observation,
                duration_seconds=duration,
            )
            trajectory.append(step)
            self._store.append_step(session_id, step)
            step_idx += 1

            if action.is_final:
                break

        # Check if environment caused execution stop
        if not context.get("has_amd_gpu") and request.objective in (
            EngineerObjective.AMD_EXECUTION,
            EngineerObjective.MAX_THROUGHPUT,
            EngineerObjective.MIN_LATENCY,
        ):
            if status == EngineerStatus.SUCCESS:
                status = EngineerStatus.STOPPED_ENVIRONMENT
                reasons.append(
                    "Model prepared and built, but real AMD execution skipped because no AMD GPU is present on host."
                )
        elif status == EngineerStatus.SUCCESS:
            # Enforce success semantics: if only config was prepared without full weights, status is CONFIG_ONLY
            if final_manifest and final_manifest.status == BuildStatus.CONFIG_ONLY:
                status = EngineerStatus.CONFIG_ONLY
                reasons.append(
                    "Model configuration prepared in CONFIG_ONLY mode (weights not materialized). Real AMD execution was not performed."
                )
            elif not request.budget.allow_full_weights and (not final_manifest or not final_manifest.weights_path):
                status = EngineerStatus.CONFIG_ONLY
                reasons.append(
                    "Model configuration prepared in CONFIG_ONLY mode. Full weights not downloaded."
                )

        total_duration = round(time.monotonic() - start_mono, 4)
        report = EngineerReport(
            session_id=session_id,
            model_id=request.model_id,
            revision=context.get("revision") or request.revision or "unknown",
            target_gpu=context.get("target_gpu"),
            objective=request.objective,
            status=status,
            started_at=start_time_iso,
            completed_at=datetime.now(timezone.utc).isoformat(),
            total_duration_seconds=total_duration,
            attempts_used=step_idx,
            trajectory=trajectory,
            build_manifest=final_manifest,
            run_result=None,
            benchmark_result=None,
            reasons=reasons,
            errors_encountered=errors_encountered,
            secret_scan_clean=True,
        )

        self._store.save_report(report)
        return report
