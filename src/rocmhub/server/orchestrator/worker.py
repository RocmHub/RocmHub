"""Worker routines executing domain tasks for ROCmHub jobs."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from rocmhub.engineer.agent import AIEngineer
from rocmhub.engineer.base import EngineerBudget, EngineerObjective, EngineerRequest
from rocmhub.forge.executor import ForgeExecutor
from rocmhub.forge.planner import ForgePlanner
from rocmhub.models.huggingface import HuggingFaceModelSource
from rocmhub.models.inspector import ModelInspector
from rocmhub.optimization.base import OptimizationRequest, OptimizationStrategy
from rocmhub.optimization.executor import OptimizationExecutor
from rocmhub.server.orchestrator.models import JobEvent, JobStatus, JobType
from rocmhub.server.security import sanitize_payload

logger = logging.getLogger(__name__)

EventEmitter = Callable[[str, str, str, Optional[str], Optional[Dict[str, Any]]], JobEvent]


def execute_job(
    job_id: str,
    job_type: JobType,
    request_data: Dict[str, Any],
    cancellation_event: threading.Event,
    emit: EventEmitter,
) -> Tuple[JobStatus, Optional[str], Optional[str], Optional[Dict[str, Any]], Optional[str], Optional[str], Optional[str]]:
    """Execute a single job with cooperative cancellation and event emissions.

    Returns:
        (job_status, domain_status, output_dir, result_payload, error_message, error_code, resolved_revision)
    """
    model_id = request_data["model_id"]
    revision = request_data.get("revision")
    target_gpu = request_data.get("target_gpu")
    output_dir_str = request_data.get("output_dir")

    emit("INITIALIZATION", "RUNNING", f"Initializing {job_type.value} job for model {model_id}", None, None)

    if cancellation_event.is_set():
        emit("CANCELLATION", "CANCELLED", "Job cancelled before execution started", None, None)
        return (JobStatus.CANCELLED, None, output_dir_str, None, "Job cancelled by user request", "JOB_CANCELLED", None)

    try:
        res: Any
        if job_type == JobType.FORGE_BUILD:
            res = _execute_forge_build(
                job_id, model_id, revision, target_gpu, request_data, cancellation_event, emit
            )
        elif job_type == JobType.ENGINEER:
            res = _execute_engineer(
                job_id, model_id, revision, target_gpu, request_data, cancellation_event, emit
            )
        elif job_type == JobType.OPTIMIZATION:
            res = _execute_optimization(
                job_id, model_id, revision, target_gpu, request_data, cancellation_event, emit
            )
        else:
            raise ValueError(f"Unsupported job type: {job_type}")

        if isinstance(res, tuple) and len(res) == 6:
            return (res[0], res[1], res[2], res[3], res[4], res[5], None)
        return (res[0], res[1], res[2], res[3], res[4], res[5], res[6])

    except Exception as exc:
        # A cancellation can race with a blocking domain call that fails while the
        # cancellation signal is being delivered. Preserve the user's terminal
        # intent instead of overwriting it with an unrelated transport failure.
        if cancellation_event.is_set():
            emit("CANCELLATION", "CANCELLED", "Job cancelled during execution", "JOB_CANCELLED", None)
            return (
                JobStatus.CANCELLED,
                None,
                output_dir_str,
                None,
                "Job cancelled by user request",
                "JOB_CANCELLED",
                None,
            )
        logger.exception("Job %s failed with exception", job_id)
        err_msg = str(exc)
        err_code = getattr(exc, "error_code", "INTERNAL_ERROR")
        emit("FAILURE", "FAILED", f"Execution failed: {err_msg}", err_code, None)
        return (JobStatus.FAILED, None, output_dir_str, None, err_msg, err_code, None)


def _execute_forge_build(
    job_id: str,
    model_id: str,
    revision: Optional[str],
    target_gpu: Optional[str],
    request_data: Dict[str, Any],
    cancellation_event: threading.Event,
    emit: EventEmitter,
) -> Tuple[JobStatus, Optional[str], Optional[str], Optional[Dict[str, Any]], Optional[str], Optional[str], Optional[str]]:
    """Execute FORGE_BUILD job."""
    precision = request_data.get("precision") or "fp16"
    allow_full_weights = request_data.get("allow_full_weights", False)
    output_dir_str = request_data.get("output_dir")
    output_dir = Path(output_dir_str) if output_dir_str else Path("builds") / f"forge--{model_id.replace('/', '--')}--{precision}"

    emit("PLANNING", "RUNNING", f"Creating Forge build plan for {model_id} ({precision})", None, None)

    source = HuggingFaceModelSource()
    inspector = ModelInspector(source)
    planner = ForgePlanner(model_inspector=inspector)

    plan = planner.create_plan(
        model_id=model_id,
        revision=revision or "main",
        precision=precision,
        target_gpu=target_gpu,
        output_dir=output_dir,
    )

    emit("PLANNING", "SUCCESS", f"Forge plan {plan.plan_id} created successfully", None, {"plan_id": plan.plan_id})

    if cancellation_event.is_set():
        emit("CANCELLATION", "CANCELLED", "Job cancelled after planning phase", None, None)
        return (JobStatus.CANCELLED, None, str(output_dir), None, "Job cancelled by user request", "JOB_CANCELLED", None)

    emit("BUILDING", "RUNNING", f"Executing build steps in {output_dir} (download_weights={allow_full_weights})", None, None)

    execute_inference = request_data.get("execute_inference", False)
    executor = ForgeExecutor()
    manifest = executor.execute(
        plan=plan,
        download_weights=allow_full_weights,
        force=True,
        execute_inference=execute_inference,
    )

    domain_status = manifest.status.value
    emit("FINALIZING", "SUCCESS", f"Forge build completed with domain status {domain_status}", None, {"manifest": manifest.model_dump(mode="json")})

    return (
        JobStatus.SUCCEEDED,
        domain_status,
        str(output_dir),
        sanitize_payload(manifest.model_dump(mode="json")),
        None,
        None,
        manifest.revision,
    )


def _execute_engineer(
    job_id: str,
    model_id: str,
    revision: Optional[str],
    target_gpu: Optional[str],
    request_data: Dict[str, Any],
    cancellation_event: threading.Event,
    emit: EventEmitter,
) -> Tuple[JobStatus, Optional[str], Optional[str], Optional[Dict[str, Any]], Optional[str], Optional[str], Optional[str]]:
    """Execute AI Engineer job."""
    objective_str = (request_data.get("objective") or "BASE_PREPARATION").strip().upper()
    if objective_str in ("PREPARE_AMD", "BASE", "PREPARE"):
        objective = EngineerObjective.BASE_PREPARATION
    else:
        try:
            objective = EngineerObjective(objective_str)
        except ValueError:
            objective = EngineerObjective.BASE_PREPARATION
    allow_full_weights = request_data.get("allow_full_weights", False)
    max_attempts = request_data.get("max_attempts") or 5
    max_minutes = (request_data.get("timeout_seconds") or 600) // 60
    max_disk_gb = request_data.get("max_disk_gb") or 10
    output_dir_str = request_data.get("output_dir")
    output_dir = Path(output_dir_str) if output_dir_str else Path("builds") / f"engineer--{model_id.replace('/', '--')}"

    budget = EngineerBudget(
        max_attempts=max_attempts,
        max_execution_time_seconds=max_minutes * 60,
        max_disk_usage_bytes=max_disk_gb * 1024 * 1024 * 1024,
        allow_full_weights=allow_full_weights,
    )

    request = EngineerRequest(
        model_id=model_id,
        revision=revision,
        target_gpu=target_gpu,
        objective=objective,
        budget=budget,
        output_dir=str(output_dir),
    )

    emit("ENGINEER_LOOP", "RUNNING", f"Starting AI Engineer autonomous loop for objective {objective.value}", None, None)

    if cancellation_event.is_set():
        emit("CANCELLATION", "CANCELLED", "Job cancelled before agent loop", None, None)
        return (JobStatus.CANCELLED, None, str(output_dir), None, "Job cancelled by user request", "JOB_CANCELLED", None)

    agent = AIEngineer()
    report = agent.run(request)

    domain_status = report.status.value
    emit("FINALIZING", "SUCCESS", f"AI Engineer completed with status {domain_status}", None, {"report": report.model_dump(mode="json")})

    resolved_revision = report.revision

    return (
        JobStatus.SUCCEEDED,
        domain_status,
        str(output_dir),
        sanitize_payload(report.model_dump(mode="json")),
        None,
        None,
        resolved_revision,
    )


def _execute_optimization(
    job_id: str,
    model_id: str,
    revision: Optional[str],
    target_gpu: Optional[str],
    request_data: Dict[str, Any],
    cancellation_event: threading.Event,
    emit: EventEmitter,
) -> Tuple[JobStatus, Optional[str], Optional[str], Optional[Dict[str, Any]], Optional[str], Optional[str], Optional[str]]:
    """Execute Optimization Engine job."""
    objective_str = request_data.get("objective") or "MAX_THROUGHPUT"
    strategies_input = request_data.get("strategies")
    strategies = [OptimizationStrategy(s) for s in strategies_input] if strategies_input else [
        OptimizationStrategy.BF16,
        OptimizationStrategy.FP16,
        OptimizationStrategy.FP32,
    ]
    max_candidates = request_data.get("max_candidates") or 3
    max_execution_time = float(request_data.get("timeout_seconds") or 600)
    allow_full_weights = request_data.get("allow_full_weights", False)
    output_dir_str = request_data.get("output_dir") or str(
        Path("optimizations") / model_id.replace("/", "--") / job_id
    )

    request = OptimizationRequest(
        model_id=model_id,
        revision=revision,
        target_gpu=target_gpu,
        objective=objective_str,
        strategies=strategies,
        max_candidates=max_candidates,
        max_execution_time_seconds=max_execution_time,
        allow_full_weights=allow_full_weights,
        output_dir=output_dir_str,
    )

    emit("OPTIMIZING", "RUNNING", f"Running Optimization Executor for {model_id} (objective: {objective_str})", None, None)

    if cancellation_event.is_set():
        emit("CANCELLATION", "CANCELLED", "Job cancelled before optimization pipeline", None, None)
        return (JobStatus.CANCELLED, None, output_dir_str, None, "Job cancelled by user request", "JOB_CANCELLED", None)

    executor = OptimizationExecutor()
    report = executor.execute(request)

    # Determine domain status: baseline status or best candidate status
    domain_status = report.baseline.status.value if report.baseline else "CONFIG_ONLY"
    emit("FINALIZING", "SUCCESS", f"Optimization completed with {len(report.candidates)} candidates", None, {"report": report.model_dump(mode="json")})

    resolved_revision = report.revision

    return (
        JobStatus.SUCCEEDED,
        domain_status,
        output_dir_str,
        sanitize_payload(report.model_dump(mode="json")),
        None,
        None,
        resolved_revision,
    )
