"""Optimization Executor orchestrating baseline and candidate lifecycle (Phase 12)."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import List, Optional

from rocmhub.benchmarks.base import BenchmarkConfig
from rocmhub.benchmarks.harness import BenchmarkHarness
from rocmhub.core.types import (
    BenchmarkResult,
    ExecutionStatus,
    RunResult,
    ValidationReport,
)
from rocmhub.forge.base import BuildStatus
from rocmhub.forge.executor import ForgeExecutor
from rocmhub.forge.planner import ForgePlanner
from rocmhub.hardware.detector import SystemObserver
from rocmhub.models.huggingface import HuggingFaceModelSource
from rocmhub.models.inspector import ModelInspector
from rocmhub.optimization.base import (
    CandidateStatus,
    ComparisonResult,
    ComparisonVerdict,
    OptimizationCandidate,
    OptimizationReport,
    OptimizationRequest,
)
from rocmhub.optimization.baseline import BaselineManager
from rocmhub.optimization.comparison import ComparisonEngine
from rocmhub.optimization.recipes import get_recipe_for_strategy
from rocmhub.runners.hf_runner import HuggingFaceRunner
from rocmhub.validation.base import ValidationConfig
from rocmhub.validation.evaluator import ValidationEvaluator


class OptimizationExecutor:
    """Orchestrates candidate generation, builds, execution, and comparison."""

    def __init__(
        self,
        inspector: Optional[ModelInspector] = None,
        observer: Optional[SystemObserver] = None,
    ) -> None:
        self._inspector = inspector or ModelInspector(source=HuggingFaceModelSource())
        self._observer = observer or SystemObserver()
        self._baseline_mgr = BaselineManager(
            inspector=self._inspector, observer=self._observer
        )
        self._comparison_engine = ComparisonEngine()

    def execute(self, request: OptimizationRequest) -> OptimizationReport:
        """Execute the end-to-end optimization pipeline."""
        start_mono = time.monotonic()
        session_id = f"opt_{hashlib.sha256(f'{request.model_id}:{time.time()}'.encode()).hexdigest()[:16]}"
        errors: List[str] = []
        recommendations: List[str] = []

        base_out_dir = (
            Path(request.output_dir)
            if request.output_dir
            else Path.cwd() / "builds" / f"opt--{request.model_id.replace('/', '--')}"
        )
        base_out_dir.mkdir(parents=True, exist_ok=True)

        # 1. Obtain Baseline
        baseline_dir = base_out_dir / "baseline"
        baseline = self._baseline_mgr.record_baseline(
            model_id=request.model_id,
            revision=request.revision,
            target_gpu=request.target_gpu,
            output_dir=baseline_dir,
            allow_full_weights=request.allow_full_weights,
            benchmark_config=request.benchmark_config,
            validation_config=request.validation_config,
        )

        detected = self._observer.observe()
        has_amd_gpu = len(detected.gpus) > 0
        model_spec = self._inspector.inspect(request.model_id, baseline.revision)
        hardware_spec = detected.gpus[0] if detected.gpus else None

        # 2. Plan and Build Candidates
        candidates: List[OptimizationCandidate] = []
        limit = min(request.max_candidates, len(request.strategies))

        forge_planner = ForgePlanner(model_inspector=self._inspector, system_observer=self._observer)
        forge_executor = ForgeExecutor(inspector=self._inspector, observer=self._observer)

        for strategy in request.strategies[:limit]:
            cand_hash = hashlib.sha256(
                f"{request.model_id}:{baseline.revision}:{strategy.value}:{request.target_gpu}".encode()
            ).hexdigest()[:16]
            cand_id = f"cand-{cand_hash}"
            cand_dir = base_out_dir / f"candidate_{strategy.value}"

            recipe = get_recipe_for_strategy(strategy)
            is_supp, supp_reason = recipe.is_supported(model_spec, hardware_spec)

            if not is_supp:
                cand = OptimizationCandidate(
                    candidate_id=cand_id,
                    model_id=request.model_id,
                    revision=baseline.revision,
                    strategy=strategy,
                    precision=recipe.precision,
                    runtime_flags=recipe.runtime_flags,
                    target_gpu=request.target_gpu,
                    build_dir=None,
                    build_manifest=None,
                    status=CandidateStatus.UNSUPPORTED,
                    errors=[supp_reason or "Strategy not supported."],
                )
                candidates.append(cand)
                continue

            # Build candidate configuration via Forge
            try:
                cand_plan = forge_planner.create_plan(
                    model_id=request.model_id,
                    revision=baseline.revision,
                    target_gpu=request.target_gpu,
                    precision=recipe.precision,
                    output_dir=cand_dir,
                )
                cand_manifest = forge_executor.execute(
                    plan=cand_plan,
                    download_weights=request.allow_full_weights,
                    execute_inference=False,
                )

                c_status = (
                    CandidateStatus.CONFIG_ONLY
                    if cand_manifest.status == BuildStatus.CONFIG_ONLY
                    else CandidateStatus.PREPARED
                )

                cand = OptimizationCandidate(
                    candidate_id=cand_id,
                    model_id=request.model_id,
                    revision=baseline.revision,
                    strategy=strategy,
                    precision=recipe.precision,
                    runtime_flags=recipe.runtime_flags,
                    target_gpu=request.target_gpu,
                    build_dir=str(cand_dir),
                    build_manifest=cand_manifest,
                    status=c_status,
                )
            except Exception as exc:
                err_msg = f"Candidate build failed for strategy '{strategy.value}': {exc}"
                errors.append(err_msg)
                cand = OptimizationCandidate(
                    candidate_id=cand_id,
                    model_id=request.model_id,
                    revision=baseline.revision,
                    strategy=strategy,
                    precision=recipe.precision,
                    runtime_flags=recipe.runtime_flags,
                    target_gpu=request.target_gpu,
                    build_dir=str(cand_dir),
                    build_manifest=None,
                    status=CandidateStatus.FAILED,
                    errors=[err_msg],
                )

            candidates.append(cand)

        # 3. Execution Phase (on real AMD GPU with full weights only)
        executed_candidates: List[OptimizationCandidate] = []
        if has_amd_gpu and request.allow_full_weights and baseline.measured:
            for cand in candidates:
                if cand.status not in (CandidateStatus.PREPARED, CandidateStatus.CONFIG_ONLY):
                    executed_candidates.append(cand)
                    continue

                if not cand.build_manifest or not cand.build_manifest.weights_path:
                    executed_candidates.append(cand)
                    continue

                # Run inference & benchmark on AMD GPU
                runner = HuggingFaceRunner()
                run_res: Optional[RunResult] = None
                bench_res: Optional[BenchmarkResult] = None
                val_rep: Optional[ValidationReport] = None
                c_status = cand.status
                cand_errors = list(cand.errors)

                try:
                    runner.load(model_spec, device_id=0, precision=cand.precision)
                    run_res = runner.generate("Hello, AMD ROCm candidate!")

                    b_cfg = request.benchmark_config or BenchmarkConfig(
                        precision=cand.precision, device_id=0
                    )
                    harness = BenchmarkHarness(runner=runner, config=b_cfg)
                    bench_res = harness.run(model_spec)

                    v_cfg = request.validation_config or ValidationConfig(
                        precision=cand.precision, device_id=0
                    )
                    evaluator = ValidationEvaluator(config=v_cfg)
                    val_rep = evaluator.run_self_validation(runner=runner, model=model_spec)

                    if bench_res.status == ExecutionStatus.SUCCESS:
                        c_status = CandidateStatus.EXECUTED
                    else:
                        c_status = CandidateStatus.FAILED
                except Exception as exc:
                    c_status = CandidateStatus.FAILED
                    cand_errors.append(f"Candidate execution failed: {exc}")
                finally:
                    runner.unload()

                executed_candidates.append(
                    OptimizationCandidate(
                        candidate_id=cand.candidate_id,
                        model_id=cand.model_id,
                        revision=cand.revision,
                        strategy=cand.strategy,
                        precision=cand.precision,
                        runtime_flags=cand.runtime_flags,
                        target_gpu=cand.target_gpu,
                        build_dir=cand.build_dir,
                        build_manifest=cand.build_manifest,
                        status=c_status,
                        run_result=run_res,
                        benchmark_result=bench_res,
                        validation_report=val_rep,
                        errors=cand_errors,
                    )
                )
        else:
            executed_candidates = candidates

        # 4. Compare all candidates against baseline
        comparisons: List[ComparisonResult] = []
        best_candidate_id: Optional[str] = None
        best_speedup = 1.0

        for cand in executed_candidates:
            comp = self._comparison_engine.compare(baseline, cand)
            comparisons.append(comp)

            if comp.verdict == ComparisonVerdict.IMPROVED:
                eff_speedup = comp.throughput_speedup or comp.ttft_speedup or 1.0
                if eff_speedup > best_speedup:
                    best_speedup = eff_speedup
                    best_candidate_id = cand.candidate_id

        # 5. Determine overall status and formulate recommendations
        if not request.allow_full_weights:
            status = "CONFIG_ONLY"
            recommendations.append(
                "Optimization variants planned and configured in CONFIG_ONLY mode without downloading full weights."
            )
            recommendations.append(
                "Use --allow-full-weights on a host with an AMD GPU to measure comparative performance."
            )
        elif not has_amd_gpu:
            status = "STOPPED_ENVIRONMENT"
            recommendations.append(
                "Candidates prepared, but GPU execution was skipped because no AMD GPU is present on host."
            )
            recommendations.append(
                "Transfer prepared builds to a ROCm host to run benchmarks and calculate speedup."
            )
        elif best_candidate_id:
            status = "SUCCESS"
            recommendations.append(
                f"Candidate {best_candidate_id} verified as superior with {best_speedup:.2f}x improvement."
            )
        else:
            status = "SUCCESS"
            recommendations.append("Optimization complete: no candidate exceeded baseline performance threshold.")

        elapsed = round(time.monotonic() - start_mono, 3)

        return OptimizationReport(
            session_id=session_id,
            model_id=request.model_id,
            revision=baseline.revision,
            target_gpu=request.target_gpu,
            objective=request.objective,
            status=status,
            baseline=baseline,
            candidates=executed_candidates,
            comparisons=comparisons,
            best_candidate_id=best_candidate_id,
            recommendations=recommendations,
            errors=errors,
            total_duration_seconds=elapsed,
        )
