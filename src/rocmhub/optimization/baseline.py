"""Baseline manager for ROCmHub Optimization Engine (Phase 12)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

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
from rocmhub.optimization.base import CandidateStatus, OptimizationBaseline
from rocmhub.runners.hf_runner import HuggingFaceRunner
from rocmhub.validation.base import ValidationConfig
from rocmhub.validation.evaluator import ValidationEvaluator


class BaselineManager:
    """Manages acquisition, execution, and provenance recording of the immutable baseline."""

    def __init__(
        self,
        inspector: Optional[ModelInspector] = None,
        observer: Optional[SystemObserver] = None,
    ) -> None:
        self._inspector = inspector or ModelInspector(source=HuggingFaceModelSource())
        self._observer = observer or SystemObserver()

    def record_baseline(
        self,
        model_id: str,
        revision: Optional[str] = None,
        target_gpu: Optional[str] = None,
        precision: str = "fp16",
        output_dir: Optional[Path] = None,
        allow_full_weights: bool = False,
        benchmark_config: Optional[BenchmarkConfig] = None,
        validation_config: Optional[ValidationConfig] = None,
    ) -> OptimizationBaseline:
        """Create and record baseline execution and provenance.

        On machines without AMD GPU, builds configuration in CONFIG_ONLY or PREPARED
        mode and safely marks metrics NOT_MEASURED.
        """
        planner = ForgePlanner(model_inspector=self._inspector, system_observer=self._observer)
        plan = planner.create_plan(
            model_id=model_id,
            revision=revision,
            target_gpu=target_gpu,
            precision=precision,
            output_dir=output_dir,
        )

        executor = ForgeExecutor(inspector=self._inspector, observer=self._observer)
        build_manifest = executor.execute(
            plan=plan,
            download_weights=allow_full_weights,
            execute_inference=False,
        )

        detected = self._observer.observe()
        has_amd_gpu = len(detected.gpus) > 0

        run_res: Optional[RunResult] = None
        bench_res: Optional[BenchmarkResult] = None
        val_rep: Optional[ValidationReport] = None
        measured = False
        status = (
            CandidateStatus.CONFIG_ONLY
            if build_manifest.status == BuildStatus.CONFIG_ONLY
            else CandidateStatus.PREPARED
        )

        # Only execute on real AMD GPU with materialized weights
        if has_amd_gpu and allow_full_weights and build_manifest.weights_path:
            # Execute real inference & benchmark
            model_spec = self._inspector.inspect(model_id, plan.revision)
            runner = HuggingFaceRunner()
            try:
                runner.load(model_spec, device_id=0, precision=precision)
                run_res = runner.generate("Hello, AMD ROCm!")

                b_cfg = benchmark_config or BenchmarkConfig(precision=precision, device_id=0)
                harness = BenchmarkHarness(runner=runner, config=b_cfg)
                bench_res = harness.run(model_spec)

                v_cfg = validation_config or ValidationConfig(precision=precision, device_id=0)
                evaluator = ValidationEvaluator(config=v_cfg)
                val_rep = evaluator.run_self_validation(runner=runner, model=model_spec)

                measured = bench_res.status == ExecutionStatus.SUCCESS
                status = CandidateStatus.EXECUTED
            finally:
                runner.unload()

        return OptimizationBaseline(
            model_id=model_id,
            revision=plan.revision,
            target_gpu=target_gpu or (detected.gpus[0].device_name if detected.gpus else None),
            precision=precision,
            runtime="pytorch_transformers_hip",
            build_dir=str(plan.output_dir),
            build_manifest=build_manifest,
            run_result=run_res,
            benchmark_result=bench_res,
            validation_report=val_rep,
            status=status,
            measured=measured,
        )
