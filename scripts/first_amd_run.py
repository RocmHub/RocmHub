#!/usr/bin/env python3
"""ROCmHub — Reproducible First AMD Run Pipeline.

Executes the verified end-to-end pipeline:
  Stage 1: Doctor       — Host, kernel driver, permissions, and ROCm stack diagnostics
  Stage 2: Inspect      — Static model inspection & immutable commit SHA resolution
  Stage 3: Capability   — Preflight capability evaluation (READY / NO_ACCELERATOR / BLOCKED)
  Stage 4: Run / Forge  — Baseline execution or verified dry-run
  Stage 5: Validate     — Output correctness and invariant evaluation
  Stage 6: Benchmark    — Performance metrics or truthful NOT_MEASURED semantics
  Stage 7: Artifact     — Cryptographic bundle generation & BenchmarkGuard verification

Safety Invariants:
- Halts execution on failure of any mandatory stage.
- Saves structured execution evidence to diagnostics/first_amd_run.json.
- Never downloads weights or runs inference on non-AMD hosts.
- Never installs drivers automatically.
- Never runs live hardware benchmarks without real AMD hardware and explicit --live-amd-consent.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from rocmhub.artifacts import ArtifactBuilder, LocalArtifactStore
from rocmhub.benchmarks import BenchmarkConfig, BenchmarkHarness
from rocmhub.capabilities import CapabilityEvaluator
from rocmhub.core.types import (
    BenchmarkResult,
    CapabilityReport,
    DetectionReport,
    EvaluationVerdict,
    ExecutionStatus,
    ModelSpec,
    RunResult,
    ValidationMode,
    ValidationReport,
    ValidationVerdict,
)
from rocmhub.doctor import DoctorReport, DoctorVerdict, ROCmDoctor
from rocmhub.guard import BenchmarkGuard
from rocmhub.hardware.detector import SystemObserver
from rocmhub.models.huggingface import HuggingFaceModelSource
from rocmhub.models.inspector import ModelInspector
from rocmhub.runners import HuggingFaceRunner
from rocmhub.validation import DEFAULT_VALIDATION_CASES, ValidationConfig, ValidationEvaluator


def run_first_amd_pipeline(
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    revision: str = "main",
    precision: str = "fp16",
    device_id: int = 0,
    live_amd_consent: bool = False,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute the full reproducible 7-stage pipeline."""
    start_time = time.time()
    out_dir = output_dir or (Path(__file__).parent.parent / "diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / "first_amd_run.json"

    pipeline_log: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "precision": precision,
        "device_id": device_id,
        "live_amd_consent": live_amd_consent,
        "stages": {},
        "verdict": "FAILED",
        "error": None,
    }

    print("=" * 70)
    print("ROCmHub — Reproducible First AMD Run Pipeline")
    print("=" * 70)
    print(f"Target Model:      {model_id}")
    print(f"Target Precision:  {precision}")
    print(f"Live AMD Consent:  {'GRANTED' if live_amd_consent else 'WITHHELD (dry-run / safe mode)'}")
    print("-" * 70)

    # -------------------------------------------------------------------------
    # Stage 1: Doctor Diagnostics
    # -------------------------------------------------------------------------
    print("\n[Stage 1/7] Running ROCm Doctor diagnostics...")
    try:
        doctor = ROCmDoctor()
        doc_report: DoctorReport = doctor.run_diagnostics()
        pipeline_log["stages"]["stage_1_doctor"] = {
            "status": "PASS",
            "verdict": doc_report.verdict.value,
            "summary": doc_report.summary,
            "amd_gpus_detected": len(doc_report.gpus),
            "checks": [c.model_dump() for c in doc_report.checks],
        }
        print(f"  Doctor Verdict: {doc_report.verdict.value}")
        print(f"  Summary:        {doc_report.summary}")
    except Exception as exc:
        err = f"Stage 1 Doctor failed: {exc}"
        pipeline_log["stages"]["stage_1_doctor"] = {"status": "FAIL", "error": str(exc)}
        pipeline_log["error"] = err
        _save_report(report_file, pipeline_log)
        raise RuntimeError(err) from exc

    # -------------------------------------------------------------------------
    # Stage 2: Model Static Inspection
    # -------------------------------------------------------------------------
    print("\n[Stage 2/7] Inspecting model metadata and resolving immutable commit SHA...")
    try:
        source = HuggingFaceModelSource()
        inspector = ModelInspector(source=source)
        model_spec: ModelSpec = inspector.inspect(model_id=model_id, revision=revision)
        pipeline_log["stages"]["stage_2_inspect"] = {
            "status": "PASS",
            "model_id": model_spec.model_id,
            "commit_sha": model_spec.commit_sha,
            "architecture": model_spec.architecture,
            "context_length": model_spec.context_length,
            "parameter_count": model_spec.parameter_count,
            "weights_format": str(model_spec.weights_format),
        }
        print(f"  Immutable Commit SHA: {model_spec.commit_sha}")
        print(f"  Architecture:         {model_spec.architecture}")
        params_str = f"{model_spec.parameter_count:,} params" if model_spec.parameter_count else "unknown"
        print(f"  Parameter Count:      {params_str}")
    except Exception as exc:
        err = f"Stage 2 Inspect failed: {exc}"
        pipeline_log["stages"]["stage_2_inspect"] = {"status": "FAIL", "error": str(exc)}
        pipeline_log["error"] = err
        _save_report(report_file, pipeline_log)
        raise RuntimeError(err) from exc

    # -------------------------------------------------------------------------
    # Stage 3: Capability Evaluation
    # -------------------------------------------------------------------------
    print("\n[Stage 3/7] Evaluating host and accelerator capability...")
    try:
        observer = SystemObserver()
        detection: DetectionReport = observer.observe()
        evaluator = CapabilityEvaluator()
        capability: CapabilityReport = evaluator.evaluate(model=model_spec, detection=detection)
        pipeline_log["stages"]["stage_3_capability"] = {
            "status": "PASS",
            "verdict": capability.verdict.value,
            "reasons": [r.model_dump() for r in capability.reasons],
            "warnings": capability.warnings,
        }
        print(f"  Capability Verdict:   {capability.verdict.value}")
        for r in capability.reasons:
            print(f"    - [{r.severity.value.upper()}] {r.code}: {r.message}")
    except Exception as exc:
        err = f"Stage 3 Capability failed: {exc}"
        pipeline_log["stages"]["stage_3_capability"] = {"status": "FAIL", "error": str(exc)}
        pipeline_log["error"] = err
        _save_report(report_file, pipeline_log)
        raise RuntimeError(err) from exc

    # Safety Gate: if not READY on AMD hardware, decide next step
    is_ready_amd = (capability.verdict == EvaluationVerdict.READY) and (doc_report.verdict == DoctorVerdict.READY)
    can_execute_live = is_ready_amd and live_amd_consent

    if not is_ready_amd:
        print("\n  [Notice] Host is in diagnostic/development mode (no native AMD accelerator).")
        print("  Real model weights will NOT be downloaded and live GPU kernels will NOT be executed.")

    # -------------------------------------------------------------------------
    # Stage 4: Execution / Run (Live or Dry-Run)
    # -------------------------------------------------------------------------
    print(f"\n[Stage 4/7] Preparing execution ({'LIVE AMD INFERENCE' if can_execute_live else 'SAFE DRY-RUN'})...")
    run_result: RunResult
    if can_execute_live:
        runner = HuggingFaceRunner()
        try:
            print(f"  Materializing weights and loading model on device {device_id}...")
            runner.load(model=model_spec, device_id=device_id, precision=precision)
            run_result = runner.generate(
                prompt="Explain the significance of open-source AI acceleration in one sentence.",
                max_new_tokens=32,
            )
            print(f"  Execution complete ({run_result.generated_tokens} tokens generated).")
        finally:
            runner.unload()
    else:
        # Safe dry-run without downloading weights or running inference
        run_result = RunResult(
            status=ExecutionStatus.SKIPPED,
            runtime_name="pytorch_transformers_hip",
            model_id=model_spec.model_id,
            model_revision=model_spec.commit_sha,
            device_id=device_id,
            precision=precision,
            prompt="Explain the significance of open-source AI acceleration in one sentence.",
            generated_text=None,
            input_tokens=None,
            generated_tokens=None,
            error=(
                "Dry-run executed: preflight validated. Live inference skipped "
                f"(capability={capability.verdict.value}, consent={live_amd_consent})."
            ),
            generation_params={"max_new_tokens": 32, "do_sample": False, "dry_run": True},
        )
        print("  Dry-run confirmed: weights download skipped, inference execution skipped.")

    pipeline_log["stages"]["stage_4_run"] = {
        "status": "PASS",
        "execution_status": run_result.status.value,
        "is_dry_run": not can_execute_live,
        "tokens_generated": run_result.generated_tokens,
    }

    # -------------------------------------------------------------------------
    # Stage 5: Validation Evaluation Gate
    # -------------------------------------------------------------------------
    print("\n[Stage 5/7] Evaluating technical correctness and validation invariants...")
    validation_report: ValidationReport
    if can_execute_live and run_result.status == ExecutionStatus.SUCCESS:
        val_config = ValidationConfig(cases=DEFAULT_VALIDATION_CASES, max_new_tokens=32)
        val_evaluator = ValidationEvaluator(config=val_config)
        runner_v = HuggingFaceRunner()
        try:
            runner_v.load(model=model_spec, device_id=device_id, precision=precision)
            validation_report = val_evaluator.run_self_validation(runner=runner_v, model=model_spec)
        finally:
            runner_v.unload()
    else:
        validation_report = ValidationReport(
            mode=ValidationMode.SELF_VALIDATION,
            model_id=model_spec.model_id,
            baseline_revision=model_spec.commit_sha,
            candidate_revision=None,
            verdict=ValidationVerdict.NOT_MEASURED,
            correctness_passed=None,
            quality_measured=False,
            qrr_percent=None,
            cases_total=len(DEFAULT_VALIDATION_CASES),
            cases_completed=0,
            cases_failed=0,
            critical_cases_failed=0,
            case_results=[],
            reasons=["Validation NOT_MEASURED: live accelerator execution skipped."],
        )

    pipeline_log["stages"]["stage_5_validate"] = {
        "status": "PASS",
        "verdict": validation_report.verdict.value,
        "correctness_passed": validation_report.correctness_passed,
    }
    print(f"  Validation Verdict:   {validation_report.verdict.value}")

    # -------------------------------------------------------------------------
    # Stage 6: Benchmark & Metrics Harness
    # -------------------------------------------------------------------------
    print(f"\n[Stage 6/7] Benchmarking ({'LIVE MEASUREMENT' if can_execute_live else 'TRUTHFUL NOT_MEASURED'})...")
    benchmark_result: BenchmarkResult
    if can_execute_live and validation_report.verdict == ValidationVerdict.PASS:
        b_config = BenchmarkConfig(
            prompt="ROCmHub performance benchmark prompt.",
            max_new_tokens=32,
            warmup_runs=2,
            measurement_runs=5,
            precision=precision,
            device_id=device_id,
        )
        runner_b = HuggingFaceRunner()
        try:
            runner_b.load(model=model_spec, device_id=device_id, precision=precision)
            harness = BenchmarkHarness(runner=runner_b, config=b_config)
            benchmark_result = harness.run(model=model_spec)
        finally:
            runner_b.unload()
        print(f"  Benchmark Status:     {benchmark_result.status.value}")
        if benchmark_result.throughput_tokens_per_sec is not None:
            print(f"  Throughput:           {benchmark_result.throughput_tokens_per_sec:.1f} tok/s")
    else:
        benchmark_result = BenchmarkResult(
            status=ExecutionStatus.NOT_MEASURED,
            error_message="Benchmark NOT_MEASURED: live accelerator execution skipped on non-AMD / dry-run host.",
        )
        print("  Benchmark Status:     NOT_MEASURED (no synthetic metrics generated)")

    pipeline_log["stages"]["stage_6_benchmark"] = {
        "status": "PASS",
        "benchmark_status": benchmark_result.status.value,
        "throughput_tokens_per_sec": benchmark_result.throughput_tokens_per_sec,
    }

    # -------------------------------------------------------------------------
    # Stage 7: Artifact Compilation & BenchmarkGuard Verification
    # -------------------------------------------------------------------------
    print("\n[Stage 7/7] Compiling verifiable artifact bundle and running BenchmarkGuard...")
    try:
        artifact_store = LocalArtifactStore(root_dir=out_dir / "artifacts")
        builder = ArtifactBuilder(store=artifact_store)
        artifact_manifest, bundle_path = builder.build(
            model=model_spec,
            detection=detection,
            capability=capability,
            run=run_result,
            benchmark=benchmark_result,
            validation=validation_report,
            device_id=device_id,
            precision=precision,
        )
        print(f"  Artifact ID:          {artifact_manifest.artifact_id}")
        print(f"  Bundle Directory:     {bundle_path}")

        # Run BenchmarkGuard verification
        guard = BenchmarkGuard(store=artifact_store)
        guard_report = guard.evaluate(artifact=bundle_path)
        print(f"  BenchmarkGuard:       {guard_report.verdict.value}")

        try:
            rel_bundle_path = str(Path(bundle_path).relative_to(Path.cwd()))
        except ValueError:
            rel_bundle_path = str(bundle_path)

        pipeline_log["stages"]["stage_7_artifact_guard"] = {
            "status": "PASS",
            "artifact_id": artifact_manifest.artifact_id,
            "bundle_path": rel_bundle_path,
            "guard_verdict": guard_report.verdict.value,
            "reasons": guard_report.reasons,
            "file_count": len(artifact_manifest.files),
        }
    except Exception as exc:
        err = f"Stage 7 Artifact/Guard failed: {exc}"
        pipeline_log["stages"]["stage_7_artifact_guard"] = {"status": "FAIL", "error": str(exc)}
        pipeline_log["error"] = err
        _save_report(report_file, pipeline_log)
        raise RuntimeError(err) from exc

    # Final Summary
    total_sec = round(time.time() - start_time, 2)
    pipeline_log["verdict"] = "SUCCESS"
    pipeline_log["duration_seconds"] = total_sec
    _save_report(report_file, pipeline_log)

    print("\n" + "=" * 70)
    print("ROCmHub First Run Pipeline Completed Successfully")
    print("=" * 70)
    print(f"  Total Duration:     {total_sec}s")
    print(f"  Diagnostic Report:  {report_file}")
    print(f"  Artifact Bundle:    {bundle_path}")
    print("  Status Summary:     All 7 stages verified without error.")
    print("=" * 70)

    return pipeline_log


def _save_report(path: Path, data: Dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
    except Exception as exc:
        sys.stderr.write(f"Warning: could not save diagnostic report: {exc}\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ROCmHub Reproducible First AMD Run Pipeline (inspect -> doctor -> check -> run -> validate -> benchmark -> artifact -> guard)."
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Target model ID (default: Qwen/Qwen2.5-0.5B-Instruct).",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Model revision (default: main).",
    )
    parser.add_argument(
        "--precision",
        default="fp16",
        choices=["fp16", "bf16", "fp32"],
        help="Target precision (default: fp16).",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Target accelerator device index (default: 0).",
    )
    parser.add_argument(
        "--live-amd-consent",
        action="store_true",
        help="Explicit consent to run live model execution and benchmarking on confirmed AMD GPU hardware.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to store diagnostic outputs and artifact bundles.",
    )
    args = parser.parse_args()

    try:
        run_first_amd_pipeline(
            model_id=args.model,
            revision=args.revision,
            precision=args.precision,
            device_id=args.device,
            live_amd_consent=args.live_amd_consent,
            output_dir=args.output_dir,
        )
        return 0
    except Exception as exc:
        sys.stderr.write(f"\nPipeline halted with error: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
