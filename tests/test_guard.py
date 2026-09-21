"""Unit tests for Benchmark Guard & Reproducibility Gate (Phase 9)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import pytest

from rocmhub.artifacts.builder import ArtifactBuilder
from rocmhub.artifacts.storage import LocalArtifactStore
from rocmhub.benchmarks.base import BenchmarkRunMeasurement
from rocmhub.capabilities import CapabilityEvaluator
from rocmhub.cli.main import main
from rocmhub.core.types import (
    BenchmarkResult,
    CapabilityReport,
    DetectionReport,
    EnvironmentSpec,
    ExecutionStatus,
    ExperimentRole,
    GuardVerdict,
    HardwareHealthSnapshot,
    HardwareSpec,
    ModelSpec,
    ReferenceMeasurement,
    RunResult,
    ValidationMode,
    ValidationReport,
    ValidationVerdict,
)
from rocmhub.guard.base import (
    ARTIFACT_INTEGRITY_FAILED,
    DEFAULT_GUARD_POLICY,
    HARDWARE_ECC_ERRORS,
    HARDWARE_THROTTLING_DETECTED,
    HIGH_VARIABILITY,
    INSUFFICIENT_MEASUREMENT_RUNS,
    NO_BENCHMARK_EXECUTION,
    REFERENCE_DRIFT,
    STABLE_MEASUREMENT,
    SUMMARY_MISMATCH,
    TELEMETRY_UNAVAILABLE,
    ABBASequence,
)
from rocmhub.guard.environment import (
    compare_fingerprints,
    extract_environment_fingerprint,
)
from rocmhub.guard.evaluator import BenchmarkGuard
from rocmhub.guard.reference import (
    compare_reference_measurements,
    evaluate_hardware_health,
)
from rocmhub.guard.statistics import (
    compute_mad,
    compute_median,
    compute_relative_mad,
    recompute_benchmark_summary,
    verify_summary_against_raw,
)

# ---------------------------------------------------------------------------
# Test Fixtures & Helpers
# ---------------------------------------------------------------------------


def make_mock_detection(os_name: str = "Linux", torch_ver: str = "2.4.0+rocm6.2") -> DetectionReport:
    """Helper to build a deterministic mock DetectionReport."""
    return DetectionReport(
        environment=EnvironmentSpec(
            os=os_name,
            kernel="6.8.0-generic",
            architecture="x86_64",
            python_version="3.11.9",
            rocm_version="6.2.0",
            hip_version="6.2.31",
            torch_version=torch_ver,
            torch_hip_available=True,
            env_vars={"HSA_OVERRIDE_GFX_VERSION": "11.0.0"},
        ),
        gpus=[
            HardwareSpec(
                gpu_present=True,
                gpu_vendor="AMD",
                device_id=0,
                device_name="AMD Radeon RX 7900 XTX",
                family="Radeon",
                gfx_target="gfx1100",
                vram_total_mb=24576,
                compute_units=96,
                bus_id="0000:03:00.0",
            )
        ],
    )


def make_mock_diagnostic_detection() -> DetectionReport:
    """Helper to build a mock DetectionReport without accelerator (e.g. macOS / CPU)."""
    return DetectionReport(
        environment=EnvironmentSpec(
            os="Darwin 24.0.0",
            kernel="24.0.0",
            architecture="arm64",
            python_version="3.11.9",
            rocm_version=None,
            hip_version=None,
            torch_version="2.4.0",
            torch_hip_available=False,
            env_vars={},
        ),
        gpus=[],
    )


def make_mock_model(model_id: str, commit_sha: str = "a" * 40) -> ModelSpec:
    """Helper to build a deterministic mock ModelSpec."""
    return ModelSpec(
        model_id=model_id,
        commit_sha=commit_sha,
        architecture="Qwen2ForCausalLM",
        parameter_count=500_000_000,
        context_length=32768,
        default_dtype="float16",
        weights_format="safetensors",
    )


def make_mock_capability(model: ModelSpec, detection: DetectionReport) -> CapabilityReport:
    """Helper to evaluate model and detection into a valid CapabilityReport."""
    evaluator = CapabilityEvaluator()
    return evaluator.evaluate(model=model, detection=detection)


def make_mock_run(
    model: ModelSpec,
    status: ExecutionStatus = ExecutionStatus.SUCCESS,
    prompt: str = "hello",
    precision: str = "fp16",
    device_id: int = 0,
    generated_text: Optional[str] = "world",
    generated_tokens: Optional[int] = 16,
    error: Optional[str] = None,
) -> RunResult:
    """Helper to build a valid RunResult adhering to strict integrity validators."""
    is_success = status == ExecutionStatus.SUCCESS
    return RunResult(
        status=status,
        runtime_name="pytorch_transformers_hip",
        model_id=model.model_id,
        model_revision=model.commit_sha,
        prompt=prompt,
        precision=precision,
        device_id=device_id,
        generated_text=generated_text if is_success else None,
        input_tokens=10 if is_success else None,
        generated_tokens=generated_tokens if is_success else None,
        error=error,
    )


def make_mock_runs(
    count: int = 5,
    base_ttft: float = 20.0,
    base_latency: float = 100.0,
    jitter: float = 0.5,
    is_warmup: bool = False,
) -> List[BenchmarkRunMeasurement]:
    """Helper to generate consistent mock BenchmarkRunMeasurement objects."""
    runs: List[BenchmarkRunMeasurement] = []
    for i in range(count):
        # Apply small predictable jitter
        ttft = base_ttft + (i % 3 - 1) * jitter
        total_lat = base_latency + (i % 3 - 1) * (jitter * 2)
        start_ns = 1_000_000_000 + i * 1_000_000_000
        ttft_ns = int(ttft * 1_000_000)
        total_ns = int(total_lat * 1_000_000)

        runs.append(
            BenchmarkRunMeasurement(
                run_index=i,
                is_warmup=is_warmup,
                status=ExecutionStatus.SUCCESS,
                input_tokens=10,
                generated_tokens=16,
                started_at_ns=start_ns,
                first_token_at_ns=start_ns + ttft_ns,
                finished_at_ns=start_ns + total_ns,
                ttft_ms=ttft,
                inter_token_latencies_ms=[5.0] * 15,
                total_latency_ms=total_lat,
                peak_vram_used_mb=4096.0,
            )
        )
    return runs


# ---------------------------------------------------------------------------
# Unit Tests: Robust Statistics (Median, MAD, Relative MAD)
# ---------------------------------------------------------------------------


def test_compute_median_hand_calculated() -> None:
    """Test median computation with odd and even length lists."""
    assert compute_median([7]) == 7.0
    assert compute_median([1, 3, 2]) == 2.0
    assert compute_median([1, 2, 3, 4]) == 2.5
    assert compute_median([10, 20, 100, 30]) == 25.0

    with pytest.raises(ValueError, match="empty sequence"):
        compute_median([])


def test_compute_mad_hand_calculated() -> None:
    """Test MAD calculation on hand-calculated sequence."""
    # X = [1, 2, 3, 4, 5, 6, 7]
    # median(X) = 4
    # |x_i - 4| = [3, 2, 1, 0, 1, 2, 3] -> sorted: [0, 1, 1, 2, 2, 3, 3]
    # median(deviations) = 2.0
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    assert compute_mad(values) == 2.0

    # Constant sequence
    assert compute_mad([50.0, 50.0, 50.0, 50.0]) == 0.0


def test_compute_relative_mad() -> None:
    """Test relative MAD normalization."""
    # Constant sequence has zero relative MAD
    assert compute_relative_mad([100.0, 100.0, 100.0]) == 0.0

    # Empty or zero median
    assert compute_relative_mad([]) == 0.0
    assert compute_relative_mad([0.0, 0.0]) == 0.0

    # X = [1, 2, 3, 4, 5, 6, 7], med=4, mad=2 -> rel_mad = 2 / 4 = 0.5
    assert compute_relative_mad([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]) == 0.5


def test_recompute_benchmark_summary_and_verify() -> None:
    """Test recomputing benchmark summary and comparing against recorded summary."""
    warmups = make_mock_runs(count=2, base_ttft=50.0, base_latency=200.0, is_warmup=True)
    measured = make_mock_runs(count=5, base_ttft=20.0, base_latency=100.0, jitter=0.0, is_warmup=False)
    all_runs = warmups + measured

    summary = recompute_benchmark_summary(all_runs)
    # Warmup runs must be strictly excluded
    assert summary["median_ttft_ms"] == 20.0
    assert summary["median_total_latency_ms"] == 100.0
    assert summary["peak_vram_used_mb"] == 4096.0

    # Construct matching BenchmarkResult
    bench_result = BenchmarkResult(
        status=ExecutionStatus.SUCCESS,
        model_id="test/model",
        model_revision="abc1234",
        runtime_name="pytorch_transformers_hip",
        precision="fp16",
        device_id=0,
        warmup_runs=2,
        measurement_runs_requested=5,
        measurement_runs_completed=5,
        failed_runs=0,
        ttft_ms=summary["median_ttft_ms"],
        total_latency_ms=summary["median_total_latency_ms"],
        throughput_tokens_per_sec=summary["median_throughput_tokens_per_sec"],
        peak_vram_used_mb=summary["peak_vram_used_mb"],
        itl_ms_mean=summary["itl_mean_ms"],
        itl_ms_p50=summary["itl_p50_ms"],
        itl_ms_p90=summary["itl_p90_ms"],
        itl_ms_p99=summary["itl_p99_ms"],
    )

    # Verification should find zero mismatches
    mismatches = verify_summary_against_raw(bench_result, all_runs)
    assert mismatches == []

    # Introduce a forged metric into the recorded result
    forged_result = bench_result.model_copy(update={"ttft_ms": 10.0})  # 50% divergence
    mismatches_forged = verify_summary_against_raw(forged_result, all_runs)
    assert len(mismatches_forged) > 0
    assert any("ttft_ms" in m for m in mismatches_forged)


# ---------------------------------------------------------------------------
# Unit Tests: Environment Fingerprinting & Drift
# ---------------------------------------------------------------------------


def test_environment_fingerprint_deterministic() -> None:
    """Test that environment fingerprinting is deterministic and catches drift."""
    det1 = make_mock_detection()
    det2 = make_mock_detection()
    fp1 = extract_environment_fingerprint(det1)
    fp2 = extract_environment_fingerprint(det2)

    assert fp1.fingerprint_hash == fp2.fingerprint_hash
    match, diff = compare_fingerprints(fp1, fp2)
    assert match is True
    assert diff == {}

    # Drift in torch_version
    det_drifted = make_mock_detection(torch_ver="2.5.0+rocm6.3")
    fp_drifted = extract_environment_fingerprint(det_drifted)
    assert fp1.fingerprint_hash != fp_drifted.fingerprint_hash

    match_drift, diff_drift = compare_fingerprints(fp1, fp_drifted)
    assert match_drift is False
    assert "torch_version" in diff_drift
    assert diff_drift["torch_version"]["expected"] == "2.4.0+rocm6.2"
    assert diff_drift["torch_version"]["observed"] == "2.5.0+rocm6.3"


# ---------------------------------------------------------------------------
# Unit Tests: Hardware Health & Telemetry
# ---------------------------------------------------------------------------


def test_evaluate_hardware_health() -> None:
    """Test hardware health checking under different conditions."""
    policy = DEFAULT_GUARD_POLICY

    # 1. No telemetry with strict_telemetry=False (non-fatal warning)
    healthy, reasons, warnings = evaluate_hardware_health(None, policy)
    assert healthy is True
    assert reasons == []
    assert len(warnings) == 1

    # 2. No telemetry with strict_telemetry=True (fatal)
    strict_policy = policy.model_copy(update={"strict_telemetry": True})
    healthy_strict, reasons_strict, _ = evaluate_hardware_health(None, strict_policy)
    assert healthy_strict is False
    assert TELEMETRY_UNAVAILABLE in reasons_strict

    # 3. Clean telemetry
    clean_snap = HardwareHealthSnapshot(
        device_id=0,
        temperature_c=65.0,
        power_w=280.0,
        throttling_detected=False,
        ecc_errors=0,
    )
    healthy_clean, reasons_clean, _ = evaluate_hardware_health(clean_snap, policy)
    assert healthy_clean is True
    assert reasons_clean == []

    # 4. Throttling detected
    throttled_snap = HardwareHealthSnapshot(
        device_id=0,
        temperature_c=98.0,
        power_w=350.0,
        throttling_detected=True,
        ecc_errors=0,
    )
    healthy_throt, reasons_throt, _ = evaluate_hardware_health(throttled_snap, policy)
    assert healthy_throt is False
    assert HARDWARE_THROTTLING_DETECTED in reasons_throt

    # 5. ECC errors detected
    ecc_snap = HardwareHealthSnapshot(
        device_id=0,
        ecc_errors=3,
        throttling_detected=False,
    )
    healthy_ecc, reasons_ecc, _ = evaluate_hardware_health(ecc_snap, policy)
    assert healthy_ecc is False
    assert HARDWARE_ECC_ERRORS in reasons_ecc


# ---------------------------------------------------------------------------
# Unit Tests: Reference Calibration Check
# ---------------------------------------------------------------------------


def test_compare_reference_measurements() -> None:
    """Test reference run comparison rules."""
    # When absent, reference_stable must be None (never True)
    stable_none, comp_none, reasons_none = compare_reference_measurements(None, None)
    assert stable_none is None
    assert comp_none is None
    assert reasons_none == []

    before = ReferenceMeasurement(reference_id="gemm_fp16", metric_name="tflops", value=100.0)
    # 2% change: stable
    after_stable = ReferenceMeasurement(reference_id="gemm_fp16", metric_name="tflops", value=98.5)
    is_stable, comp_stable, reasons_stable = compare_reference_measurements(before, after_stable)
    assert is_stable is True
    assert comp_stable is not None
    assert comp_stable.stable is True
    assert reasons_stable == []

    # 10% change (> 5% default tolerance): drift
    after_drift = ReferenceMeasurement(reference_id="gemm_fp16", metric_name="tflops", value=89.0)
    is_drift, comp_drift, reasons_drift = compare_reference_measurements(before, after_drift)
    assert is_drift is False
    assert comp_drift is not None
    assert comp_drift.stable is False
    assert REFERENCE_DRIFT in reasons_drift


# ---------------------------------------------------------------------------
# Unit Tests: ABBASequence Data Contract
# ---------------------------------------------------------------------------


def test_abba_sequence_contract() -> None:
    """Verify ABBASequence data model contract."""
    seq = ABBASequence(
        sequence_id="cycle_001",
        baseline_experiment_id="exp_baseline",
        candidate_experiment_id="exp_candidate",
    )
    assert seq.sequence_order == [
        ExperimentRole.BASELINE,
        ExperimentRole.CANDIDATE,
        ExperimentRole.CANDIDATE,
        ExperimentRole.BASELINE,
    ]
    assert seq.baseline_experiment_id == "exp_baseline"
    assert seq.candidate_experiment_id == "exp_candidate"


# ---------------------------------------------------------------------------
# Unit Tests: BenchmarkGuard End-to-End Evaluation
# ---------------------------------------------------------------------------


def test_guard_preflight_diagnostic_artifact(tmp_path: Path) -> None:
    """Diagnostic artifacts without GPU execution evaluate to NOT_MEASURED (NO_BENCHMARK_EXECUTION)."""
    store = LocalArtifactStore(root_dir=tmp_path)
    builder = ArtifactBuilder(store=store)

    detection = make_mock_diagnostic_detection()
    model = make_mock_model("test/diagnostic-model", "a" * 40)
    cap = make_mock_capability(model, detection)
    run_res = make_mock_run(model, status=ExecutionStatus.SKIPPED, prompt="diagnostic", error="Skipped")
    bench_res = BenchmarkResult(
        status=ExecutionStatus.SKIPPED,
        model_id=model.model_id,
        model_revision=model.commit_sha,
        runtime_name="pytorch_transformers_hip",
        warmup_runs=2,
        measurement_runs_requested=5,
        measurement_runs_completed=0,
        failed_runs=0,
    )
    val_res = ValidationReport(
        mode=ValidationMode.SELF_VALIDATION,
        model_id=model.model_id,
        baseline_revision=model.commit_sha,
        verdict=ValidationVerdict.NOT_MEASURED,
        cases_total=5,
        cases_completed=0,
        cases_failed=0,
        critical_cases_failed=0,
    )

    manifest, bundle_path = builder.build(
        model=model,
        detection=detection,
        capability=cap,
        run=run_res,
        benchmark=bench_res,
        validation=val_res,
    )

    guard = BenchmarkGuard(store=store)
    report = guard.evaluate(bundle_path)

    assert report.verdict == GuardVerdict.NOT_MEASURED
    assert NO_BENCHMARK_EXECUTION in report.reasons
    assert report.valid_runs == 0
    assert report.ttft_variability is None
    assert report.throughput_variability is None
    assert report.reference_stable is None


def test_guard_corrupted_artifact_fails(tmp_path: Path) -> None:
    """BenchmarkGuard immediately rejects corrupted artifacts with FAIL (ARTIFACT_INTEGRITY_FAILED)."""
    store = LocalArtifactStore(root_dir=tmp_path)
    builder = ArtifactBuilder(store=store)

    detection = make_mock_detection()
    model = make_mock_model("test/model", "b" * 40)
    cap = make_mock_capability(model, detection)
    run_res = make_mock_run(model, status=ExecutionStatus.SUCCESS, prompt="hi", generated_text="hello", generated_tokens=10)
    bench_res = BenchmarkResult(
        status=ExecutionStatus.SKIPPED,
        model_id=model.model_id,
        model_revision=model.commit_sha,
        runtime_name="pytorch_transformers_hip",
        warmup_runs=2,
        measurement_runs_requested=5,
        measurement_runs_completed=0,
        failed_runs=0,
    )
    val_res = ValidationReport(
        mode=ValidationMode.SELF_VALIDATION,
        model_id=model.model_id,
        baseline_revision=model.commit_sha,
        verdict=ValidationVerdict.PASS,
        correctness_passed=True,
        cases_total=1,
        cases_completed=1,
        cases_failed=0,
        critical_cases_failed=0,
    )

    manifest, bundle_path = builder.build(
        model=model,
        detection=detection,
        capability=cap,
        run=run_res,
        benchmark=bench_res,
        validation=val_res,
    )

    # Tamper with environment.json
    env_file = bundle_path / "environment.json"
    env_file.write_text('{"tampered": true}', encoding="utf-8")

    guard = BenchmarkGuard(store=store)
    report = guard.evaluate(bundle_path)

    assert report.verdict == GuardVerdict.FAIL
    assert ARTIFACT_INTEGRITY_FAILED in report.reasons[0]


def test_guard_stable_runs_pass(tmp_path: Path) -> None:
    """BenchmarkGuard returns PASS for stable measurements matching all criteria."""
    store = LocalArtifactStore(root_dir=tmp_path)
    builder = ArtifactBuilder(store=store)

    detection = make_mock_detection()
    model = make_mock_model("test/stable-model", "c" * 40)
    cap = make_mock_capability(model, detection)
    run_res = make_mock_run(model, status=ExecutionStatus.SUCCESS, prompt="hello", generated_text="response", generated_tokens=16)

    raw_runs = make_mock_runs(count=5, base_ttft=20.0, base_latency=100.0, jitter=0.2)
    summary = recompute_benchmark_summary(raw_runs)

    bench_res = BenchmarkResult(
        status=ExecutionStatus.SUCCESS,
        model_id=model.model_id,
        model_revision=model.commit_sha,
        runtime_name="pytorch_transformers_hip",
        precision="fp16",
        device_id=0,
        warmup_runs=0,
        measurement_runs_requested=5,
        measurement_runs_completed=5,
        failed_runs=0,
        ttft_ms=summary["median_ttft_ms"],
        total_latency_ms=summary["median_total_latency_ms"],
        throughput_tokens_per_sec=summary["median_throughput_tokens_per_sec"],
        peak_vram_used_mb=summary["peak_vram_used_mb"],
        itl_ms_mean=summary["itl_mean_ms"],
        itl_ms_p50=summary["itl_p50_ms"],
        itl_ms_p90=summary["itl_p90_ms"],
        itl_ms_p99=summary["itl_p99_ms"],
    )

    val_res = ValidationReport(
        mode=ValidationMode.SELF_VALIDATION,
        model_id=model.model_id,
        baseline_revision=model.commit_sha,
        verdict=ValidationVerdict.PASS,
        correctness_passed=True,
        cases_total=1,
        cases_completed=1,
        cases_failed=0,
        critical_cases_failed=0,
    )

    manifest, bundle_path = builder.build(
        model=model,
        detection=detection,
        capability=cap,
        run=run_res,
        benchmark=bench_res,
        validation=val_res,
        raw_benchmark_data=[r.model_dump(mode="json") for r in raw_runs],
    )

    guard = BenchmarkGuard(store=store)
    report = guard.evaluate(bundle_path)

    assert report.verdict == GuardVerdict.PASS
    assert STABLE_MEASUREMENT in report.reasons
    assert report.valid_runs == 5
    assert report.environment_consistent is True
    assert report.hardware_consistent is True
    assert report.benchmark_complete is True
    assert report.ttft_variability is not None
    assert report.ttft_variability <= 0.15


def test_guard_summary_mismatch_fails(tmp_path: Path) -> None:
    """BenchmarkGuard catches summary tampering where recorded metrics diverged from raw runs."""
    store = LocalArtifactStore(root_dir=tmp_path)
    builder = ArtifactBuilder(store=store)

    detection = make_mock_detection()
    model = make_mock_model("test/mismatch-model", "d" * 40)
    cap = make_mock_capability(model, detection)
    run_res = make_mock_run(model, status=ExecutionStatus.SUCCESS, prompt="hello", generated_text="response", generated_tokens=16)

    raw_runs = make_mock_runs(count=5, base_ttft=20.0, base_latency=100.0, jitter=0.2)
    summary = recompute_benchmark_summary(raw_runs)

    # Fake the TTFT in recorded summary: claims 5.0 ms while raw runs show 20.0 ms
    bench_res = BenchmarkResult(
        status=ExecutionStatus.SUCCESS,
        model_id=model.model_id,
        model_revision=model.commit_sha,
        runtime_name="pytorch_transformers_hip",
        precision="fp16",
        device_id=0,
        warmup_runs=0,
        measurement_runs_requested=5,
        measurement_runs_completed=5,
        failed_runs=0,
        ttft_ms=5.0,  # FORGED!
        total_latency_ms=summary["median_total_latency_ms"],
        throughput_tokens_per_sec=summary["median_throughput_tokens_per_sec"],
        peak_vram_used_mb=summary["peak_vram_used_mb"],
        itl_ms_mean=summary["itl_mean_ms"],
        itl_ms_p50=summary["itl_p50_ms"],
        itl_ms_p90=summary["itl_p90_ms"],
        itl_ms_p99=summary["itl_p99_ms"],
    )

    val_res = ValidationReport(
        mode=ValidationMode.SELF_VALIDATION,
        model_id=model.model_id,
        baseline_revision=model.commit_sha,
        verdict=ValidationVerdict.PASS,
        correctness_passed=True,
        cases_total=1,
        cases_completed=1,
        cases_failed=0,
        critical_cases_failed=0,
    )

    manifest, bundle_path = builder.build(
        model=model,
        detection=detection,
        capability=cap,
        run=run_res,
        benchmark=bench_res,
        validation=val_res,
        raw_benchmark_data=[r.model_dump(mode="json") for r in raw_runs],
    )

    guard = BenchmarkGuard(store=store)
    report = guard.evaluate(bundle_path)

    assert report.verdict == GuardVerdict.FAIL
    assert SUMMARY_MISMATCH in report.reasons


def test_guard_insufficient_runs_inconclusive(tmp_path: Path) -> None:
    """BenchmarkGuard returns INCONCLUSIVE when runs count < policy min_measurement_runs."""
    store = LocalArtifactStore(root_dir=tmp_path)
    builder = ArtifactBuilder(store=store)

    detection = make_mock_detection()
    model = make_mock_model("test/insufficient-model", "e" * 40)
    cap = make_mock_capability(model, detection)
    run_res = make_mock_run(model, status=ExecutionStatus.SUCCESS, prompt="hello", generated_text="response", generated_tokens=16)

    # Only 3 measurement runs (policy requires 5)
    raw_runs = make_mock_runs(count=3, base_ttft=20.0, base_latency=100.0, jitter=0.1)
    summary = recompute_benchmark_summary(raw_runs)

    bench_res = BenchmarkResult(
        status=ExecutionStatus.SUCCESS,
        model_id=model.model_id,
        model_revision=model.commit_sha,
        runtime_name="pytorch_transformers_hip",
        precision="fp16",
        device_id=0,
        warmup_runs=0,
        measurement_runs_requested=3,
        measurement_runs_completed=3,
        failed_runs=0,
        ttft_ms=summary["median_ttft_ms"],
        total_latency_ms=summary["median_total_latency_ms"],
        throughput_tokens_per_sec=summary["median_throughput_tokens_per_sec"],
        peak_vram_used_mb=summary["peak_vram_used_mb"],
        itl_ms_mean=summary["itl_mean_ms"],
        itl_ms_p50=summary["itl_p50_ms"],
        itl_ms_p90=summary["itl_p90_ms"],
        itl_ms_p99=summary["itl_p99_ms"],
    )

    val_res = ValidationReport(
        mode=ValidationMode.SELF_VALIDATION,
        model_id=model.model_id,
        baseline_revision=model.commit_sha,
        verdict=ValidationVerdict.PASS,
        correctness_passed=True,
        cases_total=1,
        cases_completed=1,
        cases_failed=0,
        critical_cases_failed=0,
    )

    manifest, bundle_path = builder.build(
        model=model,
        detection=detection,
        capability=cap,
        run=run_res,
        benchmark=bench_res,
        validation=val_res,
        raw_benchmark_data=[r.model_dump(mode="json") for r in raw_runs],
    )

    guard = BenchmarkGuard(store=store)
    report = guard.evaluate(bundle_path)

    assert report.verdict == GuardVerdict.INCONCLUSIVE
    assert INSUFFICIENT_MEASUREMENT_RUNS in report.reasons
    assert report.valid_runs == 3


def test_guard_high_variability_fails(tmp_path: Path) -> None:
    """BenchmarkGuard returns FAIL when relative MAD exceeds variability threshold."""
    store = LocalArtifactStore(root_dir=tmp_path)
    builder = ArtifactBuilder(store=store)

    detection = make_mock_detection()
    model = make_mock_model("test/noisy-model", "f" * 40)
    cap = make_mock_capability(model, detection)
    run_res = make_mock_run(model, status=ExecutionStatus.SUCCESS, prompt="hello", generated_text="response", generated_tokens=16)

    # Extremely noisy TTFT: [10, 50, 100, 20, 90] -> high relative MAD > 15%
    raw_runs = [
        BenchmarkRunMeasurement(
            run_index=i,
            is_warmup=False,
            status=ExecutionStatus.SUCCESS,
            input_tokens=10,
            generated_tokens=16,
            started_at_ns=1_000_000_000 + i * 1_000_000_000,
            first_token_at_ns=1_000_000_000 + i * 1_000_000_000 + int(t * 1_000_000),
            finished_at_ns=1_000_000_000 + i * 1_000_000_000 + int((t + 50) * 1_000_000),
            ttft_ms=t,
            inter_token_latencies_ms=[3.0] * 15,
            total_latency_ms=t + 50,
            peak_vram_used_mb=4000.0,
        )
        for i, t in enumerate([10.0, 50.0, 100.0, 20.0, 90.0])
    ]
    summary = recompute_benchmark_summary(raw_runs)

    bench_res = BenchmarkResult(
        status=ExecutionStatus.SUCCESS,
        model_id=model.model_id,
        model_revision=model.commit_sha,
        runtime_name="pytorch_transformers_hip",
        precision="fp16",
        device_id=0,
        warmup_runs=0,
        measurement_runs_requested=5,
        measurement_runs_completed=5,
        failed_runs=0,
        ttft_ms=summary["median_ttft_ms"],
        total_latency_ms=summary["median_total_latency_ms"],
        throughput_tokens_per_sec=summary["median_throughput_tokens_per_sec"],
        peak_vram_used_mb=summary["peak_vram_used_mb"],
        itl_ms_mean=summary["itl_mean_ms"],
        itl_ms_p50=summary["itl_p50_ms"],
        itl_ms_p90=summary["itl_p90_ms"],
        itl_ms_p99=summary["itl_p99_ms"],
    )

    val_res = ValidationReport(
        mode=ValidationMode.SELF_VALIDATION,
        model_id=model.model_id,
        baseline_revision=model.commit_sha,
        verdict=ValidationVerdict.PASS,
        correctness_passed=True,
        cases_total=1,
        cases_completed=1,
        cases_failed=0,
        critical_cases_failed=0,
    )

    manifest, bundle_path = builder.build(
        model=model,
        detection=detection,
        capability=cap,
        run=run_res,
        benchmark=bench_res,
        validation=val_res,
        raw_benchmark_data=[r.model_dump(mode="json") for r in raw_runs],
    )

    guard = BenchmarkGuard(store=store)
    report = guard.evaluate(bundle_path)

    assert report.verdict == GuardVerdict.FAIL
    assert HIGH_VARIABILITY in report.reasons


# ---------------------------------------------------------------------------
# CLI Integration Tests for rocmhub guard
# ---------------------------------------------------------------------------


def test_cli_guard_exit_code_not_measured(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """CLI rocmhub guard on diagnostic bundle must exit with code 2."""
    store = LocalArtifactStore(root_dir=tmp_path)
    builder = ArtifactBuilder(store=store)

    detection = make_mock_diagnostic_detection()
    model = make_mock_model("test/diag", "1" * 40)
    cap = make_mock_capability(model, detection)
    run_res = make_mock_run(model, status=ExecutionStatus.SKIPPED, prompt="diag", error="Skipped")
    bench_res = BenchmarkResult(
        status=ExecutionStatus.SKIPPED,
        model_id=model.model_id,
        model_revision=model.commit_sha,
        runtime_name="pytorch_transformers_hip",
        warmup_runs=2,
        measurement_runs_requested=5,
        measurement_runs_completed=0,
        failed_runs=0,
    )
    val_res = ValidationReport(
        mode=ValidationMode.SELF_VALIDATION,
        model_id=model.model_id,
        baseline_revision=model.commit_sha,
        verdict=ValidationVerdict.NOT_MEASURED,
        cases_total=1,
        cases_completed=0,
        cases_failed=0,
        critical_cases_failed=0,
    )

    _, bundle_path = builder.build(
        model=model,
        detection=detection,
        capability=cap,
        run=run_res,
        benchmark=bench_res,
        validation=val_res,
    )

    exit_code = main(["guard", str(bundle_path)])
    assert exit_code == 2

    out = capsys.readouterr().out
    assert "Guard Verdict:             NOT_MEASURED" in out
    assert "NO_BENCHMARK_EXECUTION" in out


def test_cli_guard_json_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """CLI rocmhub guard --json outputs pure valid JSON."""
    store = LocalArtifactStore(root_dir=tmp_path)
    builder = ArtifactBuilder(store=store)

    detection = make_mock_diagnostic_detection()
    model = make_mock_model("test/diag-json", "2" * 40)
    cap = make_mock_capability(model, detection)
    run_res = make_mock_run(model, status=ExecutionStatus.SKIPPED, prompt="diag", error="Skipped")
    bench_res = BenchmarkResult(
        status=ExecutionStatus.SKIPPED,
        model_id=model.model_id,
        model_revision=model.commit_sha,
        runtime_name="pytorch_transformers_hip",
        warmup_runs=2,
        measurement_runs_requested=5,
        measurement_runs_completed=0,
        failed_runs=0,
    )
    val_res = ValidationReport(
        mode=ValidationMode.SELF_VALIDATION,
        model_id=model.model_id,
        baseline_revision=model.commit_sha,
        verdict=ValidationVerdict.NOT_MEASURED,
        cases_total=1,
        cases_completed=0,
        cases_failed=0,
        critical_cases_failed=0,
    )

    _, bundle_path = builder.build(
        model=model,
        detection=detection,
        capability=cap,
        run=run_res,
        benchmark=bench_res,
        validation=val_res,
    )

    exit_code = main(["guard", str(bundle_path), "--json"])
    assert exit_code == 2

    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["verdict"] == "NOT_MEASURED"
    assert "NO_BENCHMARK_EXECUTION" in data["reasons"]
