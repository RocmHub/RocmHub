"""BenchmarkGuard evaluator: independent audit gate for benchmark reproducibility and integrity."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from rocmhub.artifacts.storage import LocalArtifactStore
from rocmhub.benchmarks.base import BenchmarkRunMeasurement
from rocmhub.core.types import (
    ArtifactManifest,
    BenchmarkResult,
    EnvironmentFingerprint,
    EnvironmentSpec,
    ExecutionStatus,
    GuardPolicy,
    GuardVerdict,
    HardwareHealthSnapshot,
    HardwareSpec,
    ReferenceMeasurement,
    ReproducibilityReport,
)
from rocmhub.guard.base import (
    ARTIFACT_INTEGRITY_FAILED,
    DEFAULT_GUARD_POLICY,
    ENVIRONMENT_DRIFT,
    EVIDENCE_INCONSISTENT,
    HIGH_VARIABILITY,
    INSUFFICIENT_MEASUREMENT_RUNS,
    NO_BENCHMARK_EXECUTION,
    STABLE_MEASUREMENT,
    SUMMARY_MISMATCH,
)
from rocmhub.guard.environment import (
    compare_fingerprints,
    extract_environment_fingerprint,
)
from rocmhub.guard.reference import (
    compare_reference_measurements,
    evaluate_hardware_health,
)
from rocmhub.guard.statistics import (
    compute_relative_mad,
    verify_summary_against_raw,
)


class BenchmarkGuard:
    """Independent arbiter evaluating benchmark reproducibility and evidence truthfulness."""

    def __init__(self, store: Optional[LocalArtifactStore] = None) -> None:
        self.store = store or LocalArtifactStore()

    def evaluate(
        self,
        artifact: Path | str | ArtifactManifest,
        expected_fingerprint: Optional[EnvironmentFingerprint] = None,
        policy: Optional[GuardPolicy] = None,
        telemetry_snapshot: Optional[HardwareHealthSnapshot] = None,
        reference_before: Optional[ReferenceMeasurement] = None,
        reference_after: Optional[ReferenceMeasurement] = None,
        # Optional overrides primarily used for isolated in-memory unit tests:
        raw_measurements: Optional[Sequence[BenchmarkRunMeasurement | Dict[str, Any]]] = None,
        benchmark_override: Optional[BenchmarkResult] = None,
        environment_override: Optional[EnvironmentSpec] = None,
        hardware_override: Optional[List[HardwareSpec]] = None,
    ) -> ReproducibilityReport:
        """Evaluate an artifact bundle against reproducibility and truthful evidence criteria.

        Args:
            artifact: Filesystem path to artifact directory, artifact_id, or ArtifactManifest.
            expected_fingerprint: Optional expected environment fingerprint to detect drift.
            policy: GuardPolicy configuration (defaults to DEFAULT_GUARD_POLICY).
            telemetry_snapshot: Optional observed hardware health and telemetry metrics.
            reference_before: Optional pre-run calibration reference measurement.
            reference_after: Optional post-run calibration reference measurement.
            raw_measurements: Optional explicit raw run measurements (overrides on-disk file).
            benchmark_override: Optional explicit BenchmarkResult (overrides on-disk file).
            environment_override: Optional explicit EnvironmentSpec (overrides on-disk file).
            hardware_override: Optional explicit HardwareSpec list (overrides on-disk file).

        Returns:
            Structured ReproducibilityReport with top-level verdict and reason codes.
        """
        active_policy = policy or DEFAULT_GUARD_POLICY
        reasons: List[str] = []
        warnings: List[str] = []

        manifest: Optional[ArtifactManifest] = None
        target_dir: Optional[Path] = None

        # 1. Resolve artifact target and verify integrity
        if isinstance(artifact, ArtifactManifest):
            manifest = artifact
            if self.store.artifact_exists(manifest.artifact_id):
                target_dir = self.store.get_artifact_path(manifest.artifact_id)
        else:
            cand_path = Path(artifact).resolve()
            if cand_path.is_dir():
                target_dir = cand_path
            elif self.store.artifact_exists(str(artifact)):
                target_dir = self.store.get_artifact_path(str(artifact))
            else:
                return ReproducibilityReport(
                    artifact_id=str(artifact),
                    experiment_id="unknown",
                    policy_version=active_policy.policy_version,
                    verdict=GuardVerdict.FAIL,
                    environment_consistent=False,
                    hardware_consistent=False,
                    benchmark_complete=False,
                    reference_stable=None,
                    measurement_runs=0,
                    valid_runs=0,
                    reasons=[ARTIFACT_INTEGRITY_FAILED, f"Artifact path or ID not found: {artifact}"],
                    warnings=[],
                )

        if target_dir is not None:
            verification = self.store.verify_artifact(target_dir)
            if not verification.valid:
                return ReproducibilityReport(
                    artifact_id=verification.artifact_id,
                    experiment_id=manifest.experiment_id if manifest else "unknown",
                    policy_version=active_policy.policy_version,
                    verdict=GuardVerdict.FAIL,
                    environment_consistent=False,
                    hardware_consistent=False,
                    benchmark_complete=False,
                    reference_stable=None,
                    measurement_runs=0,
                    valid_runs=0,
                    reasons=[ARTIFACT_INTEGRITY_FAILED] + verification.errors,
                    warnings=[],
                )
            if manifest is None:
                manifest_file = target_dir / "manifest.json"
                manifest = ArtifactManifest.model_validate_json(manifest_file.read_bytes())

        if manifest is None:
            return ReproducibilityReport(
                artifact_id="unknown",
                experiment_id="unknown",
                policy_version=active_policy.policy_version,
                verdict=GuardVerdict.FAIL,
                environment_consistent=False,
                hardware_consistent=False,
                benchmark_complete=False,
                reference_stable=None,
                measurement_runs=0,
                valid_runs=0,
                reasons=[ARTIFACT_INTEGRITY_FAILED, "Manifest object could not be resolved."],
                warnings=[],
            )

        artifact_id = manifest.artifact_id
        experiment_id = manifest.experiment_id

        # 2. Load component payloads
        benchmark_obj = benchmark_override
        if benchmark_obj is None and target_dir is not None:
            bench_file = target_dir / "benchmark.json"
            if bench_file.is_file():
                try:
                    benchmark_obj = BenchmarkResult.model_validate_json(bench_file.read_bytes())
                except Exception as exc:
                    return ReproducibilityReport(
                        artifact_id=artifact_id,
                        experiment_id=experiment_id,
                        policy_version=active_policy.policy_version,
                        verdict=GuardVerdict.FAIL,
                        environment_consistent=False,
                        hardware_consistent=False,
                        benchmark_complete=False,
                        reference_stable=None,
                        measurement_runs=0,
                        valid_runs=0,
                        reasons=[ARTIFACT_INTEGRITY_FAILED, f"Corrupted benchmark.json: {exc}"],
                    )

        # 3. Check for Diagnostic / Skipped Execution
        # If real benchmark execution never occurred, Guard MUST evaluate to NOT_MEASURED
        if (
            benchmark_obj is None
            or benchmark_obj.status in (ExecutionStatus.SKIPPED, ExecutionStatus.NOT_MEASURED)
        ):
            reasons.append(NO_BENCHMARK_EXECUTION)
            return ReproducibilityReport(
                artifact_id=artifact_id,
                experiment_id=experiment_id,
                policy_version=active_policy.policy_version,
                verdict=GuardVerdict.NOT_MEASURED,
                environment_consistent=True,
                hardware_consistent=True,
                benchmark_complete=False,
                reference_stable=None,
                measurement_runs=0,
                valid_runs=0,
                ttft_variability=None,
                throughput_variability=None,
                itl_variability=None,
                reasons=reasons,
                warnings=warnings,
            )

        # 4. Environment Drift Check
        environment_consistent = True
        if expected_fingerprint is not None:
            env_spec = environment_override
            hw_specs = hardware_override or []
            if env_spec is None and target_dir is not None:
                env_file = target_dir / "environment.json"
                if env_file.is_file():
                    env_spec = EnvironmentSpec.model_validate_json(env_file.read_bytes())
                if not hw_specs and "devices" in manifest.hardware:
                    for dev_dict in manifest.hardware["devices"]:
                        try:
                            hw_specs.append(HardwareSpec.model_validate(dev_dict))
                        except Exception:
                            pass

            if env_spec is not None:
                observed_fingerprint = extract_environment_fingerprint(
                    environment=env_spec,
                    gpus=hw_specs,
                )
                matched, diff = compare_fingerprints(expected_fingerprint, observed_fingerprint)
                if not matched:
                    environment_consistent = False
                    reasons.append(ENVIRONMENT_DRIFT)
                    warnings.append(f"Environment drift detected in attributes: {list(diff.keys())}")
            else:
                environment_consistent = False
                reasons.append(ENVIRONMENT_DRIFT)
                warnings.append("Could not resolve environment specification to check fingerprint.")

        # 5. Hardware Health & Telemetry Check
        hardware_consistent, hw_reasons, hw_warnings = evaluate_hardware_health(
            telemetry_snapshot,
            active_policy,
        )
        reasons.extend(hw_reasons)
        warnings.extend(hw_warnings)

        # 6. Raw Measurements Retrieval
        runs_list: List[BenchmarkRunMeasurement] = []
        if raw_measurements is not None:
            for item in raw_measurements:
                if isinstance(item, BenchmarkRunMeasurement):
                    runs_list.append(item)
                else:
                    runs_list.append(BenchmarkRunMeasurement.model_validate(item))
        elif target_dir is not None:
            raw_file = target_dir / "benchmark_raw.json"
            if raw_file.is_file():
                try:
                    loaded = json.loads(raw_file.read_text(encoding="utf-8"))
                    if isinstance(loaded, list):
                        runs_list = [BenchmarkRunMeasurement.model_validate(r) for r in loaded]
                    elif isinstance(loaded, dict) and "runs" in loaded:
                        runs_list = [BenchmarkRunMeasurement.model_validate(r) for r in loaded["runs"]]
                except Exception as exc:
                    return ReproducibilityReport(
                        artifact_id=artifact_id,
                        experiment_id=experiment_id,
                        policy_version=active_policy.policy_version,
                        verdict=GuardVerdict.FAIL,
                        environment_consistent=environment_consistent,
                        hardware_consistent=hardware_consistent,
                        benchmark_complete=False,
                        reference_stable=None,
                        measurement_runs=0,
                        valid_runs=0,
                        reasons=[EVIDENCE_INCONSISTENT, f"Corrupted benchmark_raw.json: {exc}"],
                    )

        # 7. Raw Evidence Validation & Completeness
        if not runs_list:
            reasons.append(EVIDENCE_INCONSISTENT)
            return ReproducibilityReport(
                artifact_id=artifact_id,
                experiment_id=experiment_id,
                policy_version=active_policy.policy_version,
                verdict=GuardVerdict.FAIL,
                environment_consistent=environment_consistent,
                hardware_consistent=hardware_consistent,
                benchmark_complete=False,
                reference_stable=None,
                measurement_runs=0,
                valid_runs=0,
                reasons=reasons,
                warnings=warnings,
            )

        # Separate measurement runs from warmup
        meas_runs = [r for r in runs_list if not r.is_warmup]
        valid_runs = [r for r in meas_runs if r.status == ExecutionStatus.SUCCESS]
        total_meas = len(meas_runs)
        total_valid = len(valid_runs)

        has_failed_runs = any(r.status != ExecutionStatus.SUCCESS for r in meas_runs)
        benchmark_complete = (not has_failed_runs) and (total_valid >= (benchmark_obj.measurement_runs_requested or 1))

        if active_policy.require_complete_benchmark and has_failed_runs:
            reasons.append(EVIDENCE_INCONSISTENT)

        # 8. Summary Recomputation Audit (Never trust recorded headline metrics blindly)
        summary_mismatches = verify_summary_against_raw(
            benchmark_obj,
            runs_list,
            tolerance=active_policy.summary_recomputation_tolerance,
        )
        if summary_mismatches:
            reasons.append(SUMMARY_MISMATCH)
            for m in summary_mismatches:
                reasons.append(f"{SUMMARY_MISMATCH}: {m}")

        # 9. Statistical Stability & Dispersion Evaluation (MAD)
        ttft_vals = [r.ttft_ms for r in valid_runs if r.ttft_ms is not None]
        latency_vals = [r.total_latency_ms for r in valid_runs]
        throughput_vals: List[float] = []
        for r in valid_runs:
            if r.generated_tokens is not None and r.generated_tokens > 0:
                if r.first_token_at_ns is not None:
                    last_token_ns = r.first_token_at_ns + int(sum(r.inter_token_latencies_ms) * 1_000_000)
                    dur_s = max(1e-9, (last_token_ns - r.started_at_ns) / 1_000_000_000.0)
                else:
                    dur_s = max(1e-9, (r.finished_at_ns - r.started_at_ns) / 1_000_000_000.0)
                throughput_vals.append(r.generated_tokens / dur_s)

        ttft_var = compute_relative_mad(ttft_vals) if ttft_vals else None
        latency_var = compute_relative_mad(latency_vals) if latency_vals else None
        throughput_var = compute_relative_mad(throughput_vals) if throughput_vals else None

        # Check sample size
        insufficient_runs = total_valid < active_policy.min_measurement_runs
        if insufficient_runs:
            reasons.append(INSUFFICIENT_MEASUREMENT_RUNS)

        # Check dispersion thresholds
        high_variability = False
        if ttft_var is not None and ttft_var > active_policy.max_relative_mad_ttft:
            high_variability = True
        if latency_var is not None and latency_var > active_policy.max_relative_mad_latency:
            high_variability = True
        if throughput_var is not None and throughput_var > active_policy.max_relative_mad_throughput:
            high_variability = True

        if high_variability:
            reasons.append(HIGH_VARIABILITY)

        # 10. Reference Calibration Check
        reference_stable, ref_comparison, ref_reasons = compare_reference_measurements(
            reference_before,
            reference_after,
        )
        if ref_reasons:
            reasons.extend(ref_reasons)

        # 11. Final Verdict Determination
        # Priority: FAIL > INCONCLUSIVE > PASS
        is_fail = (
            (active_policy.require_environment_match and not environment_consistent)
            or (not hardware_consistent)
            or (active_policy.require_complete_benchmark and not benchmark_complete)
            or bool(summary_mismatches)
            or high_variability
            or (reference_stable is False)
        )

        if is_fail:
            verdict = GuardVerdict.FAIL
        elif insufficient_runs:
            verdict = GuardVerdict.INCONCLUSIVE
        else:
            verdict = GuardVerdict.PASS
            reasons.append(STABLE_MEASUREMENT)

        return ReproducibilityReport(
            artifact_id=artifact_id,
            experiment_id=experiment_id,
            policy_version=active_policy.policy_version,
            verdict=verdict,
            environment_consistent=environment_consistent,
            hardware_consistent=hardware_consistent,
            benchmark_complete=benchmark_complete,
            reference_stable=reference_stable,
            measurement_runs=total_meas,
            valid_runs=total_valid,
            ttft_variability=ttft_var,
            throughput_variability=throughput_var,
            itl_variability=latency_var,
            reasons=reasons,
            warnings=warnings,
        )
