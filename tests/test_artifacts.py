"""Unit test suite for Phase 8: Reproducible Artifact Builder.

Tests cover:
- Deterministic canonical JSON serialization and SHA-256 hashing
- Fail-closed secret scanner (forbidden secrets rejected, legitimate tokens allowed)
- Deterministic experiment_id and content-derived artifact_id
- Sensitive changes (revision, precision, benchmark params) alter identity
- End-to-end artifact building and directory structure
- File inventory with per-file SHA-256 and byte sizes
- Detached checksums.json verification
- Tamper detection: modified file, missing file, unexpected file
- Prevention of silent overwrite and atomic conflict detection
- Atomic build cleanup on failure (leaves no corrupted final artifact)
- Separate raw benchmark data storage
- Diagnostic artifact on non-AMD systems marked COMPLETE with honest SKIPPED/NOT_MEASURED statuses
- Strict rejection of premature certification ('verified'/'certified' fields)
- CLI commands: 'artifact build' and 'artifact verify' with human-readable and JSON modes
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rocmhub.artifacts.builder import ArtifactBuilder
from rocmhub.artifacts.integrity import (
    canonical_json_bytes,
    canonical_json_dumps,
    compute_file_sha256,
    compute_sha256,
    scan_for_secrets,
)
from rocmhub.artifacts.manifest import (
    compute_artifact_id,
    compute_experiment_id,
)
from rocmhub.artifacts.storage import LocalArtifactStore
from rocmhub.cli.main import main
from rocmhub.core.errors import (
    ArtifactConflictError,
    SecretDetectedError,
)
from rocmhub.core.types import (
    ArtifactManifest,
    ArtifactStatus,
    BenchmarkResult,
    CapabilityReport,
    DetectionReport,
    EnvironmentSpec,
    EvaluationReason,
    EvaluationSeverity,
    EvaluationVerdict,
    ExecutionStatus,
    HardwareSpec,
    ModelSpec,
    RunResult,
    SystemCapabilities,
    ValidationMode,
    ValidationReport,
    ValidationVerdict,
)

# ---------------------------------------------------------------------------
# Fixtures & Test Data Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_model() -> ModelSpec:
    return ModelSpec(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        requested_revision="main",
        commit_sha="a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
        architecture="Qwen2ForCausalLM",
        parameter_count=490_000_000,
        context_length=32768,
        default_dtype="bfloat16",
        weights_format="safetensors",
    )


@pytest.fixture
def sample_hardware() -> HardwareSpec:
    return HardwareSpec(
        gpu_present=True,
        gpu_vendor="AMD",
        device_name="AMD Radeon RX 7900 XTX",
        device_id=0,
        gfx_target="gfx1100",
        vram_total_mb=24576,
        compute_units=96,
    )


@pytest.fixture
def sample_environment() -> EnvironmentSpec:
    return EnvironmentSpec(
        os="Linux 6.8.0",
        python_version="3.11.9",
        rocm_version="6.2.0",
        hip_version="6.2.0",
        torch_version="2.4.0+rocm6.2",
        torch_hip_available=True,
    )


@pytest.fixture
def sample_detection(sample_environment: EnvironmentSpec, sample_hardware: HardwareSpec) -> DetectionReport:
    return DetectionReport(
        environment=sample_environment,
        gpus=[sample_hardware],
    )


@pytest.fixture
def sample_capability(sample_model: ModelSpec, sample_environment: EnvironmentSpec, sample_hardware: HardwareSpec) -> CapabilityReport:
    return CapabilityReport(
        model=sample_model,
        environment=sample_environment,
        hardware=[sample_hardware],
        verdict=EvaluationVerdict.READY,
        reasons=[
            EvaluationReason(
                code="ROCM_READY",
                severity=EvaluationSeverity.OK,
                message="Compatible AMD GPU and ROCm stack detected.",
            )
        ],
        capabilities=SystemCapabilities(
            amd_gpu_present=True,
            rocm_detected=True,
            hip_detected=True,
            torch_available=True,
            torch_hip_available=True,
            model_metadata_complete=True,
            remote_code_required=False,
            baseline_runtime_candidate="pytorch_transformers_hip",
        ),
    )


@pytest.fixture
def sample_run(sample_model: ModelSpec) -> RunResult:
    return RunResult(
        status=ExecutionStatus.SUCCESS,
        runtime_name="pytorch_transformers_hip",
        model_id=sample_model.model_id,
        model_revision=sample_model.commit_sha,
        device_id=0,
        precision="fp16",
        prompt="Hello, world!",
        generated_text="Hello, world! I am ROCmHub.",
        input_tokens=4,
        generated_tokens=7,
        generation_params={"max_new_tokens": 16, "do_sample": False},
    )


@pytest.fixture
def sample_benchmark(sample_model: ModelSpec) -> BenchmarkResult:
    return BenchmarkResult(
        status=ExecutionStatus.SUCCESS,
        model_id=sample_model.model_id,
        model_revision=sample_model.commit_sha,
        device_id=0,
        runtime_name="pytorch_transformers_hip",
        precision="fp16",
        warmup_runs=2,
        measurement_runs_requested=5,
        measurement_runs_completed=5,
        failed_runs=0,
        ttft_ms=18.4,
        itl_ms_mean=8.2,
        itl_ms_p50=8.1,
        itl_ms_p90=8.9,
        itl_ms_p99=9.3,
        throughput_tokens_per_sec=121.5,
        peak_vram_used_mb=1450.0,
        total_latency_ms=134.8,
        generated_tokens_count=16,
    )


@pytest.fixture
def sample_validation(sample_model: ModelSpec) -> ValidationReport:
    return ValidationReport(
        mode=ValidationMode.SELF_VALIDATION,
        model_id=sample_model.model_id,
        baseline_revision=sample_model.commit_sha,
        verdict=ValidationVerdict.PASS,
        correctness_passed=True,
        quality_measured=False,
        qrr_percent=None,
        cases_total=3,
        cases_completed=3,
        cases_failed=0,
        critical_cases_failed=0,
        reasons=["All correctness gates passed."],
    )


# ---------------------------------------------------------------------------
# 1. Canonical Serialization & Determinism Tests
# ---------------------------------------------------------------------------

class TestCanonicalSerialization:
    def test_canonical_json_deterministic_key_ordering(self) -> None:
        dict_a = {"z": 1, "a": 2, "m": {"b": 3, "a": 4}}
        dict_b = {"a": 2, "m": {"a": 4, "b": 3}, "z": 1}

        json_a = canonical_json_dumps(dict_a)
        json_b = canonical_json_dumps(dict_b)

        assert json_a == json_b
        assert json_a == '{"a":2,"m":{"a":4,"b":3},"z":1}'

    def test_canonical_hashing_identical_across_invocations(self) -> None:
        data = {"precision": "fp16", "max_new_tokens": 16, "device": 0}
        hash1 = compute_sha256(canonical_json_bytes(data))
        hash2 = compute_sha256(canonical_json_bytes(data))
        assert hash1 == hash2

    def test_canonical_json_rejects_nan_and_infinity(self) -> None:
        with pytest.raises(ValueError):
            canonical_json_dumps({"metric": float("nan")})
        with pytest.raises(ValueError):
            canonical_json_dumps({"metric": float("inf")})


# ---------------------------------------------------------------------------
# 2. Secret Protection Tests
# ---------------------------------------------------------------------------

class TestSecretProtection:
    def test_scan_for_secrets_rejects_hf_token(self) -> None:
        payload = {"hf_token": "hf_secret_abc123456789"}
        with pytest.raises(SecretDetectedError) as exc_info:
            scan_for_secrets(payload)
        assert "hf_token" in str(exc_info.value)

    def test_scan_for_secrets_rejects_password_key(self) -> None:
        payload = {"db_password": "supersecretpassword"}
        with pytest.raises(SecretDetectedError):
            scan_for_secrets(payload)

    def test_scan_for_secrets_rejects_api_key(self) -> None:
        payload = {"config": {"api_key": "my-secret-key"}}
        with pytest.raises(SecretDetectedError):
            scan_for_secrets(payload)

    def test_scan_for_secrets_rejects_secret_patterns(self) -> None:
        for forbidden in ["aws_secret_access_key", "private_key", "ssh_key", "auth_token"]:
            with pytest.raises(SecretDetectedError):
                scan_for_secrets({forbidden: "val"})

    def test_scan_for_secrets_rejects_credential_value_prefixes(self) -> None:
        with pytest.raises(SecretDetectedError):
            scan_for_secrets({"data": "ghp_1234567890abcdefghijklmn"})
        with pytest.raises(SecretDetectedError):
            scan_for_secrets({"data": "AKIA1234567890ABCDEF"})

    def test_scan_for_secrets_allows_legitimate_token_counts(self) -> None:
        safe_payload = {
            "generated_tokens": 16,
            "input_tokens": 8,
            "max_new_tokens": 32,
            "prompt_tokens": 4,
            "tokens_per_sec": 120.5,
            "token_count": 24,
            "sort_keys": True,
        }
        # Must not raise SecretDetectedError
        scan_for_secrets(safe_payload)


# ---------------------------------------------------------------------------
# 3. Deterministic Identity Tests (experiment_id & artifact_id)
# ---------------------------------------------------------------------------

class TestDeterministicIdentity:
    def test_experiment_id_deterministic(self, sample_model: ModelSpec) -> None:
        hw = {"gpu_count": 1, "vendor": "AMD"}
        b_params = {"prompt": "test", "max_new_tokens": 16}
        v_params = {"mode": "SELF_VALIDATION"}

        id1 = compute_experiment_id(sample_model, hw, "pytorch_transformers_hip", "fp16", b_params, v_params)
        id2 = compute_experiment_id(sample_model, hw, "pytorch_transformers_hip", "fp16", b_params, v_params)
        assert id1 == id2
        assert id1.startswith("exp-")
        assert len(id1) == 28  # "exp-" + 24 hex chars

    def test_changing_revision_changes_experiment_id(self, sample_model: ModelSpec) -> None:
        hw = {"gpu_count": 1}
        b_params = {"max_new_tokens": 16}
        v_params = {"mode": "SELF_VALIDATION"}

        id1 = compute_experiment_id(sample_model, hw, "runtime", "fp16", b_params, v_params)

        other_model = ModelSpec(
            model_id=sample_model.model_id,
            requested_revision="v2.0",
            commit_sha="ffffffffffffffffffffffffffffffffffffffff",
            architecture=sample_model.architecture,
        )
        id2 = compute_experiment_id(other_model, hw, "runtime", "fp16", b_params, v_params)
        assert id1 != id2

    def test_changing_precision_changes_experiment_id(self, sample_model: ModelSpec) -> None:
        hw = {"gpu_count": 1}
        b_params = {"max_new_tokens": 16}
        v_params = {"mode": "SELF_VALIDATION"}

        id_fp16 = compute_experiment_id(sample_model, hw, "runtime", "fp16", b_params, v_params)
        id_bf16 = compute_experiment_id(sample_model, hw, "runtime", "bf16", b_params, v_params)
        assert id_fp16 != id_bf16

    def test_changing_benchmark_config_changes_experiment_id(self, sample_model: ModelSpec) -> None:
        hw = {"gpu_count": 1}
        v_params = {"mode": "SELF_VALIDATION"}

        id1 = compute_experiment_id(sample_model, hw, "runtime", "fp16", {"max_new_tokens": 16}, v_params)
        id2 = compute_experiment_id(sample_model, hw, "runtime", "fp16", {"max_new_tokens": 32}, v_params)
        assert id1 != id2

    def test_artifact_id_derived_from_files(self) -> None:
        exp_id = "exp-123456789012345678901234"
        files1 = {"model.json": "aaa", "benchmark.json": "bbb"}
        files2 = {"model.json": "aaa", "benchmark.json": "ccc"}

        art_id1 = compute_artifact_id(exp_id, files1)
        art_id2 = compute_artifact_id(exp_id, files2)

        assert art_id1.startswith("art-")
        assert art_id1 != art_id2


# ---------------------------------------------------------------------------
# 4. Artifact Builder & Bundle Structure Tests
# ---------------------------------------------------------------------------

class TestArtifactBuilder:
    def test_build_bundle_creates_expected_files(
        self,
        tmp_path: Path,
        sample_model: ModelSpec,
        sample_detection: DetectionReport,
        sample_capability: CapabilityReport,
        sample_run: RunResult,
        sample_benchmark: BenchmarkResult,
        sample_validation: ValidationReport,
    ) -> None:
        store = LocalArtifactStore(root_dir=tmp_path)
        builder = ArtifactBuilder(store=store)

        manifest, artifact_path = builder.build(
            model=sample_model,
            detection=sample_detection,
            capability=sample_capability,
            run=sample_run,
            benchmark=sample_benchmark,
            validation=sample_validation,
        )

        assert artifact_path.is_dir()
        assert manifest.artifact_id == artifact_path.name
        assert manifest.status == ArtifactStatus.COMPLETE

        # Verify all expected files are present on disk
        expected_files = [
            "manifest.json",
            "checksums.json",
            "model.json",
            "environment.json",
            "capabilities.json",
            "run.json",
            "benchmark.json",
            "validation.json",
            "reproduce.json",
        ]
        for name in expected_files:
            file_path = artifact_path / name
            assert file_path.is_file(), f"Missing expected file: {name}"
            assert file_path.stat().st_size > 0

    def test_build_bundle_with_raw_benchmark_data(
        self,
        tmp_path: Path,
        sample_model: ModelSpec,
        sample_detection: DetectionReport,
        sample_capability: CapabilityReport,
        sample_run: RunResult,
        sample_benchmark: BenchmarkResult,
        sample_validation: ValidationReport,
    ) -> None:
        store = LocalArtifactStore(root_dir=tmp_path)
        builder = ArtifactBuilder(store=store)

        raw_data = {"iteration_timings_ms": [18.2, 18.5, 18.1, 18.4, 18.3]}
        manifest, artifact_path = builder.build(
            model=sample_model,
            detection=sample_detection,
            capability=sample_capability,
            run=sample_run,
            benchmark=sample_benchmark,
            validation=sample_validation,
            raw_benchmark_data=raw_data,
        )

        raw_file = artifact_path / "benchmark_raw.json"
        assert raw_file.is_file()
        assert "benchmark_raw.json" in manifest.files

    def test_manifest_inventory_contains_all_payload_files(
        self,
        tmp_path: Path,
        sample_model: ModelSpec,
        sample_detection: DetectionReport,
        sample_capability: CapabilityReport,
        sample_run: RunResult,
        sample_benchmark: BenchmarkResult,
        sample_validation: ValidationReport,
    ) -> None:
        store = LocalArtifactStore(root_dir=tmp_path)
        builder = ArtifactBuilder(store=store)

        manifest, artifact_path = builder.build(
            model=sample_model,
            detection=sample_detection,
            capability=sample_capability,
            run=sample_run,
            benchmark=sample_benchmark,
            validation=sample_validation,
        )

        # Inventory in manifest covers data files with matching SHA and size
        for rel_path, entry in manifest.files.items():
            actual_file = artifact_path / rel_path
            assert actual_file.is_file()
            assert compute_file_sha256(actual_file) == entry.sha256
            assert actual_file.stat().st_size == entry.size_bytes


# ---------------------------------------------------------------------------
# 5. Integrity & Tamper Detection Tests
# ---------------------------------------------------------------------------

class TestArtifactIntegrityAndVerification:
    def test_valid_artifact_verifies_cleanly(
        self,
        tmp_path: Path,
        sample_model: ModelSpec,
        sample_detection: DetectionReport,
        sample_capability: CapabilityReport,
        sample_run: RunResult,
        sample_benchmark: BenchmarkResult,
        sample_validation: ValidationReport,
    ) -> None:
        store = LocalArtifactStore(root_dir=tmp_path)
        builder = ArtifactBuilder(store=store)

        _, artifact_path = builder.build(
            model=sample_model,
            detection=sample_detection,
            capability=sample_capability,
            run=sample_run,
            benchmark=sample_benchmark,
            validation=sample_validation,
        )

        res = store.verify_artifact(artifact_path)
        assert res.valid is True
        assert res.manifest_valid is True
        assert res.checksums_valid is True
        assert len(res.missing_files) == 0
        assert len(res.modified_files) == 0
        assert len(res.unexpected_files) == 0
        assert len(res.errors) == 0

    def test_tampered_file_detected(
        self,
        tmp_path: Path,
        sample_model: ModelSpec,
        sample_detection: DetectionReport,
        sample_capability: CapabilityReport,
        sample_run: RunResult,
        sample_benchmark: BenchmarkResult,
        sample_validation: ValidationReport,
    ) -> None:
        store = LocalArtifactStore(root_dir=tmp_path)
        builder = ArtifactBuilder(store=store)

        _, artifact_path = builder.build(
            model=sample_model,
            detection=sample_detection,
            capability=sample_capability,
            run=sample_run,
            benchmark=sample_benchmark,
            validation=sample_validation,
        )

        # Tamper with benchmark.json by appending a single character
        benchmark_file = artifact_path / "benchmark.json"
        content = benchmark_file.read_bytes()
        benchmark_file.write_bytes(content + b" ")

        res = store.verify_artifact(artifact_path)
        assert res.valid is False
        assert "benchmark.json" in res.modified_files

    def test_missing_file_detected(
        self,
        tmp_path: Path,
        sample_model: ModelSpec,
        sample_detection: DetectionReport,
        sample_capability: CapabilityReport,
        sample_run: RunResult,
        sample_benchmark: BenchmarkResult,
        sample_validation: ValidationReport,
    ) -> None:
        store = LocalArtifactStore(root_dir=tmp_path)
        builder = ArtifactBuilder(store=store)

        _, artifact_path = builder.build(
            model=sample_model,
            detection=sample_detection,
            capability=sample_capability,
            run=sample_run,
            benchmark=sample_benchmark,
            validation=sample_validation,
        )

        # Delete validation.json
        (artifact_path / "validation.json").unlink()

        res = store.verify_artifact(artifact_path)
        assert res.valid is False
        assert "validation.json" in res.missing_files

    def test_unexpected_file_detected(
        self,
        tmp_path: Path,
        sample_model: ModelSpec,
        sample_detection: DetectionReport,
        sample_capability: CapabilityReport,
        sample_run: RunResult,
        sample_benchmark: BenchmarkResult,
        sample_validation: ValidationReport,
    ) -> None:
        store = LocalArtifactStore(root_dir=tmp_path)
        builder = ArtifactBuilder(store=store)

        _, artifact_path = builder.build(
            model=sample_model,
            detection=sample_detection,
            capability=sample_capability,
            run=sample_run,
            benchmark=sample_benchmark,
            validation=sample_validation,
        )

        # Inject an unexpected extra file
        (artifact_path / "unauthorized_file.txt").write_text("rogue data")

        res = store.verify_artifact(artifact_path)
        assert res.valid is False
        assert "unauthorized_file.txt" in res.unexpected_files


# ---------------------------------------------------------------------------
# 6. No Silent Overwrite & Atomic Operations Tests
# ---------------------------------------------------------------------------

class TestAtomicAndOverwriteProtection:
    def test_conflicting_artifact_raises_conflict_error(
        self,
        tmp_path: Path,
        sample_model: ModelSpec,
        sample_detection: DetectionReport,
        sample_capability: CapabilityReport,
        sample_run: RunResult,
        sample_benchmark: BenchmarkResult,
        sample_validation: ValidationReport,
    ) -> None:
        store = LocalArtifactStore(root_dir=tmp_path)
        builder = ArtifactBuilder(store=store)

        manifest, artifact_path = builder.build(
            model=sample_model,
            detection=sample_detection,
            capability=sample_capability,
            run=sample_run,
            benchmark=sample_benchmark,
            validation=sample_validation,
        )

        # Pre-create staging directory with same ID but altered data
        staged_dir = tmp_path / f".staging_fake_{manifest.artifact_id}"
        staged_dir.mkdir(parents=True, exist_ok=True)
        (staged_dir / "checksums.json").write_text('{"files": {"other": "hash"}}')

        with pytest.raises(ArtifactConflictError):
            store.commit_staged_bundle(staged_dir, manifest.artifact_id)

    def test_failed_build_cleans_up_staging_leaving_no_final_artifact(
        self,
        tmp_path: Path,
        sample_model: ModelSpec,
        sample_detection: DetectionReport,
        sample_capability: CapabilityReport,
        sample_run: RunResult,
        sample_benchmark: BenchmarkResult,
        sample_validation: ValidationReport,
    ) -> None:
        store = LocalArtifactStore(root_dir=tmp_path)
        builder = ArtifactBuilder(store=store)

        # Mock store.commit_staged_bundle to raise an exception
        with patch.object(store, "commit_staged_bundle", side_effect=RuntimeError("Disk failure")):
            with pytest.raises(RuntimeError):
                builder.build(
                    model=sample_model,
                    detection=sample_detection,
                    capability=sample_capability,
                    run=sample_run,
                    benchmark=sample_benchmark,
                    validation=sample_validation,
                )

        # Confirm no artifacts or leftover staging directories exist in root_dir
        remaining = list(tmp_path.iterdir())
        assert len(remaining) == 0


# ---------------------------------------------------------------------------
# 7. Diagnostic Artifact & Certification Safeguards
# ---------------------------------------------------------------------------

class TestDiagnosticArtifactAndSafeguards:
    def test_diagnostic_artifact_marked_complete_with_honest_statuses(
        self,
        tmp_path: Path,
        sample_model: ModelSpec,
        sample_detection: DetectionReport,
    ) -> None:
        store = LocalArtifactStore(root_dir=tmp_path)
        builder = ArtifactBuilder(store=store)

        # Diagnostic inputs representing macOS without AMD GPU
        diag_capability = CapabilityReport(
            model=sample_model,
            environment=sample_detection.environment,
            hardware=[],
            verdict=EvaluationVerdict.NO_ACCELERATOR,
            reasons=[EvaluationReason(code="NO_AMD_GPU", severity=EvaluationSeverity.INFO, message="No AMD GPU detected.")],
            capabilities=SystemCapabilities(),
        )
        diag_run = RunResult(
            status=ExecutionStatus.SKIPPED,
            runtime_name="pytorch_transformers_hip",
            model_id=sample_model.model_id,
            model_revision=sample_model.commit_sha,
            precision="fp16",
            prompt="Hello, ROCmHub diagnostic run",
            error="Preflight NO_ACCELERATOR",
            generation_params={"max_new_tokens": 16, "do_sample": False},
        )
        diag_benchmark = BenchmarkResult(
            status=ExecutionStatus.SKIPPED,
            error_message="Preflight NO_ACCELERATOR",
            model_id=sample_model.model_id,
            model_revision=sample_model.commit_sha,
        )
        diag_validation = ValidationReport(
            mode=ValidationMode.SELF_VALIDATION,
            model_id=sample_model.model_id,
            baseline_revision=sample_model.commit_sha,
            verdict=ValidationVerdict.NOT_MEASURED,
            correctness_passed=None,
            quality_measured=False,
            qrr_percent=None,
        )

        manifest, artifact_path = builder.build(
            model=sample_model,
            detection=sample_detection,
            capability=diag_capability,
            run=diag_run,
            benchmark=diag_benchmark,
            validation=diag_validation,
        )

        assert manifest.status == ArtifactStatus.COMPLETE
        assert manifest.capability_verdict == EvaluationVerdict.NO_ACCELERATOR
        assert manifest.execution_status == ExecutionStatus.SKIPPED
        assert manifest.benchmark_status == ExecutionStatus.SKIPPED
        assert manifest.validation_verdict == ValidationVerdict.NOT_MEASURED

        # Verify bundle integrity
        res = store.verify_artifact(artifact_path)
        assert res.valid is True

    def test_artifact_manifest_forbids_certification_fields(self) -> None:
        # Pydantic validator must prevent injection of 'verified' or 'certified'
        with pytest.raises(Exception):
            ArtifactManifest(
                artifact_id="art-test",
                experiment_id="exp-test",
                verified=True,  # Prohibited!
            )  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# 8. CLI Command Tests ('artifact build' and 'artifact verify')
# ---------------------------------------------------------------------------

class TestCLIArtifactCommands:
    def _make_model_spec(self) -> ModelSpec:
        return ModelSpec(
            model_id="test/model",
            requested_revision="main",
            commit_sha="a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
            architecture="Qwen2ForCausalLM",
            parameter_count=500_000_000,
        )

    def _make_detection_report(self) -> DetectionReport:
        env = EnvironmentSpec(
            os="Darwin 24.0.0",
            python_version="3.11.9",
            torch_version="2.4.0",
            torch_hip_available=False,
        )
        return DetectionReport(environment=env, gpus=[])

    def _make_no_accelerator_report(self, model: ModelSpec, detection: DetectionReport) -> CapabilityReport:
        return CapabilityReport(
            model=model,
            environment=detection.environment,
            hardware=[],
            verdict=EvaluationVerdict.NO_ACCELERATOR,
            capabilities=SystemCapabilities(),
        )

    @patch("rocmhub.cli.main.CapabilityEvaluator")
    @patch("rocmhub.cli.main.SystemObserver")
    @patch("rocmhub.cli.main.ModelInspector")
    @patch("rocmhub.cli.main.HuggingFaceModelSource")
    def test_cli_artifact_build_diagnostic_mode(
        self,
        mock_hf: MagicMock,
        mock_inspector_cls: MagicMock,
        mock_observer_cls: MagicMock,
        mock_eval_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_model = self._make_model_spec()
        mock_detection = self._make_detection_report()
        mock_cap = self._make_no_accelerator_report(mock_model, mock_detection)

        mock_inspector_cls.return_value.inspect.return_value = mock_model
        mock_observer_cls.return_value.observe.return_value = mock_detection
        mock_eval_cls.return_value.evaluate.return_value = mock_cap

        out_dir = str(tmp_path / "artifacts")
        exit_code = main(["artifact", "build", "test/model", "--output-dir", out_dir])
        assert exit_code == 0

        # Verify an artifact directory was created
        created = list(Path(out_dir).iterdir())
        assert len(created) == 1
        artifact_path = created[0]
        assert (artifact_path / "manifest.json").is_file()
        assert (artifact_path / "checksums.json").is_file()

        # Run verify command against the built artifact
        verify_exit = main(["artifact", "verify", str(artifact_path)])
        assert verify_exit == 0

    @patch("rocmhub.cli.main.CapabilityEvaluator")
    @patch("rocmhub.cli.main.SystemObserver")
    @patch("rocmhub.cli.main.ModelInspector")
    @patch("rocmhub.cli.main.HuggingFaceModelSource")
    def test_cli_artifact_build_json_output(
        self,
        mock_hf: MagicMock,
        mock_inspector_cls: MagicMock,
        mock_observer_cls: MagicMock,
        mock_eval_cls: MagicMock,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_model = self._make_model_spec()
        mock_detection = self._make_detection_report()
        mock_cap = self._make_no_accelerator_report(mock_model, mock_detection)

        mock_inspector_cls.return_value.inspect.return_value = mock_model
        mock_observer_cls.return_value.observe.return_value = mock_detection
        mock_eval_cls.return_value.evaluate.return_value = mock_cap

        out_dir = str(tmp_path / "artifacts")
        exit_code = main(["artifact", "build", "test/model", "--output-dir", out_dir, "--json"])
        assert exit_code == 0

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["status"] == "COMPLETE"
        assert data["capability_verdict"] == "NO_ACCELERATOR"
        assert data["execution_status"] == "SKIPPED"
        assert data["validation_verdict"] == "NOT_MEASURED"
        assert "files" in data
        assert "manifest.json" not in data["files"]  # Manifest does not inventory itself
        assert "model.json" in data["files"]
