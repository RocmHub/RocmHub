"""Unit tests for ROCmHub core data models, serialization, and validation contracts."""

import pytest
from pydantic import ValidationError

from rocmhub.core.types import (
    CURRENT_SCHEMA_VERSION,
    ArtifactManifest,
    BenchmarkResult,
    EnvironmentSpec,
    ExecutionStatus,
    ExperimentSpec,
    HardwareSpec,
    ModelSpec,
)


class TestModelSpec:
    """Tests for ModelSpec data model and validation."""

    def test_valid_model_spec(self) -> None:
        spec = ModelSpec(
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
            requested_revision="main",
            commit_sha="a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
            architecture="Qwen2ForCausalLM",
            parameter_count=490_000_000,
            context_length=32768,
            default_dtype="bfloat16",
            weights_format="safetensors",
        )
        assert spec.schema_version == CURRENT_SCHEMA_VERSION
        assert spec.model_id == "Qwen/Qwen2.5-0.5B-Instruct"
        assert spec.source == "huggingface"
        assert spec.commit_sha == "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"

    def test_commit_sha_lowercased_and_trimmed(self) -> None:
        spec = ModelSpec(
            model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            commit_sha="  A1B2C3D4E5F6A1B2C3D4E5F6A1B2C3D4E5F6A1B2  ",
        )
        assert spec.commit_sha == "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"

    def test_empty_commit_sha_raises_error(self) -> None:
        with pytest.raises(ValidationError, match="commit_sha must not be empty"):
            ModelSpec(
                model_id="Qwen/Qwen2.5-0.5B-Instruct",
                commit_sha="   ",
            )

    def test_invalid_hex_commit_sha_raises_error(self) -> None:
        # 40 chars but contains non-hex character 'z'
        invalid_sha = "z1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"
        with pytest.raises(ValidationError, match="hexadecimal characters"):
            ModelSpec(
                model_id="Qwen/Qwen2.5-0.5B-Instruct",
                commit_sha=invalid_sha,
            )

    def test_negative_parameter_count_raises_error(self) -> None:
        with pytest.raises(ValidationError, match="parameter_count cannot be negative"):
            ModelSpec(
                model_id="test/model",
                commit_sha="1234567890abcdef1234567890abcdef12345678",
                parameter_count=-10,
            )

    def test_zero_or_negative_context_length_raises_error(self) -> None:
        with pytest.raises(ValidationError, match="context_length must be greater than zero"):
            ModelSpec(
                model_id="test/model",
                commit_sha="1234567890abcdef1234567890abcdef12345678",
                context_length=0,
            )


class TestHardwareSpec:
    """Tests for HardwareSpec data model, diagnostic fallback, and open gfx targets."""

    def test_diagnostic_cpu_environment(self) -> None:
        spec = HardwareSpec(
            gpu_present=False,
            gpu_vendor=None,
            device_id=None,
            device_name=None,
            family=None,
            gfx_target=None,
            vram_total_mb=None,
        )
        assert spec.gpu_present is False
        assert spec.gpu_vendor is None
        assert spec.gfx_target is None

    def test_unknown_or_new_gfx_target_as_open_string(self) -> None:
        spec = HardwareSpec(
            gpu_present=True,
            gpu_vendor="AMD",
            device_id=0,
            device_name="AMD Prototype Accelerator",
            family="NextGen",
            gfx_target="gfx1201_experimental",
            vram_total_mb=32768,
            compute_units=128,
        )
        assert spec.gfx_target == "gfx1201_experimental"
        assert spec.vram_total_mb == 32768

    def test_negative_vram_raises_error(self) -> None:
        with pytest.raises(ValidationError, match="cannot be negative"):
            HardwareSpec(
                gpu_present=True,
                vram_total_mb=-1024,
            )


class TestEnvironmentSpec:
    """Tests for EnvironmentSpec and non-ROCm host representations."""

    def test_environment_without_rocm(self) -> None:
        spec = EnvironmentSpec(
            os="Darwin 24.0.0",
            python_version="3.11.9",
            rocm_version=None,
            hip_version=None,
            torch_version="2.4.0",
            torch_hip_available=False,
            env_vars={},
        )
        assert spec.rocm_version is None
        assert spec.hip_version is None
        assert spec.torch_hip_available is False

    def test_environment_with_rocm(self) -> None:
        spec = EnvironmentSpec(
            os="Linux 6.8.0-40-generic",
            python_version="3.10.12",
            rocm_version="6.2.0",
            hip_version="6.2.41133",
            torch_version="2.4.0+rocm6.2",
            torch_hip_available=True,
            env_vars={"HSA_OVERRIDE_GFX_VERSION": "11.0.0"},
        )
        assert spec.rocm_version == "6.2.0"
        assert spec.torch_hip_available is True
        assert spec.env_vars["HSA_OVERRIDE_GFX_VERSION"] == "11.0.0"


class TestBenchmarkResult:
    """Tests for BenchmarkResult, metric units, and diagnostic integrity."""

    def test_success_benchmark_with_metrics(self) -> None:
        res = BenchmarkResult(
            status=ExecutionStatus.SUCCESS,
            ttft_ms=18.4,
            itl_ms_mean=8.2,
            itl_ms_p50=8.0,
            itl_ms_p90=8.8,
            itl_ms_p99=10.1,
            throughput_tokens_per_sec=121.9,
            peak_vram_used_mb=1850.5,
            total_latency_ms=838.4,
            generated_tokens_count=100,
            raw_latencies_ms=[18.4, 8.1, 8.2, 8.3],
        )
        assert res.status == ExecutionStatus.SUCCESS
        assert res.ttft_ms == 18.4
        assert res.throughput_tokens_per_sec == 121.9

    def test_not_measured_allows_nullable_metrics(self) -> None:
        res = BenchmarkResult(
            status=ExecutionStatus.NOT_MEASURED,
            error_message="Diagnostic mode run on host without AMD GPU. Inference skipped.",
        )
        assert res.status == ExecutionStatus.NOT_MEASURED
        assert res.ttft_ms is None
        assert res.throughput_tokens_per_sec is None
        assert res.peak_vram_used_mb is None

    def test_diagnostic_integrity_rejects_fake_metrics(self) -> None:
        """Diagnostic runs MUST NOT generate synthetic/mock performance numbers."""
        with pytest.raises(ValidationError, match="Diagnostic / non-measured results must not contain benchmark metrics"):
            BenchmarkResult(
                status=ExecutionStatus.NOT_MEASURED,
                ttft_ms=15.0,  # Fake metric not allowed
            )

        with pytest.raises(ValidationError, match="Diagnostic / non-measured results must not contain benchmark metrics"):
            BenchmarkResult(
                status=ExecutionStatus.SKIPPED,
                throughput_tokens_per_sec=100.0,
            )

    def test_negative_metrics_rejected(self) -> None:
        with pytest.raises(ValidationError, match="cannot be negative"):
            BenchmarkResult(
                status=ExecutionStatus.SUCCESS,
                ttft_ms=-1.0,
            )

    def test_invalid_status_enum(self) -> None:
        with pytest.raises(ValidationError):
            BenchmarkResult(status="SUPER_FAST")  # type: ignore


class TestArtifactManifest:
    """Tests for ArtifactManifest serialization, deserialization, and checksum verification."""

    @pytest.fixture
    def sample_manifest(self) -> ArtifactManifest:
        model = ModelSpec(
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
            requested_revision="main",
            commit_sha="a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
            architecture="Qwen2ForCausalLM",
            parameter_count=490_000_000,
        )
        hardware = HardwareSpec(
            gpu_present=True,
            gpu_vendor="AMD",
            device_name="AMD Radeon RX 7900 XTX",
            gfx_target="gfx1100",
            vram_total_mb=24576,
        )
        environment = EnvironmentSpec(
            os="Linux 6.8.0",
            python_version="3.11.9",
            rocm_version="6.2.0",
            torch_version="2.4.0+rocm6.2",
            torch_hip_available=True,
        )
        experiment = ExperimentSpec(
            experiment_id="exp-001-test",
            model=model,
            hardware=hardware,
            environment=environment,
            runtime_name="hf-transformers",
            precision="fp16",
            benchmark_params={"max_new_tokens": 64},
        )
        result = BenchmarkResult(
            status=ExecutionStatus.SUCCESS,
            ttft_ms=15.2,
            itl_ms_mean=7.8,
            throughput_tokens_per_sec=128.0,
            peak_vram_used_mb=1200.0,
        )
        return ArtifactManifest(
            manifest_id="manifest-test-001",
            experiment=experiment,
            result=result,
            reproduce_command="rocmhub run --model Qwen/Qwen2.5-0.5B-Instruct --precision fp16",
        )

    def test_manifest_roundtrip_json(self, sample_manifest: ArtifactManifest) -> None:
        # Sign manifest
        sample_manifest.sign_manifest()
        assert sample_manifest.manifest_checksum is not None

        # Serialize to JSON string
        json_str = sample_manifest.model_dump_json(indent=2)
        assert "manifest_checksum" in json_str
        assert "Qwen/Qwen2.5-0.5B-Instruct" in json_str

        # Deserialize back
        loaded = ArtifactManifest.model_validate_json(json_str)
        assert loaded.manifest_id == sample_manifest.manifest_id
        assert loaded.experiment.model.commit_sha == sample_manifest.experiment.model.commit_sha
        assert loaded.result.status == ExecutionStatus.SUCCESS
        assert loaded.verify_checksum() is True

    def test_tampered_manifest_fails_checksum_verification(self, sample_manifest: ArtifactManifest) -> None:
        sample_manifest.sign_manifest()
        assert sample_manifest.verify_checksum() is True

        # Tamper with the reproduce_command
        sample_manifest.reproduce_command = "malicious_command"
        assert sample_manifest.verify_checksum() is False
