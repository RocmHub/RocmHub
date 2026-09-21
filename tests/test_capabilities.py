"""Unit tests for CapabilityEvaluator, CapabilityReport, and rocmhub check CLI."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from rocmhub.capabilities.evaluator import CapabilityEvaluator
from rocmhub.cli.main import main
from rocmhub.core.types import (
    DetectionReport,
    EnvironmentSpec,
    EvaluationReason,
    EvaluationSeverity,
    EvaluationVerdict,
    HardwareSpec,
    ModelSpec,
)


@pytest.fixture
def sample_model_spec() -> ModelSpec:
    return ModelSpec(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        requested_revision="main",
        commit_sha="a" * 40,
        architecture="Qwen2ForCausalLM",
        parameter_count=494_032_768,
        context_length=32768,
        default_dtype="bfloat16",
        weights_format="safetensors",
        remote_code_required=False,
    )


@pytest.fixture
def ready_detection_report() -> DetectionReport:
    env = EnvironmentSpec(
        os="Linux 6.8.0-45-generic",
        kernel="6.8.0-45-generic",
        architecture="x86_64",
        python_version="3.11.9",
        rocm_version="6.2.0",
        hip_version="6.2.41133",
        torch_version="2.4.0+rocm6.2",
        torch_hip_available=True,
        env_vars={"HSA_OVERRIDE_GFX_VERSION": "11.0.0"},
    )
    gpu = HardwareSpec(
        gpu_present=True,
        gpu_vendor="AMD",
        device_id=0,
        device_name="AMD Radeon RX 7900 XTX",
        family="Radeon",
        gfx_target="gfx1100",
        vram_total_mb=24560,
        compute_units=96,
        bus_id="0000:03:00.0",
    )
    return DetectionReport(
        environment=env,
        gpus=[gpu],
        provenance={"gpu_0": "rocminfo"},
        warnings=[],
    )


@pytest.fixture
def mac_detection_report() -> DetectionReport:
    env = EnvironmentSpec(
        os="Darwin 25.5.0",
        kernel="25.5.0",
        architecture="arm64",
        python_version="3.9.6",
        rocm_version=None,
        hip_version=None,
        torch_version="not_installed",
        torch_hip_available=False,
        env_vars={},
    )
    return DetectionReport(
        environment=env,
        gpus=[],
        provenance={"os": "platform"},
        warnings=["rocminfo tool not found or returned no output"],
    )


class TestCapabilityEvaluator:
    def test_ready_amd_environment(
        self,
        sample_model_spec: ModelSpec,
        ready_detection_report: DetectionReport,
    ) -> None:
        evaluator = CapabilityEvaluator()
        report = evaluator.evaluate(sample_model_spec, ready_detection_report)

        assert report.verdict == EvaluationVerdict.READY
        assert report.capabilities.amd_gpu_present is True
        assert report.capabilities.rocm_detected is True
        assert report.capabilities.hip_detected is True
        assert report.capabilities.torch_hip_available is True
        assert report.capabilities.baseline_runtime_candidate == "pytorch_transformers_hip"

        assert len(report.device_assessments) == 1
        assert report.device_assessments[0].verdict == EvaluationVerdict.READY
        assert report.device_assessments[0].gfx_target == "gfx1100"

        codes = [r.code for r in report.reasons]
        assert "AMD_GPU_PRESENT" in codes
        assert "ROCM_DETECTED" in codes
        assert "TORCH_HIP_AVAILABLE" in codes
        assert "BASELINE_CANDIDATE_READY" in codes

    def test_mac_or_no_accelerator(
        self,
        sample_model_spec: ModelSpec,
        mac_detection_report: DetectionReport,
    ) -> None:
        evaluator = CapabilityEvaluator()
        report = evaluator.evaluate(sample_model_spec, mac_detection_report)

        assert report.verdict == EvaluationVerdict.NO_ACCELERATOR
        assert report.capabilities.amd_gpu_present is False
        assert report.capabilities.baseline_runtime_candidate is None
        assert report.device_assessments == []

        codes = {r.code: r for r in report.reasons}
        assert "NO_AMD_GPU" in codes
        assert codes["NO_AMD_GPU"].severity == EvaluationSeverity.INFO
        assert "ROCM_NOT_DETECTED" in codes
        assert codes["ROCM_NOT_DETECTED"].severity == EvaluationSeverity.INFO

    def test_amd_gpu_with_rocm_missing(
        self,
        sample_model_spec: ModelSpec,
        ready_detection_report: DetectionReport,
    ) -> None:
        env_no_rocm = EnvironmentSpec(
            os="Linux 6.8.0",
            python_version="3.11.0",
            rocm_version=None,
            hip_version=None,
            torch_version="2.4.0",
            torch_hip_available=False,
        )
        report_data = ready_detection_report.model_dump()
        report_data["environment"] = env_no_rocm.model_dump()
        detection = DetectionReport.model_validate(report_data)

        evaluator = CapabilityEvaluator()
        report = evaluator.evaluate(sample_model_spec, detection)

        assert report.verdict == EvaluationVerdict.BLOCKED
        assert report.capabilities.baseline_runtime_candidate is None
        assert any(r.code == "ROCM_NOT_DETECTED" and r.severity == EvaluationSeverity.BLOCKER for r in report.reasons)

    def test_rocm_present_torch_without_hip(
        self,
        sample_model_spec: ModelSpec,
        ready_detection_report: DetectionReport,
    ) -> None:
        env_cpu_torch = EnvironmentSpec(
            os="Linux 6.8.0",
            python_version="3.11.0",
            rocm_version="6.2.0",
            hip_version="6.2.41133",
            torch_version="2.4.0",  # standard CPU/CUDA build, not ROCm
            torch_hip_available=False,
        )
        report_data = ready_detection_report.model_dump()
        report_data["environment"] = env_cpu_torch.model_dump()
        detection = DetectionReport.model_validate(report_data)

        evaluator = CapabilityEvaluator()
        report = evaluator.evaluate(sample_model_spec, detection)

        assert report.verdict == EvaluationVerdict.BLOCKED
        assert any(r.code == "TORCH_NO_HIP" and r.severity == EvaluationSeverity.BLOCKER for r in report.reasons)

    def test_torch_not_installed(
        self,
        sample_model_spec: ModelSpec,
        ready_detection_report: DetectionReport,
    ) -> None:
        env_no_torch = EnvironmentSpec(
            os="Linux 6.8.0",
            python_version="3.11.0",
            rocm_version="6.2.0",
            hip_version="6.2.41133",
            torch_version="not_installed",
            torch_hip_available=False,
        )
        report_data = ready_detection_report.model_dump()
        report_data["environment"] = env_no_torch.model_dump()
        detection = DetectionReport.model_validate(report_data)

        evaluator = CapabilityEvaluator()
        report = evaluator.evaluate(sample_model_spec, detection)

        assert report.verdict == EvaluationVerdict.BLOCKED
        assert any(r.code == "TORCH_NOT_INSTALLED" and r.severity == EvaluationSeverity.BLOCKER for r in report.reasons)

    def test_unknown_amd_gfx_target_not_blocked(
        self,
        sample_model_spec: ModelSpec,
        ready_detection_report: DetectionReport,
    ) -> None:
        # GPU with unlisted custom architecture
        unlisted_gpu = HardwareSpec(
            gpu_present=True,
            gpu_vendor="AMD",
            device_id=0,
            device_name="AMD Custom Accelerator",
            family="Custom",
            gfx_target="gfx1205_custom",
            vram_total_mb=32768,
        )
        detection = DetectionReport(
            environment=ready_detection_report.environment,
            gpus=[unlisted_gpu],
        )

        evaluator = CapabilityEvaluator()
        report = evaluator.evaluate(sample_model_spec, detection)

        # Invariant: UNKNOWN is NEVER automatically converted to BLOCKED!
        assert report.verdict == EvaluationVerdict.UNKNOWN
        assert report.verdict != EvaluationVerdict.BLOCKED
        assert len(report.device_assessments) == 1
        assert report.device_assessments[0].verdict == EvaluationVerdict.UNKNOWN
        assert any(r.code == "UNKNOWN_GFX_TARGET" for r in report.device_assessments[0].reasons)

    def test_unknown_does_not_become_blocked(
        self,
        sample_model_spec: ModelSpec,
        ready_detection_report: DetectionReport,
    ) -> None:
        # Indeterminate architecture target
        indeterminate_gpu = HardwareSpec(
            gpu_present=True,
            gpu_vendor="AMD",
            device_id=0,
            device_name="AMD GPU Indeterminate",
            gfx_target=None,
        )
        detection = DetectionReport(
            environment=ready_detection_report.environment,
            gpus=[indeterminate_gpu],
        )

        evaluator = CapabilityEvaluator()
        report = evaluator.evaluate(sample_model_spec, detection)

        assert report.verdict == EvaluationVerdict.UNKNOWN
        assert report.verdict != EvaluationVerdict.BLOCKED
        assert any(r.code == "GFX_TARGET_INDETERMINATE" for r in report.device_assessments[0].reasons)

    def test_incomplete_optional_model_metadata(
        self,
        ready_detection_report: DetectionReport,
    ) -> None:
        incomplete_model = ModelSpec(
            model_id="Test/IncompleteModel",
            requested_revision="main",
            commit_sha="b" * 40,
            architecture="LlamaForCausalLM",
            parameter_count=None,  # missing optional
            context_length=None,   # missing optional
            default_dtype=None,
            weights_format="safetensors",
        )

        evaluator = CapabilityEvaluator()
        report = evaluator.evaluate(incomplete_model, ready_detection_report)

        assert report.capabilities.model_metadata_complete is False
        assert report.verdict == EvaluationVerdict.READY  # Still READY for baseline attempt
        assert any(r.code == "MODEL_METADATA_INCOMPLETE" and r.severity == EvaluationSeverity.WARNING for r in report.reasons)

    def test_unsupported_weights_format(
        self,
        ready_detection_report: DetectionReport,
    ) -> None:
        gguf_model = ModelSpec(
            model_id="TheBloke/Model-GGUF",
            requested_revision="main",
            commit_sha="c" * 40,
            architecture="LlamaForCausalLM",
            weights_format="gguf",  # GGUF unsupported by HF Transformers baseline runner
        )

        evaluator = CapabilityEvaluator()
        report = evaluator.evaluate(gguf_model, ready_detection_report)

        assert report.verdict == EvaluationVerdict.BLOCKED
        assert any(r.code == "UNSUPPORTED_WEIGHTS_FORMAT" and r.severity == EvaluationSeverity.BLOCKER for r in report.reasons)

    def test_remote_code_required(
        self,
        ready_detection_report: DetectionReport,
    ) -> None:
        remote_code_model = ModelSpec(
            model_id="Custom/RemoteCodeModel",
            requested_revision="main",
            commit_sha="d" * 40,
            architecture="CustomRemoteModel",
            remote_code_required=True,
        )

        evaluator = CapabilityEvaluator()
        report = evaluator.evaluate(remote_code_model, ready_detection_report)

        assert report.verdict == EvaluationVerdict.BLOCKED
        assert any(r.code == "REMOTE_CODE_REQUIRED" and r.severity == EvaluationSeverity.BLOCKER for r in report.reasons)

    def test_multiple_gpus_individual_evaluation(
        self,
        sample_model_spec: ModelSpec,
        ready_detection_report: DetectionReport,
    ) -> None:
        gpu_0 = HardwareSpec(
            gpu_present=True,
            gpu_vendor="AMD",
            device_id=0,
            device_name="AMD Radeon RX 7900 XTX",
            family="Radeon",
            gfx_target="gfx1100",  # Known
        )
        gpu_1 = HardwareSpec(
            gpu_present=True,
            gpu_vendor="AMD",
            device_id=1,
            device_name="AMD Custom Prototype",
            family="Instinct",
            gfx_target="gfx9999_unlisted",  # Unlisted
        )
        detection = DetectionReport(
            environment=ready_detection_report.environment,
            gpus=[gpu_0, gpu_1],
        )

        evaluator = CapabilityEvaluator()
        report = evaluator.evaluate(sample_model_spec, detection)

        # Device-level evaluation
        assert len(report.device_assessments) == 2
        assert report.device_assessments[0].verdict == EvaluationVerdict.READY
        assert report.device_assessments[0].device_id == 0
        assert report.device_assessments[1].verdict == EvaluationVerdict.UNKNOWN
        assert report.device_assessments[1].device_id == 1

        # System-level evaluation: at least one device is READY -> overall READY
        assert report.verdict == EvaluationVerdict.READY
        assert report.capabilities.baseline_runtime_candidate == "pytorch_transformers_hip"

    def test_multiple_gpus_all_unknown(
        self,
        sample_model_spec: ModelSpec,
        ready_detection_report: DetectionReport,
    ) -> None:
        gpu_0 = HardwareSpec(
            gpu_present=True,
            gpu_vendor="AMD",
            device_id=0,
            device_name="AMD Prototype 1",
            gfx_target="gfx1205_custom",
        )
        gpu_1 = HardwareSpec(
            gpu_present=True,
            gpu_vendor="AMD",
            device_id=1,
            device_name="AMD Prototype 2",
            gfx_target="gfx9999_custom",
        )
        detection = DetectionReport(
            environment=ready_detection_report.environment,
            gpus=[gpu_0, gpu_1],
        )

        evaluator = CapabilityEvaluator()
        report = evaluator.evaluate(sample_model_spec, detection)

        assert report.device_assessments[0].verdict == EvaluationVerdict.UNKNOWN
        assert report.device_assessments[1].verdict == EvaluationVerdict.UNKNOWN
        assert report.verdict == EvaluationVerdict.UNKNOWN

    def test_structured_reason_codes(
        self,
        sample_model_spec: ModelSpec,
        ready_detection_report: DetectionReport,
    ) -> None:
        evaluator = CapabilityEvaluator()
        report = evaluator.evaluate(sample_model_spec, ready_detection_report)

        for reason in report.reasons:
            assert isinstance(reason, EvaluationReason)
            assert isinstance(reason.code, str) and len(reason.code) > 0
            assert isinstance(reason.severity, EvaluationSeverity)
            assert isinstance(reason.message, str) and len(reason.message) > 0
            assert isinstance(reason.evidence, dict)

    def test_evaluator_never_calls_inference_or_downloads_weights(
        self,
        sample_model_spec: ModelSpec,
        ready_detection_report: DetectionReport,
    ) -> None:
        # Patch any possible torch or download calls
        with patch("subprocess.run") as mock_subproc, patch("urllib.request.urlopen") as mock_url:
            evaluator = CapabilityEvaluator()
            report = evaluator.evaluate(sample_model_spec, ready_detection_report)

            mock_subproc.assert_not_called()
            mock_url.assert_not_called()
            assert report.verdict == EvaluationVerdict.READY


class TestCliCheckCommand:
    @patch("rocmhub.cli.main.HuggingFaceModelSource")
    @patch("rocmhub.cli.main.SystemObserver")
    def test_cli_check_json_output(
        self,
        mock_observer_cls: MagicMock,
        mock_source_cls: MagicMock,
        sample_model_spec: ModelSpec,
        mac_detection_report: DetectionReport,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_source_cls.return_value.source_name = "huggingface"
        mock_source_cls.return_value.resolve_revision.return_value = sample_model_spec.commit_sha
        mock_source_cls.return_value.get_repository_metadata.return_value = MagicMock(
            files=["config.json", "model.safetensors"],
            safetensors_metadata={"total": 494032768},
        )
        mock_source_cls.return_value.fetch_metadata_file.return_value = json.dumps(
            {"architectures": ["Qwen2ForCausalLM"], "max_position_embeddings": 32768}
        )
        mock_observer_cls.return_value.observe.return_value = mac_detection_report

        exit_code = main(["check", "Qwen/Qwen2.5-0.5B-Instruct", "--json"])
        # On Mac / no accelerator, exit code must be 2
        assert exit_code == 2

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["verdict"] == "NO_ACCELERATOR"
        assert data["model"]["model_id"] == "Qwen/Qwen2.5-0.5B-Instruct"
        assert data["capabilities"]["amd_gpu_present"] is False
        assert data["capabilities"]["baseline_runtime_candidate"] is None

    @patch("rocmhub.cli.main.HuggingFaceModelSource")
    @patch("rocmhub.cli.main.SystemObserver")
    def test_cli_check_human_readable_ready(
        self,
        mock_observer_cls: MagicMock,
        mock_source_cls: MagicMock,
        sample_model_spec: ModelSpec,
        ready_detection_report: DetectionReport,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_source_cls.return_value.source_name = "huggingface"
        mock_source_cls.return_value.resolve_revision.return_value = sample_model_spec.commit_sha
        mock_source_cls.return_value.get_repository_metadata.return_value = MagicMock(
            files=["config.json", "model.safetensors"],
            safetensors_metadata={"total": 494032768},
        )
        mock_source_cls.return_value.fetch_metadata_file.return_value = json.dumps(
            {"architectures": ["Qwen2ForCausalLM"], "max_position_embeddings": 32768}
        )
        mock_observer_cls.return_value.observe.return_value = ready_detection_report

        exit_code = main(["check", "Qwen/Qwen2.5-0.5B-Instruct"])
        assert exit_code == 0

        captured = capsys.readouterr()
        output = captured.out
        assert "Model:" in output
        assert "Qwen/Qwen2.5-0.5B-Instruct" in output
        assert "Environment:" in output
        assert "ROCm:    6.2.0" in output
        assert "HIP:     6.2.41133" in output
        assert "GPU 0:" in output
        assert "AMD Radeon RX 7900 XTX" in output
        assert "gfx: gfx1100" in output
        assert "Baseline candidate:" in output
        assert "pytorch_transformers_hip" in output
        assert "Verdict:" in output
        assert "READY" in output
        assert "[OK] AMD_GPU_PRESENT" in output
        assert "[OK] ROCM_DETECTED" in output
        assert "[OK] TORCH_HIP_AVAILABLE" in output

    @patch("rocmhub.cli.main.HuggingFaceModelSource")
    @patch("rocmhub.cli.main.SystemObserver")
    def test_cli_check_human_readable_mac(
        self,
        mock_observer_cls: MagicMock,
        mock_source_cls: MagicMock,
        sample_model_spec: ModelSpec,
        mac_detection_report: DetectionReport,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_source_cls.return_value.source_name = "huggingface"
        mock_source_cls.return_value.resolve_revision.return_value = sample_model_spec.commit_sha
        mock_source_cls.return_value.get_repository_metadata.return_value = MagicMock(
            files=["config.json", "model.safetensors"],
            safetensors_metadata={"total": 494032768},
        )
        mock_source_cls.return_value.fetch_metadata_file.return_value = json.dumps(
            {"architectures": ["Qwen2ForCausalLM"], "max_position_embeddings": 32768}
        )
        mock_observer_cls.return_value.observe.return_value = mac_detection_report

        exit_code = main(["check", "Qwen/Qwen2.5-0.5B-Instruct"])
        assert exit_code == 2

        captured = capsys.readouterr()
        output = captured.out
        assert "Detected GPUs: 0" in output
        assert "Baseline candidate:\n  none" in output
        assert "Verdict:\n  NO_ACCELERATOR" in output
        assert "[INFO] NO_AMD_GPU" in output
        assert "[INFO] ROCM_NOT_DETECTED" in output

    @patch("rocmhub.cli.main.HuggingFaceModelSource")
    @patch("rocmhub.cli.main.SystemObserver")
    def test_cli_exit_codes(
        self,
        mock_observer_cls: MagicMock,
        mock_source_cls: MagicMock,
        sample_model_spec: ModelSpec,
        ready_detection_report: DetectionReport,
        mac_detection_report: DetectionReport,
    ) -> None:
        # Mock model inspect
        mock_source_cls.return_value.source_name = "huggingface"
        mock_source_cls.return_value.resolve_revision.return_value = sample_model_spec.commit_sha
        mock_source_cls.return_value.get_repository_metadata.return_value = MagicMock(
            files=["config.json", "model.safetensors"],
            safetensors_metadata={"total": 494032768},
        )
        mock_source_cls.return_value.fetch_metadata_file.return_value = json.dumps(
            {"architectures": ["Qwen2ForCausalLM"], "max_position_embeddings": 32768}
        )

        # 0 = READY
        mock_observer_cls.return_value.observe.return_value = ready_detection_report
        assert main(["check", "Qwen/Qwen2.5-0.5B-Instruct"]) == 0

        # 2 = NO_ACCELERATOR
        mock_observer_cls.return_value.observe.return_value = mac_detection_report
        assert main(["check", "Qwen/Qwen2.5-0.5B-Instruct"]) == 2

        # 3 = BLOCKED (e.g. no ROCm)
        env_blocked = EnvironmentSpec(
            os="Linux 6.8.0",
            python_version="3.11.0",
            rocm_version=None,
            torch_version="2.4.0",
            torch_hip_available=False,
        )
        blocked_report = DetectionReport(
            environment=env_blocked,
            gpus=[ready_detection_report.gpus[0]],
        )
        mock_observer_cls.return_value.observe.return_value = blocked_report
        assert main(["check", "Qwen/Qwen2.5-0.5B-Instruct"]) == 3

        # 4 = UNKNOWN (e.g. unknown gfx target)
        unknown_gpu = HardwareSpec(
            gpu_present=True,
            gpu_vendor="AMD",
            device_id=0,
            device_name="AMD Custom",
            gfx_target="gfx1205_custom",
        )
        unknown_report = DetectionReport(
            environment=ready_detection_report.environment,
            gpus=[unknown_gpu],
        )
        mock_observer_cls.return_value.observe.return_value = unknown_report
        assert main(["check", "Qwen/Qwen2.5-0.5B-Instruct"]) == 4

        # 1 = Error (e.g. model not found)
        from rocmhub.core.errors import ModelNotFoundError
        mock_source_cls.return_value.resolve_revision.side_effect = ModelNotFoundError("Repo not found")
        assert main(["check", "NonExistent/Model"]) == 1
