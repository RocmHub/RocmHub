"""Unit tests for BaseRunner, HuggingFaceRunner, and rocmhub run CLI."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from rocmhub.cli.main import main
from rocmhub.core.errors import (
    DeviceNotAvailableError,
    GenerationError,
    ModelLoadError,
    RunnerNotReadyError,
    UnsupportedModelTypeError,
    UnsupportedPrecisionError,
)
from rocmhub.core.types import (
    CapabilityReport,
    DetectionReport,
    DeviceCapabilityAssessment,
    EnvironmentSpec,
    EvaluationVerdict,
    ExecutionStatus,
    HardwareSpec,
    ModelSpec,
    RunResult,
    SystemCapabilities,
)
from rocmhub.runners.hf_runner import HuggingFaceRunner


class MockTensor:
    """Lightweight mock tensor enabling offline tests without requiring PyTorch installation."""

    def __init__(self, data: list[Any]) -> None:
        self._data = data
        if data and isinstance(data[0], list):
            self.shape = (len(data), len(data[0]))
        elif data:
            self.shape = (len(data),)
        else:
            self.shape = (0,)

    def to(self, device: Any) -> MockTensor:
        return self

    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, tuple):
            row_idx, col_slice = item
            if isinstance(col_slice, slice):
                return MockTensor(self._data[row_idx][col_slice])
            return self._data[row_idx][col_slice]
        if isinstance(item, slice):
            return MockTensor(self._data[item])
        return self._data[item]

    def __len__(self) -> int:
        return len(self._data)


@pytest.fixture
def sample_causal_model_spec() -> ModelSpec:
    return ModelSpec(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        requested_revision="main",
        commit_sha="a" * 40,
        architecture="Qwen2ForCausalLM",
        parameter_count=494_032_768,
        context_length=32768,
        default_dtype="bfloat16",
        weights_format="safetensors",
    )


@pytest.fixture
def mock_tokenizer() -> MagicMock:
    tok = MagicMock()
    # Mock tokenization: prompt -> 3 tokens
    input_ids = MockTensor([[101, 102, 103]])
    tok.return_value = {"input_ids": input_ids}
    # Mock decoding
    tok.decode.return_value = "This is a deterministic generated response."
    return tok


@pytest.fixture
def mock_causal_model() -> MagicMock:
    model = MagicMock()
    model.to.return_value = model
    model.eval.return_value = model
    # Mock generation: 3 prompt tokens + 4 new tokens = 7 total tokens
    output_ids = MockTensor([[101, 102, 103, 201, 202, 203, 204]])
    model.generate.return_value = output_ids
    return model


class TestHuggingFaceRunner:
    def test_lifecycle_load_generate_unload(
        self,
        sample_causal_model_spec: ModelSpec,
        mock_causal_model: MagicMock,
        mock_tokenizer: MagicMock,
    ) -> None:
        model_loader = MagicMock(return_value=mock_causal_model)
        tokenizer_loader = MagicMock(return_value=mock_tokenizer)

        runner = HuggingFaceRunner(
            model_loader=model_loader,
            tokenizer_loader=tokenizer_loader,
            device_override="cpu",
        )

        assert not runner.is_loaded

        # 1. Load
        runner.load(sample_causal_model_spec, device_id=0, precision="fp16")
        assert runner.is_loaded

        # 2. Generate
        result = runner.generate(prompt="Hello", max_new_tokens=16)
        assert isinstance(result, RunResult)
        assert result.status == ExecutionStatus.SUCCESS
        assert result.model_id == sample_causal_model_spec.model_id
        assert result.model_revision == sample_causal_model_spec.commit_sha
        assert result.device_id == 0
        assert result.precision == "fp16"
        assert result.prompt == "Hello"
        assert result.generated_text == "This is a deterministic generated response."
        assert result.input_tokens == 3
        assert result.generated_tokens == 4  # 7 total - 3 prompt = 4
        assert result.generation_params == {"max_new_tokens": 16, "do_sample": False}

        # 3. Unload
        runner.unload()
        assert not runner.is_loaded

    def test_generate_before_load_raises_runner_not_ready(self) -> None:
        runner = HuggingFaceRunner()
        assert not runner.is_loaded

        with pytest.raises(RunnerNotReadyError) as exc_info:
            runner.generate(prompt="Hello")

        assert exc_info.value.error_code == "RUNNER_NOT_READY"

    def test_immutable_commit_sha_and_trust_remote_code_false(
        self,
        sample_causal_model_spec: ModelSpec,
        mock_causal_model: MagicMock,
        mock_tokenizer: MagicMock,
    ) -> None:
        model_loader = MagicMock(return_value=mock_causal_model)
        tokenizer_loader = MagicMock(return_value=mock_tokenizer)

        runner = HuggingFaceRunner(
            model_loader=model_loader,
            tokenizer_loader=tokenizer_loader,
            device_override="cpu",
        )
        runner.load(sample_causal_model_spec, device_id=0, precision="fp16")

        # Invariant: revision MUST be the immutable 40-char SHA, NOT mutable "main"
        assert sample_causal_model_spec.commit_sha == "a" * 40
        assert model_loader.call_args[0][0] == sample_causal_model_spec.model_id
        assert model_loader.call_args[1]["revision"] == sample_causal_model_spec.commit_sha
        assert model_loader.call_args[1]["trust_remote_code"] is False

        tokenizer_loader.assert_called_once_with(
            sample_causal_model_spec.model_id,
            revision=sample_causal_model_spec.commit_sha,
            trust_remote_code=False,
        )

    def test_device_map_auto_never_used(
        self,
        sample_causal_model_spec: ModelSpec,
        mock_causal_model: MagicMock,
        mock_tokenizer: MagicMock,
    ) -> None:
        model_loader = MagicMock(return_value=mock_causal_model)
        tokenizer_loader = MagicMock(return_value=mock_tokenizer)

        runner = HuggingFaceRunner(
            model_loader=model_loader,
            tokenizer_loader=tokenizer_loader,
            device_override="cpu",
        )
        runner.load(sample_causal_model_spec, device_id=0, precision="fp16")

        # Invariant: device_map must NOT be in model_loader kwargs
        call_kwargs = model_loader.call_args[1]
        assert "device_map" not in call_kwargs

    def test_precision_mappings(
        self,
        sample_causal_model_spec: ModelSpec,
        mock_causal_model: MagicMock,
        mock_tokenizer: MagicMock,
    ) -> None:
        for prec_str in ["fp32", "fp16", "bf16"]:
            model_loader = MagicMock(return_value=mock_causal_model)
            tokenizer_loader = MagicMock(return_value=mock_tokenizer)

            runner = HuggingFaceRunner(
                model_loader=model_loader,
                tokenizer_loader=tokenizer_loader,
                device_override="cpu",
            )
            runner.load(sample_causal_model_spec, device_id=0, precision=prec_str)

            assert "torch_dtype" in model_loader.call_args[1]
            runner.unload()

    def test_unsupported_precision_raises_error(
        self,
        sample_causal_model_spec: ModelSpec,
    ) -> None:
        runner = HuggingFaceRunner(device_override="cpu")

        with pytest.raises(UnsupportedPrecisionError) as exc_info:
            runner.load(sample_causal_model_spec, device_id=0, precision="int8")

        assert exc_info.value.error_code == "UNSUPPORTED_PRECISION"

    def test_unsupported_model_type_raises_error(self) -> None:
        vision_spec = ModelSpec(
            model_id="google/vit-base-patch16-224",
            requested_revision="main",
            commit_sha="e" * 40,
            architecture="ViTForImageClassification",  # Not a causal LM
        )
        runner = HuggingFaceRunner(device_override="cpu")

        with pytest.raises(UnsupportedModelTypeError) as exc_info:
            runner.load(vision_spec, device_id=0)

        assert exc_info.value.error_code == "UNSUPPORTED_MODEL_TYPE"

    def test_invalid_device_raises_device_not_available(
        self,
        sample_causal_model_spec: ModelSpec,
    ) -> None:
        runner = HuggingFaceRunner()  # No override -> uses real cuda check

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1

        with patch("rocmhub.runners.hf_runner.torch", mock_torch):
            # Asking for device 2 when only 1 exists
            with pytest.raises(DeviceNotAvailableError) as exc_info:
                runner.load(sample_causal_model_spec, device_id=2)

            assert exc_info.value.error_code == "DEVICE_NOT_AVAILABLE"

    def test_model_load_failure_raises_model_load_error(
        self,
        sample_causal_model_spec: ModelSpec,
    ) -> None:
        def failing_loader(*args: object, **kwargs: object) -> None:
            raise RuntimeError("Corrupted checkpoint tensor")

        runner = HuggingFaceRunner(
            model_loader=failing_loader,
            tokenizer_loader=MagicMock(),
            device_override="cpu",
        )

        with pytest.raises(ModelLoadError) as exc_info:
            runner.load(sample_causal_model_spec, device_id=0)

        assert exc_info.value.error_code == "MODEL_LOAD_FAILED"

    def test_generation_failure_raises_generation_error(
        self,
        sample_causal_model_spec: ModelSpec,
        mock_tokenizer: MagicMock,
    ) -> None:
        failing_model = MagicMock()
        failing_model.to.return_value = failing_model
        failing_model.eval.return_value = failing_model
        failing_model.generate.side_effect = RuntimeError("CUDA out of memory during decode")

        runner = HuggingFaceRunner(
            model_loader=MagicMock(return_value=failing_model),
            tokenizer_loader=MagicMock(return_value=mock_tokenizer),
            device_override="cpu",
        )
        runner.load(sample_causal_model_spec, device_id=0)

        with pytest.raises(GenerationError) as exc_info:
            runner.generate(prompt="Hello")

        assert exc_info.value.error_code == "GENERATION_FAILED"

    def test_memory_cleanup_on_unload(
        self,
        sample_causal_model_spec: ModelSpec,
        mock_causal_model: MagicMock,
        mock_tokenizer: MagicMock,
    ) -> None:
        runner = HuggingFaceRunner(
            model_loader=MagicMock(return_value=mock_causal_model),
            tokenizer_loader=MagicMock(return_value=mock_tokenizer),
            device_override="cpu",
        )
        runner.load(sample_causal_model_spec, device_id=0)
        assert runner.is_loaded

        with patch("gc.collect") as mock_gc:
            runner.unload()

            assert not runner.is_loaded
            mock_gc.assert_called_once()

    def test_supports_method(
        self,
        sample_causal_model_spec: ModelSpec,
    ) -> None:
        runner = HuggingFaceRunner()

        ready_assessment = DeviceCapabilityAssessment(
            device_id=0,
            device_name="AMD Radeon RX 7900 XTX",
            gfx_target="gfx1100",
            verdict=EvaluationVerdict.READY,
        )
        unknown_assessment = DeviceCapabilityAssessment(
            device_id=1,
            device_name="AMD Prototype",
            gfx_target="gfx9999",
            verdict=EvaluationVerdict.UNKNOWN,
        )

        ready_report = CapabilityReport(
            model=sample_causal_model_spec,
            environment=EnvironmentSpec(
                os="Linux",
                python_version="3.11",
                rocm_version="6.2.0",
                torch_version="2.4.0+rocm6.2",
                torch_hip_available=True,
            ),
            hardware=[],
            device_assessments=[ready_assessment, unknown_assessment],
            verdict=EvaluationVerdict.READY,
            capabilities=SystemCapabilities(
                amd_gpu_present=True,
                rocm_detected=True,
                torch_hip_available=True,
                baseline_runtime_candidate="pytorch_transformers_hip",
            ),
        )

        # Device 0 is READY -> supported
        assert runner.supports(ready_report, device_id=0) is True
        # Device 1 is UNKNOWN -> not supported
        assert runner.supports(ready_report, device_id=1) is False
        # Device 2 not in assessments -> not supported
        assert runner.supports(ready_report, device_id=2) is False

        # Non-READY report
        no_accel_report = CapabilityReport(
            model=sample_causal_model_spec,
            environment=EnvironmentSpec(
                os="Darwin",
                python_version="3.9",
                torch_version="not_installed",
            ),
            hardware=[],
            device_assessments=[],
            verdict=EvaluationVerdict.NO_ACCELERATOR,
            capabilities=SystemCapabilities(),
        )
        assert runner.supports(no_accel_report, device_id=0) is False


class TestCliRunCommand:
    @patch("rocmhub.cli.main.HuggingFaceModelSource")
    @patch("rocmhub.cli.main.SystemObserver")
    def test_cli_run_stops_on_no_accelerator_without_loading_or_downloading(
        self,
        mock_observer_cls: MagicMock,
        mock_source_cls: MagicMock,
        sample_causal_model_spec: ModelSpec,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_source_cls.return_value.source_name = "huggingface"
        mock_source_cls.return_value.resolve_revision.return_value = sample_causal_model_spec.commit_sha
        mock_source_cls.return_value.get_repository_metadata.return_value = MagicMock(
            files=["config.json", "model.safetensors"],
            safetensors_metadata={"total": 494032768},
        )
        mock_source_cls.return_value.fetch_metadata_file.return_value = json.dumps(
            {"architectures": ["Qwen2ForCausalLM"], "max_position_embeddings": 32768}
        )

        # Mac environment -> NO_ACCELERATOR
        mac_env = EnvironmentSpec(
            os="Darwin 25.5.0",
            python_version="3.9.6",
            rocm_version=None,
            torch_version="not_installed",
        )
        mock_observer_cls.return_value.observe.return_value = DetectionReport(
            environment=mac_env,
            gpus=[],
        )

        with patch("rocmhub.cli.main.HuggingFaceRunner") as mock_runner_cls:
            exit_code = main(["run", "Qwen/Qwen2.5-0.5B-Instruct", "--prompt", "Hello"])

            # Must exit with code 2 (NO_ACCELERATOR)
            assert exit_code == 2
            # Invariant: Runner must NEVER be instantiated or loaded on non-accelerator host!
            mock_runner_cls.assert_not_called()

        captured = capsys.readouterr()
        assert "Preflight Check: NO_ACCELERATOR" in captured.out
        assert "Model weights were NOT downloaded and inference was NOT executed." in captured.out

    @patch("rocmhub.cli.main.HuggingFaceModelSource")
    @patch("rocmhub.cli.main.SystemObserver")
    def test_cli_run_stops_on_blocked_preflight(
        self,
        mock_observer_cls: MagicMock,
        mock_source_cls: MagicMock,
        sample_causal_model_spec: ModelSpec,
    ) -> None:
        mock_source_cls.return_value.source_name = "huggingface"
        mock_source_cls.return_value.resolve_revision.return_value = sample_causal_model_spec.commit_sha
        mock_source_cls.return_value.get_repository_metadata.return_value = MagicMock(
            files=["config.json"],
            safetensors_metadata={"total": 494032768},
        )
        mock_source_cls.return_value.fetch_metadata_file.return_value = json.dumps(
            {"architectures": ["Qwen2ForCausalLM"]}
        )

        # GPU present, but ROCm missing -> BLOCKED
        blocked_env = EnvironmentSpec(
            os="Linux 6.8.0",
            python_version="3.11",
            rocm_version=None,
            torch_version="2.4.0",
            torch_hip_available=False,
        )
        gpu = HardwareSpec(gpu_present=True, gpu_vendor="AMD", device_name="Radeon")
        mock_observer_cls.return_value.observe.return_value = DetectionReport(
            environment=blocked_env,
            gpus=[gpu],
        )

        with patch("rocmhub.cli.main.HuggingFaceRunner") as mock_runner_cls:
            exit_code = main(["run", "Qwen/Qwen2.5-0.5B-Instruct"])
            assert exit_code == 3  # BLOCKED
            mock_runner_cls.assert_not_called()

    @patch("rocmhub.cli.main.HuggingFaceModelSource")
    @patch("rocmhub.cli.main.SystemObserver")
    def test_cli_run_json_output_on_skipped(
        self,
        mock_observer_cls: MagicMock,
        mock_source_cls: MagicMock,
        sample_causal_model_spec: ModelSpec,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_source_cls.return_value.source_name = "huggingface"
        mock_source_cls.return_value.resolve_revision.return_value = sample_causal_model_spec.commit_sha
        mock_source_cls.return_value.get_repository_metadata.return_value = MagicMock(
            files=["config.json"],
            safetensors_metadata={"total": 494032768},
        )
        mock_source_cls.return_value.fetch_metadata_file.return_value = json.dumps(
            {"architectures": ["Qwen2ForCausalLM"]}
        )

        mac_env = EnvironmentSpec(os="Darwin", python_version="3.9", torch_version="not_installed")
        mock_observer_cls.return_value.observe.return_value = DetectionReport(environment=mac_env, gpus=[])

        exit_code = main(["run", "Qwen/Qwen2.5-0.5B-Instruct", "--json"])
        assert exit_code == 2

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["status"] == "SKIPPED"
        assert data["model_id"] == "Qwen/Qwen2.5-0.5B-Instruct"
        assert data["generated_text"] is None
        assert data["input_tokens"] is None
        assert "NO_ACCELERATOR" in data["error"]

    @patch("rocmhub.cli.main.HuggingFaceModelSource")
    @patch("rocmhub.cli.main.SystemObserver")
    @patch("rocmhub.cli.main.HuggingFaceRunner")
    def test_cli_run_ready_executes_runner_and_unloads(
        self,
        mock_runner_cls: MagicMock,
        mock_observer_cls: MagicMock,
        mock_source_cls: MagicMock,
        sample_causal_model_spec: ModelSpec,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_source_cls.return_value.source_name = "huggingface"
        mock_source_cls.return_value.resolve_revision.return_value = sample_causal_model_spec.commit_sha
        mock_source_cls.return_value.get_repository_metadata.return_value = MagicMock(
            files=["config.json"],
            safetensors_metadata={"total": 494032768},
        )
        mock_source_cls.return_value.fetch_metadata_file.return_value = json.dumps(
            {"architectures": ["Qwen2ForCausalLM"]}
        )

        ready_env = EnvironmentSpec(
            os="Linux 6.8.0",
            python_version="3.11",
            rocm_version="6.2.0",
            hip_version="6.2.41133",
            torch_version="2.4.0+rocm6.2",
            torch_hip_available=True,
        )
        ready_gpu = HardwareSpec(
            gpu_present=True,
            gpu_vendor="AMD",
            device_id=0,
            device_name="AMD Radeon RX 7900 XTX",
            gfx_target="gfx1100",
            vram_total_mb=24560,
        )
        mock_observer_cls.return_value.observe.return_value = DetectionReport(
            environment=ready_env,
            gpus=[ready_gpu],
        )

        mock_instance = MagicMock()
        mock_runner_cls.return_value = mock_instance
        mock_instance.generate.return_value = RunResult(
            status=ExecutionStatus.SUCCESS,
            runtime_name="pytorch_transformers_hip",
            model_id=sample_causal_model_spec.model_id,
            model_revision=sample_causal_model_spec.commit_sha,
            device_id=0,
            precision="fp16",
            prompt="Hello",
            generated_text="I am ready.",
            input_tokens=2,
            generated_tokens=4,
        )

        exit_code = main(["run", "Qwen/Qwen2.5-0.5B-Instruct", "--prompt", "Hello", "--precision", "fp16"])
        assert exit_code == 0

        mock_instance.load.assert_called_once()
        mock_instance.generate.assert_called_once_with(prompt="Hello", max_new_tokens=16)
        mock_instance.unload.assert_called_once()

        captured = capsys.readouterr()
        assert "Model:" in captured.out
        assert "Status: SUCCESS" in captured.out
        assert "I am ready." in captured.out
