"""Unit tests for ROCmHub Model Forge (Phase 10)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rocmhub.cli.main import main
from rocmhub.core.errors import (
    AuthRequiredError,
    BuildConflictError,
    BuildExecutionError,
    InsufficientDiskSpaceError,
    ModelMaterializationError,
    ModelNotFoundError,
    SecretDetectedError,
    UnsupportedModelArchitectureError,
    UnsupportedPrecisionError,
)
from rocmhub.core.types import (
    DetectionReport,
    EnvironmentSpec,
    HardwareSpec,
    ModelSpec,
)
from rocmhub.forge.base import (
    BuildStatus,
    ForgePlan,
    MaterializationMode,
    MaterializedModel,
    StepStatus,
)
from rocmhub.forge.executor import ForgeExecutor
from rocmhub.forge.manifest import (
    BuildManifest,
    assert_no_secrets,
    sanitize_secrets_in_obj,
    write_manifest,
)
from rocmhub.forge.materializer import ModelMaterializer
from rocmhub.forge.planner import ForgePlanner
from rocmhub.forge.recipes import (
    TESTED_CAUSAL_LM_ARCHITECTURES,
    PyTorchTransformersHipRecipe,
    get_recipe,
)

SAMPLE_COMMIT_SHA = "4828b8a07cbdf69ff9f074d0089f2d593006a88b"


@pytest.fixture
def sample_qwen_spec() -> ModelSpec:
    return ModelSpec(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        requested_revision="main",
        commit_sha=SAMPLE_COMMIT_SHA,
        architecture="Qwen2ForCausalLM",
        parameter_count=494_032_768,
        context_length=32768,
        default_dtype="bfloat16",
        weights_format="safetensors",
        remote_code_required=False,
    )


@pytest.fixture
def sample_llama_spec() -> ModelSpec:
    return ModelSpec(
        model_id="meta-llama/Llama-3.2-1B",
        requested_revision="main",
        commit_sha="1111222233334444555566667777888899990000",
        architecture="LlamaForCausalLM",
        parameter_count=1_235_814_400,
        context_length=131072,
        default_dtype="bfloat16",
        weights_format="safetensors",
        remote_code_required=False,
    )


@pytest.fixture
def non_amd_detection_report() -> DetectionReport:
    return DetectionReport(
        environment=EnvironmentSpec(
            os="Darwin",
            python_version="3.9.6",
            torch_version="2.2.0",
        ),
        gpus=[],
    )


@pytest.fixture
def amd_detection_report() -> DetectionReport:
    return DetectionReport(
        environment=EnvironmentSpec(
            os="Linux",
            python_version="3.10.0",
            torch_version="2.3.0+rocm6.0",
            rocm_version="6.0.0",
            hip_version="6.0.0",
        ),
        gpus=[
            HardwareSpec(
                gpu_present=True,
                gpu_vendor="AMD",
                device_id=0,
                device_name="AMD Radeon RX 7900 XTX",
                gfx_target="gfx1100",
                vram_total_mb=24576,
            )
        ],
    )


# ==============================================================================
# Recipe Tests
# ==============================================================================


class TestForgeRecipes:
    def test_get_recipe_success(self) -> None:
        recipe = get_recipe("pytorch_transformers_hip")
        assert isinstance(recipe, PyTorchTransformersHipRecipe)
        assert recipe.recipe_id == "pytorch_transformers_hip"
        assert recipe.version == "1.0.0"

    def test_get_recipe_not_found_raises(self) -> None:
        with pytest.raises(UnsupportedModelArchitectureError) as exc_info:
            get_recipe("nonexistent_recipe")
        assert "nonexistent_recipe" in str(exc_info.value)

    def test_recipe_matches_causal_lm(self, sample_qwen_spec: ModelSpec, sample_llama_spec: ModelSpec) -> None:
        recipe = PyTorchTransformersHipRecipe()
        assert recipe.matches(sample_qwen_spec) is True
        assert recipe.matches(sample_llama_spec) is True

    def test_recipe_rejects_non_causal_lm(self) -> None:
        recipe = PyTorchTransformersHipRecipe()
        bert_spec = ModelSpec(
            model_id="google-bert/bert-base-uncased",
            requested_revision="main",
            commit_sha="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            architecture="BertForMaskedLM",
        )
        assert recipe.matches(bert_spec) is False
        with pytest.raises(UnsupportedModelArchitectureError):
            recipe.validate(bert_spec, "fp16")

    def test_recipe_rejects_unsupported_precision(self, sample_qwen_spec: ModelSpec) -> None:
        recipe = PyTorchTransformersHipRecipe()
        with pytest.raises(UnsupportedPrecisionError):
            recipe.validate(sample_qwen_spec, "int4")

    def test_recipe_supported_precisions(self, sample_qwen_spec: ModelSpec) -> None:
        recipe = PyTorchTransformersHipRecipe()
        for p in ["fp16", "bf16", "fp32"]:
            recipe.validate(sample_qwen_spec, p)

    def test_generate_runtime_config(self, sample_qwen_spec: ModelSpec) -> None:
        recipe = PyTorchTransformersHipRecipe()
        cfg = recipe.generate_runtime_config(sample_qwen_spec, "bf16", target_gpu="gfx1100")
        assert cfg["model_id"] == "Qwen/Qwen2.5-0.5B-Instruct"
        assert cfg["revision"] == SAMPLE_COMMIT_SHA
        assert cfg["precision"] == "bf16"
        assert cfg["torch_dtype"] == "bfloat16"
        assert cfg["target_gpu"] == "gfx1100"
        assert cfg["trust_remote_code"] is False

    def test_generate_launch_script(self, sample_qwen_spec: ModelSpec) -> None:
        recipe = PyTorchTransformersHipRecipe()
        script = recipe.generate_launch_script(sample_qwen_spec, "fp16", target_gpu="gfx90a")
        assert "Qwen/Qwen2.5-0.5B-Instruct" in script
        assert SAMPLE_COMMIT_SHA in script
        assert "torch.float16" in script
        assert "AutoModelForCausalLM" in script

    def test_tested_causal_lm_architectures_whitelist(self) -> None:
        recipe = PyTorchTransformersHipRecipe()
        for arch in TESTED_CAUSAL_LM_ARCHITECTURES:
            spec = ModelSpec(
                model_id=f"test/{arch}",
                requested_revision="main",
                commit_sha=SAMPLE_COMMIT_SHA,
                architecture=arch,
            )
            assert recipe.matches(spec) is True

    def test_untested_causal_lm_architecture_rejected(self) -> None:
        recipe = PyTorchTransformersHipRecipe()
        spec = ModelSpec(
            model_id="test/untested-model",
            requested_revision="main",
            commit_sha=SAMPLE_COMMIT_SHA,
            architecture="UntestedCustomForCausalLM",
        )
        assert recipe.matches(spec) is False
        with pytest.raises(UnsupportedModelArchitectureError) as exc_info:
            recipe.validate(spec, "fp16")
        assert exc_info.value.details.get("reason_code") == "UNTESTED_ARCHITECTURE"


# ==============================================================================
# Planner Tests
# ==============================================================================


class TestForgePlanner:
    def test_create_plan_deterministic(
        self, sample_qwen_spec: ModelSpec, non_amd_detection_report: DetectionReport
    ) -> None:
        mock_inspector = MagicMock()
        mock_inspector.inspect.return_value = sample_qwen_spec

        mock_observer = MagicMock()
        mock_observer.observe.return_value = non_amd_detection_report

        planner = ForgePlanner(
            model_inspector=mock_inspector,
            system_observer=mock_observer,
        )

        plan1 = planner.create_plan(
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
            precision="fp16",
            output_dir="/tmp/build1",
        )
        plan2 = planner.create_plan(
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
            precision="fp16",
            output_dir="/tmp/build1",
        )

        assert plan1.plan_id == plan2.plan_id
        assert plan1.revision == SAMPLE_COMMIT_SHA
        assert plan1.precision == "fp16"
        assert plan1.recipe_id == "pytorch_transformers_hip"
        assert len(plan1.steps) == 5
        assert plan1.steps[0].name == "check_prerequisites"
        assert plan1.steps[1].name == "materialize_model"
        assert plan1.steps[4].name == "verify_build"

    def test_create_plan_never_downloads_weights(
        self, sample_qwen_spec: ModelSpec, non_amd_detection_report: DetectionReport
    ) -> None:
        mock_inspector = MagicMock()
        mock_inspector.inspect.return_value = sample_qwen_spec
        mock_observer = MagicMock()
        mock_observer.observe.return_value = non_amd_detection_report

        with patch("huggingface_hub.snapshot_download") as mock_download:
            planner = ForgePlanner(model_inspector=mock_inspector, system_observer=mock_observer)
            plan = planner.create_plan("Qwen/Qwen2.5-0.5B-Instruct")
            mock_download.assert_not_called()

        assert plan.model_id == "Qwen/Qwen2.5-0.5B-Instruct"

    def test_create_plan_detects_amd_gpu(
        self, sample_qwen_spec: ModelSpec, amd_detection_report: DetectionReport
    ) -> None:
        mock_inspector = MagicMock()
        mock_inspector.inspect.return_value = sample_qwen_spec
        mock_observer = MagicMock()
        mock_observer.observe.return_value = amd_detection_report

        planner = ForgePlanner(model_inspector=mock_inspector, system_observer=mock_observer)
        plan = planner.create_plan("Qwen/Qwen2.5-0.5B-Instruct")
        assert plan.target_gpu == "gfx1100"

    def test_create_plan_unsupported_arch_raises(self, non_amd_detection_report: DetectionReport) -> None:
        mock_inspector = MagicMock()
        mock_inspector.inspect.return_value = ModelSpec(
            model_id="bert-base",
            requested_revision="main",
            commit_sha=SAMPLE_COMMIT_SHA,
            architecture="BertModel",
        )
        mock_observer = MagicMock()
        mock_observer.observe.return_value = non_amd_detection_report

        planner = ForgePlanner(model_inspector=mock_inspector, system_observer=mock_observer)
        with pytest.raises(UnsupportedModelArchitectureError):
            planner.create_plan("bert-base")

    def test_disk_space_estimation(self, sample_qwen_spec: ModelSpec) -> None:
        planner = ForgePlanner()
        # fp16: 2 bytes per param * 1.15
        bytes_fp16 = planner.estimate_required_disk_bytes(sample_qwen_spec, "fp16")
        assert sample_qwen_spec.parameter_count is not None
        expected_fp16 = int(sample_qwen_spec.parameter_count * 2 * 1.15)
        assert bytes_fp16 == expected_fp16

        # fp32: 4 bytes per param * 1.15
        bytes_fp32 = planner.estimate_required_disk_bytes(sample_qwen_spec, "fp32")
        expected_fp32 = int(sample_qwen_spec.parameter_count * 4 * 1.15)
        assert bytes_fp32 == expected_fp32


# ==============================================================================
# Materializer Tests
# ==============================================================================


class TestModelMaterializer:
    def test_insufficient_disk_space_raises(self, tmp_path: Path, sample_qwen_spec: ModelSpec) -> None:
        materializer = ModelMaterializer(cache_dir=tmp_path)
        with patch("shutil.disk_usage") as mock_usage:
            # 100 MB free, but 2 GB requested
            mock_usage.return_value = MagicMock(free=100 * 1024 * 1024)
            with pytest.raises(InsufficientDiskSpaceError) as exc_info:
                materializer.materialize(sample_qwen_spec, required_disk_bytes=2 * 1024 * 1024 * 1024)

        assert "Insufficient disk space" in str(exc_info.value)

    def test_materialize_gated_repo_raises_auth_required(
        self, tmp_path: Path, sample_qwen_spec: ModelSpec
    ) -> None:
        from huggingface_hub.errors import GatedRepoError

        materializer = ModelMaterializer(cache_dir=tmp_path)
        mock_resp = MagicMock(status_code=403)
        with patch("shutil.disk_usage") as mock_usage:
            mock_usage.return_value = MagicMock(free=100 * 1024**3)
            with patch(
                "rocmhub.forge.materializer.snapshot_download",
                side_effect=GatedRepoError("Access denied", response=mock_resp),
            ):
                with pytest.raises(AuthRequiredError) as exc_info:
                    materializer.materialize(sample_qwen_spec)

        assert "gated or private" in str(exc_info.value)

    def test_materialize_missing_repo_raises_model_not_found(
        self, tmp_path: Path, sample_qwen_spec: ModelSpec
    ) -> None:
        from huggingface_hub.errors import RepositoryNotFoundError

        materializer = ModelMaterializer(cache_dir=tmp_path)
        mock_resp = MagicMock(status_code=404)
        with patch("shutil.disk_usage") as mock_usage:
            mock_usage.return_value = MagicMock(free=100 * 1024**3)
            with patch(
                "rocmhub.forge.materializer.snapshot_download",
                side_effect=RepositoryNotFoundError("Repo not found", response=mock_resp),
            ):
                with pytest.raises(ModelNotFoundError):
                    materializer.materialize(sample_qwen_spec)

    def test_materialize_success_no_weights(self, tmp_path: Path, sample_qwen_spec: ModelSpec) -> None:
        # Create a mock snapshot dir
        snapshot_dir = tmp_path / "mock_snapshot"
        snapshot_dir.mkdir(parents=True)
        (snapshot_dir / "config.json").write_text('{"vocab_size": 100}')
        (snapshot_dir / "tokenizer.json").write_text('{"version": "1.0"}')

        materializer = ModelMaterializer(cache_dir=tmp_path)
        with patch("shutil.disk_usage") as mock_usage:
            mock_usage.return_value = MagicMock(free=100 * 1024**3)
            with patch("rocmhub.forge.materializer.snapshot_download", return_value=str(snapshot_dir)):
                result = materializer.materialize(sample_qwen_spec, download_weights=False)

        assert result.local_path == str(snapshot_dir)
        assert result.mode == MaterializationMode.METADATA_ONLY
        assert result.has_weights is False
        assert "config.json" in result.files
        assert "tokenizer.json" in result.files
        assert result.weights_size_bytes == 0

    def test_materialize_full_weights_standalone_success(
        self, tmp_path: Path, sample_qwen_spec: ModelSpec
    ) -> None:
        snapshot_dir = tmp_path / "mock_snapshot_full"
        snapshot_dir.mkdir(parents=True)
        (snapshot_dir / "config.json").write_text('{"vocab_size": 100}')
        (snapshot_dir / "tokenizer.json").write_text('{"version": "1.0"}')
        (snapshot_dir / "model.safetensors").write_bytes(b"dummy weights" * 100)

        materializer = ModelMaterializer(cache_dir=tmp_path)
        with patch("shutil.disk_usage") as mock_usage:
            mock_usage.return_value = MagicMock(free=100 * 1024**3)
            with patch("rocmhub.forge.materializer.snapshot_download", return_value=str(snapshot_dir)):
                result = materializer.materialize(sample_qwen_spec, download_weights=True)

        assert result.mode == MaterializationMode.FULL_WEIGHTS
        assert result.has_weights is True
        assert result.weights_size_bytes == len(b"dummy weights" * 100)

    def test_materialize_full_weights_sharded_success(
        self, tmp_path: Path, sample_qwen_spec: ModelSpec
    ) -> None:
        snapshot_dir = tmp_path / "mock_snapshot_sharded"
        snapshot_dir.mkdir(parents=True)
        (snapshot_dir / "config.json").write_text('{"vocab_size": 100}')
        (snapshot_dir / "tokenizer.json").write_text('{"version": "1.0"}')
        index_data = {
            "metadata": {"total_size": 200},
            "weight_map": {
                "layer.0.weight": "model-00001-of-00002.safetensors",
                "layer.1.weight": "model-00002-of-00002.safetensors",
            },
        }
        (snapshot_dir / "model.safetensors.index.json").write_text(json.dumps(index_data))
        (snapshot_dir / "model-00001-of-00002.safetensors").write_bytes(b"shard 1" * 20)
        (snapshot_dir / "model-00002-of-00002.safetensors").write_bytes(b"shard 2" * 20)

        materializer = ModelMaterializer(cache_dir=tmp_path)
        with patch("shutil.disk_usage") as mock_usage:
            mock_usage.return_value = MagicMock(free=100 * 1024**3)
            with patch("rocmhub.forge.materializer.snapshot_download", return_value=str(snapshot_dir)):
                result = materializer.materialize(sample_qwen_spec, mode=MaterializationMode.FULL_WEIGHTS)

        assert result.has_weights is True
        assert result.weights_size_bytes == len(b"shard 1" * 20) + len(b"shard 2" * 20)

    def test_materialize_full_weights_missing_shard_raises(
        self, tmp_path: Path, sample_qwen_spec: ModelSpec
    ) -> None:
        snapshot_dir = tmp_path / "mock_snapshot_missing_shard"
        snapshot_dir.mkdir(parents=True)
        (snapshot_dir / "config.json").write_text('{"vocab_size": 100}')
        (snapshot_dir / "tokenizer.json").write_text('{"version": "1.0"}')
        index_data = {
            "weight_map": {
                "layer.0.weight": "model-00001-of-00002.safetensors",
                "layer.1.weight": "model-00002-of-00002.safetensors",
            }
        }
        (snapshot_dir / "model.safetensors.index.json").write_text(json.dumps(index_data))
        (snapshot_dir / "model-00001-of-00002.safetensors").write_bytes(b"shard 1")
        # Note: model-00002-of-00002.safetensors is intentionally missing!

        materializer = ModelMaterializer(cache_dir=tmp_path)
        with patch("shutil.disk_usage") as mock_usage:
            mock_usage.return_value = MagicMock(free=100 * 1024**3)
            with patch("rocmhub.forge.materializer.snapshot_download", return_value=str(snapshot_dir)):
                with pytest.raises(ModelMaterializationError) as exc_info:
                    materializer.materialize(sample_qwen_spec, mode=MaterializationMode.FULL_WEIGHTS)

        assert "missing shards: model-00002-of-00002.safetensors" in str(exc_info.value)

    def test_materialize_full_weights_empty_shard_raises(
        self, tmp_path: Path, sample_qwen_spec: ModelSpec
    ) -> None:
        snapshot_dir = tmp_path / "mock_snapshot_empty_shard"
        snapshot_dir.mkdir(parents=True)
        (snapshot_dir / "config.json").write_text('{"vocab_size": 100}')
        (snapshot_dir / "tokenizer.json").write_text('{"version": "1.0"}')
        index_data = {"weight_map": {"layer.0.weight": "model-00001-of-00001.safetensors"}}
        (snapshot_dir / "model.safetensors.index.json").write_text(json.dumps(index_data))
        (snapshot_dir / "model-00001-of-00001.safetensors").write_bytes(b"")  # 0 bytes!

        materializer = ModelMaterializer(cache_dir=tmp_path)
        with patch("shutil.disk_usage") as mock_usage:
            mock_usage.return_value = MagicMock(free=100 * 1024**3)
            with patch("rocmhub.forge.materializer.snapshot_download", return_value=str(snapshot_dir)):
                with pytest.raises(ModelMaterializationError) as exc_info:
                    materializer.materialize(sample_qwen_spec, mode=MaterializationMode.FULL_WEIGHTS)

        assert "empty shards: model-00001-of-00001.safetensors" in str(exc_info.value)

    def test_materialize_full_weights_missing_config_raises(
        self, tmp_path: Path, sample_qwen_spec: ModelSpec
    ) -> None:
        snapshot_dir = tmp_path / "mock_snapshot_no_config"
        snapshot_dir.mkdir(parents=True)
        (snapshot_dir / "tokenizer.json").write_text('{"version": "1.0"}')
        (snapshot_dir / "model.safetensors").write_bytes(b"dummy")

        materializer = ModelMaterializer(cache_dir=tmp_path)
        with patch("shutil.disk_usage") as mock_usage:
            mock_usage.return_value = MagicMock(free=100 * 1024**3)
            with patch("rocmhub.forge.materializer.snapshot_download", return_value=str(snapshot_dir)):
                with pytest.raises(ModelMaterializationError) as exc_info:
                    materializer.materialize(sample_qwen_spec, mode=MaterializationMode.FULL_WEIGHTS)

        assert "config.json' missing" in str(exc_info.value)

    def test_materialize_full_weights_missing_tokenizer_raises(
        self, tmp_path: Path, sample_qwen_spec: ModelSpec
    ) -> None:
        snapshot_dir = tmp_path / "mock_snapshot_no_tok"
        snapshot_dir.mkdir(parents=True)
        (snapshot_dir / "config.json").write_text('{"vocab_size": 100}')
        (snapshot_dir / "model.safetensors").write_bytes(b"dummy")

        materializer = ModelMaterializer(cache_dir=tmp_path)
        with patch("shutil.disk_usage") as mock_usage:
            mock_usage.return_value = MagicMock(free=100 * 1024**3)
            with patch("rocmhub.forge.materializer.snapshot_download", return_value=str(snapshot_dir)):
                with pytest.raises(ModelMaterializationError) as exc_info:
                    materializer.materialize(sample_qwen_spec, mode=MaterializationMode.FULL_WEIGHTS)

        assert "tokenizer files missing" in str(exc_info.value)


# ==============================================================================
# Manifest & Secrets Tests
# ==============================================================================


class TestBuildManifest:
    def test_sanitize_secrets(self) -> None:
        raw = {
            "token": "hf_abcdefghijklmnopqrstuvwxyz12345678",
            "normal_field": "public_value",
            "secret_scan_clean": True,
            "nested": {
                "auth_key": "secret1234567890",
                "text": "Using token hf_1111222233334444555566667777888899 in request",
            },
        }
        sanitized = sanitize_secrets_in_obj(raw)
        assert sanitized["token"] == "[REDACTED]"
        assert sanitized["normal_field"] == "public_value"
        assert sanitized["secret_scan_clean"] is True
        assert sanitized["nested"]["auth_key"] == "[REDACTED]"
        assert "hf_" not in sanitized["nested"]["text"]

    def test_assert_no_secrets_raises(self) -> None:
        bad_json = '{"api_key": "hf_ABCDEFGHIJKLMNOPQRSTUVWXYZ12345678"}'
        with pytest.raises(SecretDetectedError):
            assert_no_secrets(bad_json)

    def test_write_manifest_creates_canonical_json(self, tmp_path: Path) -> None:
        manifest = BuildManifest(
            build_id="bld_1234567890abcdef",
            plan_id="plan_1234567890abcdef",
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
            revision=SAMPLE_COMMIT_SHA,
            precision="fp16",
            recipe_id="pytorch_transformers_hip",
            recipe_version="1.0.0",
            runtime="pytorch_transformers_hip",
            status=BuildStatus.PREPARED,
            build_dir=str(tmp_path),
            weights_path=str(tmp_path / "weights"),
            steps=[],
            amd_validated=False,
            secret_scan_clean=True,
            artifacts={"runtime_config.json": "abc123sha256"},
        )
        target = tmp_path / "build_manifest.json"
        write_manifest(manifest, target)
        assert target.exists()
        loaded = json.loads(target.read_text())
        assert loaded["build_id"] == "bld_1234567890abcdef"
        assert loaded["status"] == "PREPARED"
        assert loaded["amd_validated"] is False


# ==============================================================================
# Executor Tests
# ==============================================================================


class TestForgeExecutor:
    def test_execute_success_on_non_amd(
        self, tmp_path: Path, sample_qwen_spec: ModelSpec, non_amd_detection_report: DetectionReport
    ) -> None:
        build_dir = tmp_path / "build_out"
        mock_snapshot = tmp_path / "hf_snapshot"
        mock_snapshot.mkdir(parents=True)
        (mock_snapshot / "config.json").write_text("{}")
        (mock_snapshot / "model.safetensors").write_bytes(b"dummy weight tensors")

        mock_materializer = MagicMock()
        mock_materializer.check_disk_space.return_value = None
        mock_materializer.materialize.return_value = MaterializedModel(
            local_path=str(mock_snapshot),
            mode=MaterializationMode.FULL_WEIGHTS,
            files=["config.json", "model.safetensors"],
            has_weights=True,
            weights_size_bytes=20,
            cached=True,
        )

        mock_inspector = MagicMock()
        mock_inspector.inspect.return_value = sample_qwen_spec

        mock_observer = MagicMock()
        mock_observer.observe.return_value = non_amd_detection_report

        plan = ForgePlan(
            plan_id="plan_test123",
            model_id=sample_qwen_spec.model_id,
            revision=sample_qwen_spec.commit_sha,
            precision="fp16",
            recipe_id="pytorch_transformers_hip",
            recipe_version="1.0.0",
            output_dir=str(build_dir),
            estimated_disk_space_bytes=1000,
            steps=PyTorchTransformersHipRecipe().get_steps(),
        )

        executor = ForgeExecutor(
            materializer=mock_materializer,
            inspector=mock_inspector,
            observer=mock_observer,
        )

        manifest = executor.execute(plan=plan, model_spec=sample_qwen_spec, execute_inference=True)

        # 1. Output files must all exist
        assert (build_dir / "runtime_config.json").exists()
        assert (build_dir / "model_config.json").exists()
        assert (build_dir / "recipe.json").exists()
        assert (build_dir / "run_inference.py").exists()
        assert (build_dir / "build_manifest.json").exists()

        # 2. Status must be PREPARED and amd_validated must be False
        assert manifest.status == BuildStatus.PREPARED
        assert manifest.amd_validated is False

        # 3. Steps recorded correctly
        step_names = [s.name for s in manifest.steps]
        assert "check_prerequisites" in step_names
        assert "materialize_model" in step_names
        assert "configure_runtime" in step_names
        assert "generate_launch_scripts" in step_names
        assert "verify_build" in step_names
        assert "execute_inference" in step_names

        # execute_inference must be SKIPPED on non-AMD
        exec_step = next(s for s in manifest.steps if s.name == "execute_inference")
        assert exec_step.status == StepStatus.SKIPPED
        assert "no AMD ROCm GPU detected" in (exec_step.message or "")

    def test_execute_directory_conflict_without_force_raises(
        self, tmp_path: Path, sample_qwen_spec: ModelSpec
    ) -> None:
        build_dir = tmp_path / "existing_build"
        build_dir.mkdir(parents=True)
        (build_dir / "old_file.txt").write_text("existing content")

        plan = ForgePlan(
            plan_id="plan_conflict_test",
            model_id=sample_qwen_spec.model_id,
            revision=sample_qwen_spec.commit_sha,
            precision="fp16",
            recipe_id="pytorch_transformers_hip",
            recipe_version="1.0.0",
            output_dir=str(build_dir),
            estimated_disk_space_bytes=1000,
            steps=[],
        )

        executor = ForgeExecutor()
        with pytest.raises(BuildConflictError):
            executor.execute(plan=plan, model_spec=sample_qwen_spec, force=False)

    def test_execute_config_only_when_no_weights(
        self, tmp_path: Path, sample_qwen_spec: ModelSpec, non_amd_detection_report: DetectionReport
    ) -> None:
        build_dir = tmp_path / "config_only_build"
        mock_snapshot = tmp_path / "hf_meta"
        mock_snapshot.mkdir(parents=True)
        (mock_snapshot / "config.json").write_text("{}")

        mock_materializer = MagicMock()
        mock_materializer.check_disk_space.return_value = None
        mock_materializer.materialize.return_value = MaterializedModel(
            local_path=str(mock_snapshot),
            mode=MaterializationMode.METADATA_ONLY,
            files=["config.json"],
            has_weights=False,
            weights_size_bytes=0,
            cached=True,
        )

        plan = ForgePlan(
            plan_id="plan_config_only",
            model_id=sample_qwen_spec.model_id,
            revision=sample_qwen_spec.commit_sha,
            precision="fp16",
            recipe_id="pytorch_transformers_hip",
            recipe_version="1.0.0",
            output_dir=str(build_dir),
            estimated_disk_space_bytes=0,
            steps=PyTorchTransformersHipRecipe().get_steps(),
        )

        executor = ForgeExecutor(materializer=mock_materializer)
        manifest = executor.execute(
            plan=plan,
            model_spec=sample_qwen_spec,
            download_weights=False,
            execute_inference=False,
        )

        assert manifest.status == BuildStatus.CONFIG_ONLY
        assert manifest.amd_validated is False
        assert (build_dir / "runtime_config.json").exists()
        assert (build_dir / "run_inference.py").exists()

    def test_execute_inference_on_config_only_raises(
        self, tmp_path: Path, sample_qwen_spec: ModelSpec
    ) -> None:
        build_dir = tmp_path / "config_only_inference"
        mock_snapshot = tmp_path / "hf_meta2"
        mock_snapshot.mkdir(parents=True)

        mock_materializer = MagicMock()
        mock_materializer.check_disk_space.return_value = None
        mock_materializer.materialize.return_value = MaterializedModel(
            local_path=str(mock_snapshot),
            mode=MaterializationMode.METADATA_ONLY,
            files=["config.json"],
            has_weights=False,
            weights_size_bytes=0,
            cached=True,
        )

        plan = ForgePlan(
            plan_id="plan_forbid_exec",
            model_id=sample_qwen_spec.model_id,
            revision=sample_qwen_spec.commit_sha,
            precision="fp16",
            recipe_id="pytorch_transformers_hip",
            recipe_version="1.0.0",
            output_dir=str(build_dir),
            estimated_disk_space_bytes=0,
            steps=[],
        )

        executor = ForgeExecutor(materializer=mock_materializer)
        with pytest.raises(BuildExecutionError) as exc_info:
            executor.execute(
                plan=plan,
                model_spec=sample_qwen_spec,
                download_weights=False,
                execute_inference=True,
            )
        assert "Inference execution is forbidden when build status is CONFIG_ONLY" in str(exc_info.value)


# ==============================================================================
# CLI Integration Tests
# ==============================================================================


class TestForgeCLI:
    def test_cli_forge_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["forge", "--help"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "plan" in captured.out
        assert "build" in captured.out

    def test_cli_forge_plan_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["forge", "plan", "--help"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "--precision" in captured.out
        assert "--target-gpu" in captured.out
        assert "--output-dir" in captured.out

    def test_cli_forge_plan_mocked(
        self,
        capsys: pytest.CaptureFixture[str],
        sample_qwen_spec: ModelSpec,
        non_amd_detection_report: DetectionReport,
    ) -> None:
        with patch("rocmhub.forge.planner.ModelInspector") as mock_insp_cls, \
             patch("rocmhub.forge.planner.SystemObserver") as mock_obs_cls:
            mock_insp = MagicMock()
            mock_insp.inspect.return_value = sample_qwen_spec
            mock_insp_cls.return_value = mock_insp

            mock_obs = MagicMock()
            mock_obs.observe.return_value = non_amd_detection_report
            mock_obs_cls.return_value = mock_obs

            exit_code = main(["forge", "plan", "Qwen/Qwen2.5-0.5B-Instruct"])
            assert exit_code == 0
            captured = capsys.readouterr()
            assert "ROCmHub Model Forge Plan" in captured.out
            assert "Qwen/Qwen2.5-0.5B-Instruct" in captured.out
            assert "pytorch_transformers_hip" in captured.out
            assert "Planned Build Steps:" in captured.out

    def test_cli_forge_plan_json(
        self,
        capsys: pytest.CaptureFixture[str],
        sample_qwen_spec: ModelSpec,
        non_amd_detection_report: DetectionReport,
    ) -> None:
        with patch("rocmhub.forge.planner.ModelInspector") as mock_insp_cls, \
             patch("rocmhub.forge.planner.SystemObserver") as mock_obs_cls:
            mock_insp = MagicMock()
            mock_insp.inspect.return_value = sample_qwen_spec
            mock_insp_cls.return_value = mock_insp

            mock_obs = MagicMock()
            mock_obs.observe.return_value = non_amd_detection_report
            mock_obs_cls.return_value = mock_obs

            exit_code = main(["forge", "plan", "Qwen/Qwen2.5-0.5B-Instruct", "--json"])
            assert exit_code == 0
            captured = capsys.readouterr()
            data = json.loads(captured.out)
            assert data["model_id"] == "Qwen/Qwen2.5-0.5B-Instruct"
            assert data["revision"] == SAMPLE_COMMIT_SHA
            assert data["recipe_id"] == "pytorch_transformers_hip"

    def test_cli_forge_missing_model_id(self, capsys: pytest.CaptureFixture[str]) -> None:
        exit_code = main(["forge", "plan"])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Model ID must be specified" in captured.err

    def test_cli_forge_build_mocked(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        sample_qwen_spec: ModelSpec,
        non_amd_detection_report: DetectionReport,
    ) -> None:
        build_dir = tmp_path / "cli_build"
        mock_snapshot = tmp_path / "snapshot"
        mock_snapshot.mkdir(parents=True)
        (mock_snapshot / "config.json").write_text("{}")

        with patch("rocmhub.forge.planner.ModelInspector") as mock_pinsp_cls, \
             patch("rocmhub.forge.planner.SystemObserver") as mock_pobs_cls, \
             patch("rocmhub.forge.executor.ModelInspector") as mock_einsp_cls, \
             patch("rocmhub.forge.executor.SystemObserver") as mock_eobs_cls, \
             patch("rocmhub.forge.executor.ModelMaterializer") as mock_mat_cls:

            mock_insp = MagicMock()
            mock_insp.inspect.return_value = sample_qwen_spec
            mock_pinsp_cls.return_value = mock_insp
            mock_einsp_cls.return_value = mock_insp

            mock_obs = MagicMock()
            mock_obs.observe.return_value = non_amd_detection_report
            mock_pobs_cls.return_value = mock_obs
            mock_eobs_cls.return_value = mock_obs

            mock_mat = MagicMock()
            mock_mat.check_disk_space.return_value = None
            mock_mat.materialize.return_value = MaterializedModel(
                local_path=str(mock_snapshot),
                mode=MaterializationMode.METADATA_ONLY,
                files=["config.json"],
                has_weights=False,
                weights_size_bytes=0,
                cached=True,
            )
            mock_mat_cls.return_value = mock_mat

            exit_code = main([
                "forge", "build", "Qwen/Qwen2.5-0.5B-Instruct",
                "--output-dir", str(build_dir),
                "--no-weights",
            ])
            assert exit_code == 0
            captured = capsys.readouterr()
            assert "ROCmHub Model Forge Build" in captured.out
            assert "CONFIG_ONLY" in captured.out
            assert "AMD Validated:" in captured.out
            assert (build_dir / "build_manifest.json").exists()


class TestForgeExecutionReadiness:
    """Test suite for Phase 17: AMD Execution Readiness."""

    def test_execute_inference_non_amd_skips(
        self,
        tmp_path: Path,
        sample_qwen_spec: ModelSpec,
        non_amd_detection_report: DetectionReport,
    ) -> None:
        """On non-AMD hosts, execute_inference=True cleanly skips and stays PREPARED."""
        build_dir = tmp_path / "bld_non_amd"
        plan = ForgePlan(
            plan_id="plan_test_non_amd",
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
            revision=SAMPLE_COMMIT_SHA,
            precision="fp16",
            recipe_id="pytorch_transformers_hip",
            recipe_version="1.0.0",
            output_dir=str(build_dir),
        )

        mock_obs = MagicMock()
        mock_obs.observe.return_value = non_amd_detection_report

        mock_mat = MagicMock()
        mock_mat.materialize.return_value = MaterializedModel(
            local_path=str(tmp_path / "snap"),
            mode=MaterializationMode.FULL_WEIGHTS,
            files=["config.json", "model.safetensors"],
            has_weights=True,
            weights_size_bytes=1000,
            cached=True,
        )

        mock_insp = MagicMock()
        mock_insp.inspect.return_value = sample_qwen_spec

        executor = ForgeExecutor(materializer=mock_mat, inspector=mock_insp, observer=mock_obs)
        manifest = executor.execute(
            plan=plan,
            download_weights=True,
            force=True,
            execute_inference=True,
        )

        assert manifest.status == BuildStatus.PREPARED
        assert manifest.amd_validated is False
        exec_step = next(s for s in manifest.steps if s.name == "execute_inference")
        assert exec_step.status == StepStatus.SKIPPED
        assert "no AMD ROCm GPU detected" in exec_step.message

    def test_execute_inference_amd_success(
        self,
        tmp_path: Path,
        sample_qwen_spec: ModelSpec,
        amd_detection_report: DetectionReport,
    ) -> None:
        """On AMD GPU hosts, execute_inference=True invokes launcher and transitions to EXECUTED."""
        build_dir = tmp_path / "bld_amd_success"
        plan = ForgePlan(
            plan_id="plan_test_amd",
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
            revision=SAMPLE_COMMIT_SHA,
            precision="fp16",
            recipe_id="pytorch_transformers_hip",
            recipe_version="1.0.0",
            output_dir=str(build_dir),
        )

        mock_obs = MagicMock()
        mock_obs.observe.return_value = amd_detection_report

        mock_mat = MagicMock()
        mock_mat.materialize.return_value = MaterializedModel(
            local_path=str(tmp_path / "snap"),
            mode=MaterializationMode.FULL_WEIGHTS,
            files=["config.json", "model.safetensors"],
            has_weights=True,
            weights_size_bytes=1000,
            cached=True,
        )

        mock_insp = MagicMock()
        mock_insp.inspect.return_value = sample_qwen_spec

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = json.dumps({
            "status": "SUCCESS",
            "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
            "revision": SAMPLE_COMMIT_SHA,
            "device": "cuda:0",
            "precision": "fp16",
            "prompt": "Hello",
            "generated_text": "Open-source AI acceleration powers the future.",
            "tokens_generated": 16,
            "load_time_seconds": 0.45,
            "generation_time_seconds": 0.20,
            "tokens_per_second": 80.0,
            "is_hip": True,
            "amd_gpu_used": True,
        })
        mock_proc.stderr = ""

        with patch("subprocess.run", return_value=mock_proc) as mock_sub:
            executor = ForgeExecutor(materializer=mock_mat, inspector=mock_insp, observer=mock_obs)
            manifest = executor.execute(
                plan=plan,
                download_weights=True,
                force=True,
                execute_inference=True,
            )

            assert mock_sub.called
            assert manifest.status == BuildStatus.EXECUTED
            assert manifest.amd_validated is True
            exec_step = next(s for s in manifest.steps if s.name == "execute_inference")
            assert exec_step.status == StepStatus.SUCCESS
            assert exec_step.details["tokens_generated"] == 16
            assert exec_step.details["tokens_per_second"] == 80.0

    def test_execute_inference_amd_failure(
        self,
        tmp_path: Path,
        sample_qwen_spec: ModelSpec,
        amd_detection_report: DetectionReport,
    ) -> None:
        """When launcher exits with non-zero returncode on AMD, manifest reflects FAILED."""
        build_dir = tmp_path / "bld_amd_failure"
        plan = ForgePlan(
            plan_id="plan_test_amd_fail",
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
            revision=SAMPLE_COMMIT_SHA,
            precision="fp16",
            recipe_id="pytorch_transformers_hip",
            recipe_version="1.0.0",
            output_dir=str(build_dir),
        )

        mock_obs = MagicMock()
        mock_obs.observe.return_value = amd_detection_report

        mock_mat = MagicMock()
        mock_mat.materialize.return_value = MaterializedModel(
            local_path=str(tmp_path / "snap"),
            mode=MaterializationMode.FULL_WEIGHTS,
            files=["config.json", "model.safetensors"],
            has_weights=True,
            weights_size_bytes=1000,
            cached=True,
        )

        mock_insp = MagicMock()
        mock_insp.inspect.return_value = sample_qwen_spec

        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stdout = json.dumps({"status": "FAILED", "error": "CUDA out of memory."})
        mock_proc.stderr = "RuntimeError: CUDA out of memory."

        with patch("subprocess.run", return_value=mock_proc):
            executor = ForgeExecutor(materializer=mock_mat, inspector=mock_insp, observer=mock_obs)
            manifest = executor.execute(
                plan=plan,
                download_weights=True,
                force=True,
                execute_inference=True,
            )

            assert manifest.status == BuildStatus.FAILED
            assert manifest.amd_validated is False
            exec_step = next(s for s in manifest.steps if s.name == "execute_inference")
            assert exec_step.status == StepStatus.FAILED
            assert "CUDA out of memory" in exec_step.message

    def test_execute_build_missing_script(self, tmp_path: Path) -> None:
        """Calling execute_build on empty dir returns FAILED ExecutionResult."""
        executor = ForgeExecutor()
        res = executor.execute_build(tmp_path)
        assert res.status.value == "FAILED"
        assert "not found" in (res.error_message or "")
        assert res.exit_code == 1

    def test_cli_forge_execute_subcommand(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        non_amd_detection_report: DetectionReport,
    ) -> None:
        """CLI rocmhub forge execute works and outputs JSON."""
        build_dir = tmp_path / "cli_exec_test"
        build_dir.mkdir()
        launcher = build_dir / "run_inference.py"
        launcher.write_text("#!/usr/bin/env python3\nprint('hello')\n")

        with patch("rocmhub.forge.executor.SystemObserver") as mock_obs_cls:
            mock_obs = MagicMock()
            mock_obs.observe.return_value = non_amd_detection_report
            mock_obs_cls.return_value = mock_obs

            # Without AMD GPU, default --device cuda returns SKIPPED with exit code 0
            exit_code = main(["forge", "execute", str(build_dir), "--json"])
            assert exit_code == 0
            captured = capsys.readouterr()
            data = json.loads(captured.out)
            assert data["status"] == "SKIPPED"
            assert data["amd_validated"] is False

