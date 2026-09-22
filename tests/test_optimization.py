"""Unit and integration tests for ROCmHub Optimization Engine (Phase 12)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from rocmhub.cli.main import main
from rocmhub.core.errors import (
    IncomparableResultsError,
    SecurityBoundaryError,
)
from rocmhub.core.types import (
    BenchmarkResult,
    DetectionReport,
    EnvironmentSpec,
    ExecutionStatus,
    HardwareSpec,
    ModelSpec,
    ValidationMode,
    ValidationReport,
    ValidationVerdict,
)
from rocmhub.engineer.tools import ToolRegistry
from rocmhub.forge.base import MaterializationMode
from rocmhub.forge.materializer import MaterializedModel
from rocmhub.optimization.base import (
    CandidateStatus,
    ComparisonVerdict,
    OptimizationBaseline,
    OptimizationCandidate,
    OptimizationRequest,
    OptimizationStrategy,
)
from rocmhub.optimization.comparison import ComparisonEngine
from rocmhub.optimization.executor import OptimizationExecutor
from rocmhub.optimization.recipes import (
    BF16OptimizationRecipe,
    FP16OptimizationRecipe,
    FP32OptimizationRecipe,
    QuantizationRecipe,
    TorchCompileRecipe,
    get_recipe_for_strategy,
)

SAMPLE_COMMIT_SHA = "7ae557604adf67be50417f59c2c2f167def9a775"


@pytest.fixture
def sample_model_spec() -> ModelSpec:
    return ModelSpec(
        schema_version="1.0.0",
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        source="huggingface",
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
def non_amd_report() -> DetectionReport:
    return DetectionReport(
        schema_version="1.0.0",
        environment=EnvironmentSpec(
            schema_version="1.0.0",
            os="Darwin 24.0.0",
            python_version="3.9.6",
            kernel=None,
            rocm_version=None,
            hip_version=None,
            torch_version="2.2.0",
            torch_hip_available=False,
            env_vars={},
        ),
        gpus=[],
        provenance={},
        warnings=["No AMD ROCm GPU detected on host."],
    )


@pytest.fixture
def amd_mi300x_gpu() -> HardwareSpec:
    return HardwareSpec(
        schema_version="1.0.0",
        gpu_present=True,
        gpu_vendor="AMD",
        device_id=0,
        device_name="AMD Instinct MI300X",
        family="Instinct",
        gfx_target="gfx942",
        vram_total_mb=196608,
        compute_units=304,
        bus_id="0000:43:00.0",
    )


@pytest.fixture
def amd_report(amd_mi300x_gpu: HardwareSpec) -> DetectionReport:
    return DetectionReport(
        schema_version="1.0.0",
        environment=EnvironmentSpec(
            schema_version="1.0.0",
            os="Linux 6.8.0",
            python_version="3.9.6",
            kernel="6.8.0",
            rocm_version="6.2.0",
            hip_version="6.2.0",
            torch_version="2.4.0+rocm6.2",
            torch_hip_available=True,
            env_vars={},
        ),
        gpus=[amd_mi300x_gpu],
        provenance={},
        warnings=[],
    )


# ---------------------------------------------------------------------------
# 1. Data Models and Invariants
# ---------------------------------------------------------------------------

class TestOptimizationDataModels:
    def test_baseline_immutable(self) -> None:
        baseline = OptimizationBaseline(
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
            revision=SAMPLE_COMMIT_SHA,
            precision="fp16",
            status=CandidateStatus.CONFIG_ONLY,
            measured=False,
        )
        with pytest.raises(ValidationError):
            # Baseline is frozen
            baseline.precision = "bf16"  # type: ignore[misc]

    def test_deterministic_candidate_ids(self) -> None:
        cand1 = OptimizationCandidate(
            candidate_id="cand-12345678abcdef01",
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
            revision=SAMPLE_COMMIT_SHA,
            strategy=OptimizationStrategy.BF16,
            precision="bf16",
            status=CandidateStatus.PLANNED,
        )
        cand2 = OptimizationCandidate(
            candidate_id="cand-12345678abcdef01",
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
            revision=SAMPLE_COMMIT_SHA,
            strategy=OptimizationStrategy.BF16,
            precision="bf16",
            status=CandidateStatus.PLANNED,
        )
        assert cand1.candidate_id == cand2.candidate_id

    def test_candidate_models_frozen(self) -> None:
        cand = OptimizationCandidate(
            candidate_id="cand-12345678abcdef01",
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
            revision=SAMPLE_COMMIT_SHA,
            strategy=OptimizationStrategy.FP16,
            status=CandidateStatus.PLANNED,
        )
        with pytest.raises(ValidationError):
            cand.status = CandidateStatus.EXECUTED  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 2. Optimization Recipes
# ---------------------------------------------------------------------------

class TestOptimizationRecipes:
    def test_bf16_recipe(self, sample_model_spec: ModelSpec) -> None:
        recipe = get_recipe_for_strategy(OptimizationStrategy.BF16)
        assert isinstance(recipe, BF16OptimizationRecipe)
        assert recipe.precision == "bf16"
        is_supp, reason = recipe.is_supported(sample_model_spec, None)
        assert is_supp is True
        assert reason is None

    def test_fp16_recipe(self, sample_model_spec: ModelSpec) -> None:
        recipe = get_recipe_for_strategy(OptimizationStrategy.FP16)
        assert isinstance(recipe, FP16OptimizationRecipe)
        assert recipe.precision == "fp16"
        is_supp, reason = recipe.is_supported(sample_model_spec, None)
        assert is_supp is True
        assert reason is None

    def test_fp32_recipe_vram_exceeded(self, sample_model_spec: ModelSpec) -> None:
        recipe = FP32OptimizationRecipe()
        # Tiny GPU with only 100MB VRAM
        tiny_gpu = HardwareSpec(
            gpu_present=True,
            device_id=0,
            device_name="Tiny GPU",
            vram_total_mb=100,
        )
        is_supp, reason = recipe.is_supported(sample_model_spec, tiny_gpu)
        assert is_supp is False
        assert reason is not None
        assert "exceeding device capacity" in reason

    def test_unsupported_quantization_int8_without_libs(self, sample_model_spec: ModelSpec) -> None:
        recipe = QuantizationRecipe(OptimizationStrategy.INT8)
        with patch("importlib.util.find_spec", return_value=None):
            is_supp, reason = recipe.is_supported(sample_model_spec, None)
            assert is_supp is False
            assert reason is not None
            assert "UNSUPPORTED" in reason
            assert "bitsandbytes" in reason

    def test_unsupported_quantization_fp8_on_non_cdna3(self, sample_model_spec: ModelSpec) -> None:
        recipe = QuantizationRecipe(OptimizationStrategy.FP8)
        # RDNA3 GPU (gfx1100) does not support native AMD FP8 CDNA3 kernels
        rdna_gpu = HardwareSpec(
            gpu_present=True,
            device_id=0,
            device_name="AMD Radeon RX 7900 XTX",
            gfx_target="gfx1100",
        )
        is_supp, reason = recipe.is_supported(sample_model_spec, rdna_gpu)
        assert is_supp is False
        assert reason is not None
        assert "UNSUPPORTED" in reason
        assert "gfx942" in reason

    def test_torch_compile_recipe(self, sample_model_spec: ModelSpec) -> None:
        recipe = TorchCompileRecipe(mode="max-autotune")
        assert recipe.runtime_flags["torch_compile"] is True
        assert recipe.runtime_flags["compile_mode"] == "max-autotune"
        is_supp, _ = recipe.is_supported(sample_model_spec, None)
        assert is_supp is True


# ---------------------------------------------------------------------------
# 3. Comparison Engine
# ---------------------------------------------------------------------------

class TestComparisonEngine:
    def test_compare_model_id_or_revision_mismatch_raises(self) -> None:
        engine = ComparisonEngine()
        baseline = OptimizationBaseline(
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
            revision=SAMPLE_COMMIT_SHA,
            status=CandidateStatus.CONFIG_ONLY,
            measured=False,
        )
        cand_diff_model = OptimizationCandidate(
            candidate_id="cand-001",
            model_id="meta-llama/Llama-3-8B",
            revision=SAMPLE_COMMIT_SHA,
            strategy=OptimizationStrategy.FP16,
            status=CandidateStatus.PLANNED,
        )
        with pytest.raises(IncomparableResultsError) as exc_info:
            engine.compare(baseline, cand_diff_model)
        assert "Model ID mismatch" in str(exc_info.value)

        cand_diff_rev = OptimizationCandidate(
            candidate_id="cand-002",
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
            revision="1111222233334444555566667777888899990000",
            strategy=OptimizationStrategy.FP16,
            status=CandidateStatus.PLANNED,
        )
        with pytest.raises(IncomparableResultsError) as exc_info:
            engine.compare(baseline, cand_diff_rev)
        assert "Revision mismatch" in str(exc_info.value)

    def test_compare_not_measured_when_baseline_unmeasured(self) -> None:
        engine = ComparisonEngine()
        baseline = OptimizationBaseline(
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
            revision=SAMPLE_COMMIT_SHA,
            status=CandidateStatus.CONFIG_ONLY,
            measured=False,
        )
        cand = OptimizationCandidate(
            candidate_id="cand-bf16",
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
            revision=SAMPLE_COMMIT_SHA,
            strategy=OptimizationStrategy.BF16,
            status=CandidateStatus.CONFIG_ONLY,
        )
        res = engine.compare(baseline, cand)
        assert res.verdict == ComparisonVerdict.NOT_MEASURED
        assert res.ttft_speedup is None
        assert res.throughput_speedup is None
        assert res.vram_reduction_mb is None

    def test_no_fabricated_metrics_or_speedup(self) -> None:
        engine = ComparisonEngine()
        baseline = OptimizationBaseline(
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
            revision=SAMPLE_COMMIT_SHA,
            status=CandidateStatus.PREPARED,
            measured=False,
            benchmark_result=None,
        )
        cand = OptimizationCandidate(
            candidate_id="cand-003",
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
            revision=SAMPLE_COMMIT_SHA,
            strategy=OptimizationStrategy.TORCH_COMPILE,
            status=CandidateStatus.PREPARED,
            benchmark_result=None,
        )
        res = engine.compare(baseline, cand)
        assert res.verdict == ComparisonVerdict.NOT_MEASURED
        assert res.ttft_speedup is None
        assert res.throughput_speedup is None

    def test_compare_honest_speedup(self) -> None:
        engine = ComparisonEngine()
        b_bench = BenchmarkResult(
            schema_version="1.0.0",
            status=ExecutionStatus.SUCCESS,
            ttft_ms=50.0,
            throughput_tokens_per_sec=100.0,
            peak_vram_used_mb=2048.0,
        )
        c_bench = BenchmarkResult(
            schema_version="1.0.0",
            status=ExecutionStatus.SUCCESS,
            ttft_ms=40.0,  # 1.25x faster TTFT
            throughput_tokens_per_sec=125.0,  # 1.25x faster throughput
            peak_vram_used_mb=1800.0,  # 248 MB reduction
        )
        baseline = OptimizationBaseline(
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
            revision=SAMPLE_COMMIT_SHA,
            status=CandidateStatus.EXECUTED,
            measured=True,
            benchmark_result=b_bench,
        )
        cand = OptimizationCandidate(
            candidate_id="cand-bf16",
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
            revision=SAMPLE_COMMIT_SHA,
            strategy=OptimizationStrategy.BF16,
            status=CandidateStatus.EXECUTED,
            benchmark_result=c_bench,
        )
        res = engine.compare(baseline, cand)
        assert res.verdict == ComparisonVerdict.IMPROVED
        assert res.throughput_speedup == 1.25
        assert res.ttft_speedup == 1.25
        assert res.vram_reduction_mb == 248.0

    def test_compare_quality_regression_verdict(self) -> None:
        engine = ComparisonEngine(quality_threshold=0.95)
        b_bench = BenchmarkResult(
            schema_version="1.0.0",
            status=ExecutionStatus.SUCCESS,
            ttft_ms=50.0,
            throughput_tokens_per_sec=100.0,
            peak_vram_used_mb=2048.0,
        )
        c_bench = BenchmarkResult(
            schema_version="1.0.0",
            status=ExecutionStatus.SUCCESS,
            ttft_ms=30.0,
            throughput_tokens_per_sec=150.0,  # faster!
            peak_vram_used_mb=1000.0,
        )
        val_rep = ValidationReport(
            schema_version="1.0.0",
            mode=ValidationMode.COMPARISON,
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
            baseline_revision=SAMPLE_COMMIT_SHA,
            candidate_revision=SAMPLE_COMMIT_SHA,
            verdict=ValidationVerdict.FAIL,
            correctness_passed=True,
            quality_measured=True,
            qrr_percent=85.0,  # 85% is below 95% threshold!
        )
        baseline = OptimizationBaseline(
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
            revision=SAMPLE_COMMIT_SHA,
            status=CandidateStatus.EXECUTED,
            measured=True,
            benchmark_result=b_bench,
        )
        cand = OptimizationCandidate(
            candidate_id="cand-corrupted",
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
            revision=SAMPLE_COMMIT_SHA,
            strategy=OptimizationStrategy.INT8,
            status=CandidateStatus.EXECUTED,
            benchmark_result=c_bench,
            validation_report=val_rep,
        )
        res = engine.compare(baseline, cand)
        assert res.verdict == ComparisonVerdict.REGRESSED
        assert res.qrr_percent == 85.0
        assert any("Quality regression" in r for r in res.reasons)


# ---------------------------------------------------------------------------
# 4. Optimization Executor and Pipeline
# ---------------------------------------------------------------------------

class TestOptimizationExecutor:
    def test_optimization_executor_on_mac_safely_stops(
        self,
        tmp_path: Path,
        sample_model_spec: ModelSpec,
        non_amd_report: DetectionReport,
    ) -> None:
        out_dir = tmp_path / "opt_test_run"

        mock_insp = MagicMock()
        mock_insp.inspect.return_value = sample_model_spec

        mock_obs = MagicMock()
        mock_obs.observe.return_value = non_amd_report

        mock_mat = MagicMock()
        mock_mat.check_disk_space.return_value = None
        mock_mat.materialize.return_value = MaterializedModel(
            local_path=str(tmp_path / "mock_weights"),
            mode=MaterializationMode.METADATA_ONLY,
            files=["config.json"],
            has_weights=False,
            weights_size_bytes=0,
            cached=True,
        )

        with patch("rocmhub.optimization.baseline.ModelInspector", return_value=mock_insp), \
             patch("rocmhub.optimization.executor.ModelInspector", return_value=mock_insp), \
             patch("rocmhub.forge.planner.ModelInspector", return_value=mock_insp), \
             patch("rocmhub.forge.executor.ModelInspector", return_value=mock_insp), \
             patch("rocmhub.optimization.baseline.SystemObserver", return_value=mock_obs), \
             patch("rocmhub.optimization.executor.SystemObserver", return_value=mock_obs), \
             patch("rocmhub.forge.planner.SystemObserver", return_value=mock_obs), \
             patch("rocmhub.forge.executor.SystemObserver", return_value=mock_obs), \
             patch("rocmhub.forge.executor.ModelMaterializer", return_value=mock_mat):

            req = OptimizationRequest(
                model_id=sample_model_spec.model_id,
                revision=SAMPLE_COMMIT_SHA,
                output_dir=str(out_dir),
                strategies=[OptimizationStrategy.BF16, OptimizationStrategy.FP16],
                allow_full_weights=False,
            )
            executor = OptimizationExecutor(inspector=mock_insp, observer=mock_obs)
            report = executor.execute(req)

            assert report.status == "CONFIG_ONLY"
            assert report.model_id == sample_model_spec.model_id
            assert report.baseline.measured is False
            assert report.baseline.benchmark_result is None
            assert len(report.candidates) == 2
            assert all(c.status == CandidateStatus.CONFIG_ONLY for c in report.candidates)
            assert all(comp.verdict == ComparisonVerdict.NOT_MEASURED for comp in report.comparisons)
            assert report.best_candidate_id is None

    def test_optimization_executor_respects_max_candidates(
        self,
        tmp_path: Path,
        sample_model_spec: ModelSpec,
        non_amd_report: DetectionReport,
    ) -> None:
        out_dir = tmp_path / "opt_limit_test"

        mock_insp = MagicMock()
        mock_insp.inspect.return_value = sample_model_spec
        mock_obs = MagicMock()
        mock_obs.observe.return_value = non_amd_report

        with patch("rocmhub.optimization.baseline.ModelInspector", return_value=mock_insp), \
             patch("rocmhub.optimization.executor.ModelInspector", return_value=mock_insp), \
             patch("rocmhub.forge.planner.ModelInspector", return_value=mock_insp), \
             patch("rocmhub.forge.executor.ModelInspector", return_value=mock_insp), \
             patch("rocmhub.optimization.baseline.SystemObserver", return_value=mock_obs), \
             patch("rocmhub.optimization.executor.SystemObserver", return_value=mock_obs), \
             patch("rocmhub.forge.planner.SystemObserver", return_value=mock_obs), \
             patch("rocmhub.forge.executor.SystemObserver", return_value=mock_obs):

            req = OptimizationRequest(
                model_id=sample_model_spec.model_id,
                revision=SAMPLE_COMMIT_SHA,
                output_dir=str(out_dir),
                strategies=[
                    OptimizationStrategy.BF16,
                    OptimizationStrategy.FP16,
                    OptimizationStrategy.FP32,
                    OptimizationStrategy.TORCH_COMPILE,
                ],
                max_candidates=2,  # limit to 2
                allow_full_weights=False,
            )
            executor = OptimizationExecutor(inspector=mock_insp, observer=mock_obs)
            report = executor.execute(req)
            assert len(report.candidates) == 2


# ---------------------------------------------------------------------------
# 5. Security and Sandboxing
# ---------------------------------------------------------------------------

class TestOptimizationSecurity:
    def test_llm_cannot_override_execution_verdict(self) -> None:
        # Verify that comparison verdict is solely derived from verified BenchmarkResult data
        engine = ComparisonEngine()
        baseline = OptimizationBaseline(
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
            revision=SAMPLE_COMMIT_SHA,
            status=CandidateStatus.CONFIG_ONLY,
            measured=False,
        )
        cand = OptimizationCandidate(
            candidate_id="cand-bf16",
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
            revision=SAMPLE_COMMIT_SHA,
            strategy=OptimizationStrategy.BF16,
            status=CandidateStatus.CONFIG_ONLY,
        )
        # Even if someone claimed IMPROVED in candidate metadata or prompt
        res = engine.compare(baseline, cand)
        assert res.verdict == ComparisonVerdict.NOT_MEASURED

    def test_tool_registry_optimization_tools(self, tmp_path: Path, sample_model_spec: ModelSpec) -> None:
        registry = ToolRegistry()
        # Test create_optimization_plan tool
        mock_insp = MagicMock()
        mock_insp.inspect.return_value = sample_model_spec
        registry._inspector = mock_insp

        plan_res = registry.invoke("create_optimization_plan", {
            "model_id": sample_model_spec.model_id,
            "revision": SAMPLE_COMMIT_SHA,
            "strategies": ["bf16", "fp16"],
        })
        assert plan_res["status"] == "SUCCESS"
        assert len(plan_res["candidates"]) == 2

        # Test path traversal protection on build_candidate
        with pytest.raises(SecurityBoundaryError):
            registry.invoke("build_candidate", {
                "model_id": sample_model_spec.model_id,
                "output_dir": "/etc/malicious_build",
            })


# ---------------------------------------------------------------------------
# 6. CLI Command: rocmhub optimize
# ---------------------------------------------------------------------------

class TestCLIOptimizeCommand:
    def test_cli_optimize_missing_model_id(self) -> None:
        exit_code = main(["optimize"])
        assert exit_code == 1

    def test_cli_optimize_runs_cleanly_on_mac(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        sample_model_spec: ModelSpec,
        non_amd_report: DetectionReport,
    ) -> None:
        out_dir = tmp_path / "cli_opt_dir"

        mock_insp = MagicMock()
        mock_insp.inspect.return_value = sample_model_spec
        mock_obs = MagicMock()
        mock_obs.observe.return_value = non_amd_report

        with patch("rocmhub.optimization.baseline.ModelInspector", return_value=mock_insp), \
             patch("rocmhub.optimization.executor.ModelInspector", return_value=mock_insp), \
             patch("rocmhub.forge.planner.ModelInspector", return_value=mock_insp), \
             patch("rocmhub.forge.executor.ModelInspector", return_value=mock_insp), \
             patch("rocmhub.optimization.baseline.SystemObserver", return_value=mock_obs), \
             patch("rocmhub.optimization.executor.SystemObserver", return_value=mock_obs), \
             patch("rocmhub.forge.planner.SystemObserver", return_value=mock_obs), \
             patch("rocmhub.forge.executor.SystemObserver", return_value=mock_obs):

            exit_code = main([
                "optimize", "Qwen/Qwen2.5-0.5B-Instruct",
                "--output-dir", str(out_dir),
                "--revision", SAMPLE_COMMIT_SHA,
            ])
            assert exit_code == 0
            captured = capsys.readouterr()
            assert "ROCmHub Optimization Engine Report" in captured.out
            assert "Baseline Reference:" in captured.out
            assert "Optimization Candidates:" in captured.out
            assert "Comparative Evaluation:" in captured.out
            assert "NOT_MEASURED" in captured.out

    def test_cli_optimize_json_output(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        sample_model_spec: ModelSpec,
        non_amd_report: DetectionReport,
    ) -> None:
        out_dir = tmp_path / "cli_opt_json_dir"

        mock_insp = MagicMock()
        mock_insp.inspect.return_value = sample_model_spec
        mock_obs = MagicMock()
        mock_obs.observe.return_value = non_amd_report

        with patch("rocmhub.optimization.baseline.ModelInspector", return_value=mock_insp), \
             patch("rocmhub.optimization.executor.ModelInspector", return_value=mock_insp), \
             patch("rocmhub.forge.planner.ModelInspector", return_value=mock_insp), \
             patch("rocmhub.forge.executor.ModelInspector", return_value=mock_insp), \
             patch("rocmhub.optimization.baseline.SystemObserver", return_value=mock_obs), \
             patch("rocmhub.optimization.executor.SystemObserver", return_value=mock_obs), \
             patch("rocmhub.forge.planner.SystemObserver", return_value=mock_obs), \
             patch("rocmhub.forge.executor.SystemObserver", return_value=mock_obs):

            exit_code = main([
                "optimize", "Qwen/Qwen2.5-0.5B-Instruct",
                "--output-dir", str(out_dir),
                "--revision", SAMPLE_COMMIT_SHA,
                "--json",
            ])
            assert exit_code == 0
            captured = capsys.readouterr()
            data = json.loads(captured.out)
            assert data["model_id"] == "Qwen/Qwen2.5-0.5B-Instruct"
            assert data["status"] == "CONFIG_ONLY"
            assert len(data["candidates"]) > 0
            assert len(data["comparisons"]) > 0
            assert all(c["verdict"] == "NOT_MEASURED" for c in data["comparisons"])
