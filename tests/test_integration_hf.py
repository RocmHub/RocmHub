"""Integration test with live Hugging Face Hub for Qwen/Qwen2.5-0.5B-Instruct.

Marked as 'network' and 'integration' so that offline unit test runs exclude it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rocmhub.cli.main import main
from rocmhub.core.types import ModelSpec
from rocmhub.models.huggingface import HuggingFaceModelSource
from rocmhub.models.inspector import ModelInspector


@pytest.mark.network
@pytest.mark.integration
def test_live_hf_inspection_qwen_model() -> None:
    """Live integration test querying real Hugging Face Hub metadata for Qwen2.5-0.5B."""
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    source = HuggingFaceModelSource()
    inspector = ModelInspector(source=source)

    spec: ModelSpec = inspector.inspect(model_id=model_id, revision="main")

    # 1. Verify immutable commit SHA (must be 40-character hex string)
    assert len(spec.commit_sha) == 40
    assert all(c in "0123456789abcdef" for c in spec.commit_sha)

    # 2. Verify model identifier and source
    assert spec.model_id == model_id
    assert spec.source == "huggingface"
    assert spec.requested_revision == "main"

    # 3. Verify static architecture extraction
    assert spec.architecture == "Qwen2ForCausalLM"

    # 4. Verify context length
    assert spec.context_length == 32768

    # 5. Verify default dtype
    assert spec.default_dtype == "bfloat16"

    # 6. Verify weights format
    assert spec.weights_format == "safetensors"

    # 7. Verify parameter count (derived from server-side safetensors metadata without downloading weights)
    assert spec.parameter_count is not None
    assert spec.parameter_count > 400_000_000  # ~494M parameters


@pytest.mark.network
@pytest.mark.integration
def test_live_check_qwen_model_on_current_mac(capsys: pytest.CaptureFixture[str]) -> None:
    """Live integration test: check Qwen2.5-0.5B against real local environment (Mac / diagnostic).

    Expected behavior:
    - Real Hugging Face inspection succeeds without downloading weights.
    - Real SystemObserver observes current Mac host.
    - CapabilityEvaluator evaluates system as NO_ACCELERATOR.
    - CLI exits with code 2.
    """
    import json

    from rocmhub.cli.main import main

    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    exit_code = main(["check", model_id, "--json"])

    # On macOS Darwin without AMD GPU, exit code must be 2 (NO_ACCELERATOR)
    assert exit_code == 2

    captured = capsys.readouterr()
    report_dict = json.loads(captured.out)

    assert report_dict["verdict"] == "NO_ACCELERATOR"
    assert report_dict["model"]["model_id"] == model_id
    assert report_dict["model"]["architecture"] == "Qwen2ForCausalLM"
    assert report_dict["capabilities"]["amd_gpu_present"] is False
    assert report_dict["capabilities"]["baseline_runtime_candidate"] is None

    reason_codes = [r["code"] for r in report_dict["reasons"]]
    assert "NO_AMD_GPU" in reason_codes
    assert "ROCM_NOT_DETECTED" in reason_codes


@pytest.mark.network
@pytest.mark.integration
def test_live_run_qwen_model_on_current_mac_stops_at_preflight(capsys: pytest.CaptureFixture[str]) -> None:
    """Live integration test: rocmhub run stops at preflight on Mac without downloading weights."""
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    exit_code = main(["run", model_id, "--prompt", "Hello", "--max-new-tokens", "8", "--json"])

    # Must exit with code 2 (NO_ACCELERATOR)
    assert exit_code == 2

    captured = capsys.readouterr()
    run_dict = json.loads(captured.out)

    assert run_dict["status"] == "SKIPPED"
    assert run_dict["model_id"] == model_id
    assert run_dict["generated_text"] is None
    assert run_dict["input_tokens"] is None
    assert run_dict["generated_tokens"] is None
    assert "NO_ACCELERATOR" in run_dict["error"]


@pytest.mark.network
@pytest.mark.integration
def test_live_benchmark_qwen_model_on_current_mac_stops_at_preflight(capsys: pytest.CaptureFixture[str]) -> None:
    """Live integration test: rocmhub benchmark stops at preflight on Mac without downloading weights."""
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    exit_code = main(["benchmark", model_id, "--prompt", "Hello", "--max-new-tokens", "8", "--json"])

    # Must exit with code 2 (NO_ACCELERATOR)
    assert exit_code == 2

    captured = capsys.readouterr()
    bench_dict = json.loads(captured.out)

    assert bench_dict["status"] == "SKIPPED"
    assert bench_dict["model_id"] == model_id
    assert bench_dict["ttft_ms"] is None
    assert bench_dict["itl_ms_mean"] is None
    assert bench_dict["throughput_tokens_per_sec"] is None
    assert bench_dict["peak_vram_used_mb"] is None
    assert bench_dict["total_latency_ms"] is None
    assert "NO_ACCELERATOR" in bench_dict["error_message"]




@pytest.mark.network
@pytest.mark.integration
def test_live_validate_qwen_model_on_current_mac_stops_at_preflight(capsys: pytest.CaptureFixture[str]) -> None:
    """Live integration test: rocmhub validate stops at preflight on Mac without downloading weights.

    Verifies:
    - Real Hugging Face Hub metadata is fetched (commit SHA resolved).
    - Real SystemObserver observes Mac host (no AMD GPU).
    - CapabilityEvaluator verdict is NO_ACCELERATOR.
    - CLI exits with code 2 (NOT_MEASURED).
    - ValidationReport has verdict=NOT_MEASURED, all metrics None.
    - No model weights were downloaded (cases_completed == 0).
    """
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    exit_code = main(["validate", model_id, "--json"])

    # On macOS without AMD GPU, exit code must be 2 (NOT_MEASURED)
    assert exit_code == 2

    captured = capsys.readouterr()
    report_dict = json.loads(captured.out)

    assert report_dict["verdict"] == "NOT_MEASURED"
    assert report_dict["model_id"] == model_id
    assert report_dict["mode"] == "SELF_VALIDATION"
    assert report_dict["correctness_passed"] is None
    assert report_dict["quality_measured"] is False
    assert report_dict["qrr_percent"] is None
    assert report_dict["cases_completed"] == 0
    assert report_dict["cases_failed"] == 0
    assert report_dict["critical_cases_failed"] == 0
    assert len(report_dict["case_results"]) == 0

    # Baseline revision must be a real immutable commit SHA (40 hex chars)
    assert len(report_dict["baseline_revision"]) == 40
    assert all(c in "0123456789abcdef" for c in report_dict["baseline_revision"])

    # Reasons must mention the skip
    reasons_text = " ".join(report_dict["reasons"])
    assert (
        "NO_ACCELERATOR" in reasons_text
        or "preflight" in reasons_text.lower()
        or "skipped" in reasons_text.lower()
    )


@pytest.mark.network
@pytest.mark.integration
def test_live_artifact_build_and_verify_qwen_model_on_current_mac(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Live integration test: rocmhub artifact build & verify for Qwen2.5-0.5B on Mac.

    Verifies:
    - Real Hugging Face Hub metadata is fetched (40-char commit SHA resolved).
    - Real SystemObserver observes Mac host (no AMD GPU).
    - Preflight evaluation produces NO_ACCELERATOR.
    - Zero weights are downloaded, zero inference executed.
    - Artifact bundle is saved with status COMPLETE.
    - rocmhub artifact verify validates bundle integrity (valid=True).
    """
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    out_dir = tmp_path / "artifacts"
    exit_code = main(["artifact", "build", model_id, "--output-dir", str(out_dir), "--json"])
    assert exit_code == 0

    captured = capsys.readouterr()
    manifest_data = json.loads(captured.out)

    assert manifest_data["status"] == "COMPLETE"
    assert manifest_data["capability_verdict"] == "NO_ACCELERATOR"
    assert manifest_data["execution_status"] == "SKIPPED"
    assert manifest_data["benchmark_status"] == "SKIPPED"
    assert manifest_data["validation_verdict"] == "NOT_MEASURED"
    assert manifest_data["model"]["model_id"] == model_id
    assert len(manifest_data["model"]["immutable_revision"]) == 40
    assert "model.json" in manifest_data["files"]
    assert "checksums.json" not in manifest_data["files"]
    assert "manifest.json" not in manifest_data["files"]

    created_dirs = list(out_dir.iterdir())
    assert len(created_dirs) == 1
    artifact_dir = created_dirs[0]

    # Verify the created artifact bundle using CLI verify command
    verify_code = main(["artifact", "verify", str(artifact_dir), "--json"])
    assert verify_code == 0

    verify_captured = capsys.readouterr()
    verify_data = json.loads(verify_captured.out)
    assert verify_data["valid"] is True
    assert verify_data["manifest_valid"] is True
    assert verify_data["checksums_valid"] is True
    assert len(verify_data["missing_files"]) == 0
    assert len(verify_data["modified_files"]) == 0
    assert len(verify_data["unexpected_files"]) == 0


@pytest.mark.network
@pytest.mark.integration
def test_live_guard_qwen_model_on_current_mac(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Live integration test: rocmhub guard audits diagnostic bundle on Mac and returns NOT_MEASURED (exit 2)."""
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    out_dir = tmp_path / "artifacts"
    build_code = main(["artifact", "build", model_id, "--output-dir", str(out_dir), "--json"])
    assert build_code == 0
    capsys.readouterr()  # clear buffer

    created_dirs = list(out_dir.iterdir())
    assert len(created_dirs) == 1
    artifact_dir = created_dirs[0]

    # 1. Guard audit with JSON output
    guard_code_json = main(["guard", str(artifact_dir), "--json"])
    assert guard_code_json == 2

    captured = capsys.readouterr()
    report_dict = json.loads(captured.out)
    assert report_dict["verdict"] == "NOT_MEASURED"
    assert "NO_BENCHMARK_EXECUTION" in report_dict["reasons"]
    assert report_dict["valid_runs"] == 0
    assert report_dict["ttft_variability"] is None
    assert report_dict["throughput_variability"] is None
    assert report_dict["reference_stable"] is None

    # 2. Guard audit with human table output
    guard_code_text = main(["guard", str(artifact_dir)])
    assert guard_code_text == 2

    captured_text = capsys.readouterr()
    assert "Guard Verdict:             NOT_MEASURED" in captured_text.out
    assert "NO_BENCHMARK_EXECUTION" in captured_text.out


@pytest.mark.network
@pytest.mark.integration
def test_live_forge_plan_qwen_metadata_only(capsys: pytest.CaptureFixture[str]) -> None:
    """Live integration test: rocmhub forge plan resolves Qwen2.5-0.5B metadata without downloading weights."""
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"

    # 1. Test pure JSON output
    exit_code_json = main(["forge", "plan", model_id, "--json"])
    assert exit_code_json == 0

    captured = capsys.readouterr()
    plan_dict = json.loads(captured.out)

    assert plan_dict["model_id"] == model_id
    assert len(plan_dict["revision"]) == 40
    assert all(c in "0123456789abcdef" for c in plan_dict["revision"])
    assert plan_dict["recipe_id"] == "pytorch_transformers_hip"
    assert plan_dict["precision"] == "fp16"
    assert plan_dict["estimated_disk_space_bytes"] > 500_000_000  # ~1.1GB
    assert len(plan_dict["steps"]) == 5
    assert plan_dict["steps"][0]["name"] == "check_prerequisites"
    assert plan_dict["steps"][1]["name"] == "materialize_model"
    assert plan_dict["steps"][4]["name"] == "verify_build"

    # 2. Test human-readable formatted output
    exit_code_text = main(["forge", "plan", model_id])
    assert exit_code_text == 0

    captured_text = capsys.readouterr()
    assert "ROCmHub Model Forge Plan" in captured_text.out
    assert "Plan ID:" in captured_text.out
    assert model_id in captured_text.out
    assert "pytorch_transformers_hip" in captured_text.out
    assert "Planned Build Steps:" in captured_text.out


@pytest.mark.network
@pytest.mark.integration
def test_live_engineer_qwen_model_on_current_mac(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Live integration test: rocmhub engineer autonomous run on Mac for Qwen2.5-0.5B.

    Verifies:
    - Real Hugging Face Hub metadata is fetched (40-character commit SHA resolved).
    - Real SystemObserver detects Mac host.
    - Autonomous AI Engineer executes OBSERVE -> PLAN -> ACT -> EVALUATE loop.
    - Prepares build in CONFIG_ONLY status without downloading weights.
    - amd_validated is False.
    - Zero fake GPU execution metrics.
    - Trajectory and report are saved to disk with zero secrets.
    """
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    build_dir = tmp_path / "live_engineer_build"

    exit_code = main([
        "engineer", model_id,
        "--output-dir", str(build_dir),
        "--json",
    ])
    assert exit_code == 0

    captured = capsys.readouterr()
    report_dict = json.loads(captured.out)

    assert report_dict["status"] == "CONFIG_ONLY"
    assert report_dict["model_id"] == model_id
    assert len(report_dict["revision"]) == 40
    assert all(c in "0123456789abcdef" for c in report_dict["revision"])
    assert report_dict["objective"] == "BASE_PREPARATION"
    assert report_dict["secret_scan_clean"] is True
    assert len(report_dict["trajectory"]) >= 5

    # Check Build Manifest
    manifest_dict = report_dict["build_manifest"]
    assert manifest_dict is not None
    assert manifest_dict["status"] == "CONFIG_ONLY"
    assert manifest_dict["amd_validated"] is False
    assert (build_dir / "build_manifest.json").exists()
    assert (build_dir / "runtime_config.json").exists()
    assert (build_dir / "run_inference.py").exists()


@pytest.mark.network
@pytest.mark.integration
def test_live_optimize_qwen_model_on_current_mac(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Live integration test: rocmhub optimize run on Mac for Qwen2.5-0.5B.

    Verifies:
    - Real Hugging Face Hub metadata is fetched (40-char commit SHA resolved).
    - Real SystemObserver detects Mac host.
    - Baseline and candidate plans are created.
    - Candidates built with CONFIG_ONLY status.
    - AMD GPU execution not performed.
    - Comparative measurements report NOT_MEASURED.
    - Zero fake speedup metrics.
    """
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    opt_dir = tmp_path / "live_optimize_run"

    exit_code = main([
        "optimize", model_id,
        "--output-dir", str(opt_dir),
        "--max-candidates", "2",
        "--json",
    ])
    assert exit_code == 0

    captured = capsys.readouterr()
    report_dict = json.loads(captured.out)

    assert report_dict["model_id"] == model_id
    assert len(report_dict["revision"]) == 40
    assert report_dict["status"] == "CONFIG_ONLY"
    assert len(report_dict["candidates"]) == 2
    assert all(c["status"] == "CONFIG_ONLY" for c in report_dict["candidates"])
    assert len(report_dict["comparisons"]) == 2
    assert all(comp["verdict"] == "NOT_MEASURED" for comp in report_dict["comparisons"])
    assert report_dict["best_candidate_id"] is None


