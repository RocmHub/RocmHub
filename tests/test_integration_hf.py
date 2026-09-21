"""Integration test with live Hugging Face Hub for Qwen/Qwen2.5-0.5B-Instruct.

Marked as 'network' and 'integration' so that offline unit test runs exclude it.
"""

from __future__ import annotations

import json

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


