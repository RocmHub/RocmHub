"""Integration test with live Hugging Face Hub for Qwen/Qwen2.5-0.5B-Instruct.

Marked as 'network' and 'integration' so that offline unit test runs exclude it.
"""

from __future__ import annotations

import pytest

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
