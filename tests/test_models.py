"""Unit tests for ModelSource interface, HuggingFaceModelSource adapter, and ModelInspector."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pytest
from huggingface_hub.errors import (
    GatedRepoError,
    RepositoryNotFoundError,
)
from huggingface_hub.errors import (
    RevisionNotFoundError as HFRevisionNotFoundError,
)

from rocmhub.core.errors import (
    AuthRequiredError,
    InvalidModelMetadataError,
    ModelNotFoundError,
    RemoteCodeRequiredError,
    RevisionNotFoundError,
)
from rocmhub.models.base import RepositoryMetadata
from rocmhub.models.huggingface import HuggingFaceModelSource
from rocmhub.models.inspector import ModelInspector


class FakeModelSource:
    """Mock ModelSource for hermetic, offline unit testing."""

    def __init__(
        self,
        revisions: Optional[Dict[str, str]] = None,
        files: Optional[List[str]] = None,
        metadata_files: Optional[Dict[str, str]] = None,
        safetensors_metadata: Optional[Dict[str, Any]] = None,
        error_on_resolve: Optional[Exception] = None,
        error_on_metadata: Optional[Exception] = None,
    ) -> None:
        self.revisions = revisions or {"main": "1111222233334444555566667777888899990000"}
        self.files = files or []
        self.metadata_files = metadata_files or {}
        self.safetensors_metadata = safetensors_metadata
        self.error_on_resolve = error_on_resolve
        self.error_on_metadata = error_on_metadata

    @property
    def source_name(self) -> str:
        return "fake_hub"

    def resolve_revision(self, model_id: str, revision: str = "main") -> str:
        if self.error_on_resolve:
            raise self.error_on_resolve
        if revision in self.revisions:
            return self.revisions[revision]
        if len(revision) == 40:
            return revision.lower()
        raise RevisionNotFoundError(f"Revision '{revision}' for model '{model_id}' was not found.")

    def get_repository_metadata(self, model_id: str, revision: str = "main") -> RepositoryMetadata:
        if self.error_on_metadata:
            raise self.error_on_metadata
        return RepositoryMetadata(
            model_id=model_id,
            resolved_commit_sha=revision,
            files=self.files,
            safetensors_metadata=self.safetensors_metadata,
        )

    def fetch_metadata_file(
        self, model_id: str, filename: str, revision: str = "main"
    ) -> Optional[str]:
        return self.metadata_files.get(filename)


class TestModelInspector:
    """Unit tests for ModelInspector business logic with isolated mock sources."""

    def test_main_revision_resolves_to_immutable_commit_sha(self) -> None:
        source = FakeModelSource(
            revisions={"main": "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"},
            files=["config.json", "model.safetensors"],
            metadata_files={"config.json": json.dumps({"architectures": ["Qwen2ForCausalLM"]})},
        )
        inspector = ModelInspector(source=source)
        spec = inspector.inspect("Qwen/Qwen2.5-0.5B-Instruct", revision="main")

        assert spec.requested_revision == "main"
        assert spec.commit_sha == "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"
        assert spec.architecture == "Qwen2ForCausalLM"

    def test_explicit_commit_sha_preserved(self) -> None:
        sha = "9999888877776666555544443333222211110000"
        source = FakeModelSource(
            files=["config.json", "model.safetensors"],
            metadata_files={"config.json": json.dumps({"architectures": ["LlamaForCausalLM"]})},
        )
        inspector = ModelInspector(source=source)
        spec = inspector.inspect("test/model", revision=sha)

        assert spec.requested_revision == sha
        assert spec.commit_sha == sha

    def test_architecture_extraction_and_fallback(self) -> None:
        # 1. Standard architectures list
        source1 = FakeModelSource(
            metadata_files={"config.json": json.dumps({"architectures": ["MistralForCausalLM"]})},
        )
        spec1 = ModelInspector(source=source1).inspect("test/model")
        assert spec1.architecture == "MistralForCausalLM"

        # 2. Fallback to model_type
        source2 = FakeModelSource(
            metadata_files={"config.json": json.dumps({"model_type": "falcon"})},
        )
        spec2 = ModelInspector(source=source2).inspect("test/model")
        assert spec2.architecture == "falcon"

    def test_parameter_count_from_safetensors_metadata(self) -> None:
        source = FakeModelSource(
            safetensors_metadata={"total": 494_032_768},
            files=["model.safetensors"],
            metadata_files={"config.json": json.dumps({"architectures": ["Qwen2ForCausalLM"]})},
        )
        inspector = ModelInspector(source=source)
        spec = inspector.inspect("Qwen/Qwen2.5-0.5B-Instruct")
        assert spec.parameter_count == 494_032_768

    def test_parameter_count_from_index_json(self) -> None:
        index_json = json.dumps({"metadata": {"total_parameters": 1_100_000_000}})
        source = FakeModelSource(
            files=["model.safetensors.index.json"],
            metadata_files={
                "config.json": json.dumps({"architectures": ["TinyLlamaForCausalLM"]}),
                "model.safetensors.index.json": index_json,
            },
        )
        inspector = ModelInspector(source=source)
        spec = inspector.inspect("TinyLlama/TinyLlama-1.1B")
        assert spec.parameter_count == 1_100_000_000

    def test_parameter_count_unknown_is_none(self) -> None:
        source = FakeModelSource(
            files=["pytorch_model.bin"],
            metadata_files={"config.json": json.dumps({"architectures": ["GPT2LMHeadModel"]})},
        )
        inspector = ModelInspector(source=source)
        spec = inspector.inspect("gpt2")
        assert spec.parameter_count is None

    def test_context_length_extraction(self) -> None:
        source = FakeModelSource(
            metadata_files={"config.json": json.dumps({"max_position_embeddings": 32768})},
        )
        spec = ModelInspector(source=source).inspect("test/model")
        assert spec.context_length == 32768

        # Fallback to seq_length
        source2 = FakeModelSource(
            metadata_files={"config.json": json.dumps({"seq_length": 4096})},
        )
        spec2 = ModelInspector(source=source2).inspect("test/model")
        assert spec2.context_length == 4096

    def test_weights_format_detection(self) -> None:
        # Safetensors
        s1 = FakeModelSource(files=["model-00001-of-00002.safetensors", "model.safetensors.index.json"])
        assert ModelInspector(source=s1).inspect("m1").weights_format == "safetensors"

        # PyTorch bin
        s2 = FakeModelSource(files=["pytorch_model.bin", "config.json"])
        assert ModelInspector(source=s2).inspect("m2").weights_format == "pytorch_bin"

        # GGUF
        s3 = FakeModelSource(files=["qwen-0.5b-q4_k_m.gguf"])
        assert ModelInspector(source=s3).inspect("m3").weights_format == "gguf"

        # Unknown
        s4 = FakeModelSource(files=["README.md", "tokenizer.json"])
        assert ModelInspector(source=s4).inspect("m4").weights_format == "unknown"

    def test_missing_optional_metadata_handled_gracefully(self) -> None:
        source = FakeModelSource(files=["model.safetensors"], metadata_files={})
        spec = ModelInspector(source=source).inspect("sparse/model")
        assert spec.architecture is None
        assert spec.context_length is None
        assert spec.parameter_count is None
        assert spec.default_dtype is None
        assert spec.weights_format == "safetensors"

    def test_invalid_config_json_raises_error(self) -> None:
        source = FakeModelSource(metadata_files={"config.json": "CORRUPTED_NOT_JSON{{"})
        with pytest.raises(InvalidModelMetadataError, match="Corrupted config.json"):
            ModelInspector(source=source).inspect("corrupted/model")

    def test_remote_code_required_raises_error(self) -> None:
        # Config has auto_map pointing to remote file and no standard architectures
        source = FakeModelSource(
            metadata_files={
                "config.json": json.dumps(
                    {
                        "auto_map": {
                            "AutoModelForCausalLM": "custom_modeling.CustomModelClass"
                        }
                    }
                )
            }
        )
        with pytest.raises(RemoteCodeRequiredError, match="trust_remote_code=False"):
            ModelInspector(source=source).inspect("custom/remote-model")

    def test_model_not_found_propagated(self) -> None:
        source = FakeModelSource(error_on_resolve=ModelNotFoundError("Repository not found"))
        with pytest.raises(ModelNotFoundError):
            ModelInspector(source=source).inspect("nonexistent/repo")

    def test_auth_required_propagated(self) -> None:
        source = FakeModelSource(error_on_resolve=AuthRequiredError("Gated repository"))
        with pytest.raises(AuthRequiredError):
            ModelInspector(source=source).inspect("meta-llama/Llama-3")

    def test_revision_not_found_propagated(self) -> None:
        source = FakeModelSource(revisions={"main": "1111222233334444555566667777888899990000"})
        with pytest.raises(RevisionNotFoundError):
            ModelInspector(source=source).inspect("test/model", revision="unknown-branch")


class TestHuggingFaceModelSourceErrorTranslation:
    """Tests for exception translation in HuggingFaceModelSource."""

    def test_translates_gated_repo_error(self) -> None:
        from unittest.mock import MagicMock
        src = HuggingFaceModelSource()
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        hf_err = GatedRepoError("Access restricted", response=mock_resp)
        domain_err = src._translate_error(hf_err, "gated/model", "main")
        assert isinstance(domain_err, AuthRequiredError)
        assert domain_err.error_code == "AUTH_REQUIRED"

    def test_translates_repository_not_found(self) -> None:
        from unittest.mock import MagicMock
        src = HuggingFaceModelSource()
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        hf_err = RepositoryNotFoundError("Repo 404", response=mock_resp)
        domain_err = src._translate_error(hf_err, "missing/model", "main")
        assert isinstance(domain_err, ModelNotFoundError)
        assert domain_err.error_code == "MODEL_NOT_FOUND"

    def test_translates_revision_not_found(self) -> None:
        from unittest.mock import MagicMock
        src = HuggingFaceModelSource()
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        hf_err = HFRevisionNotFoundError("Branch 404", response=mock_resp)
        domain_err = src._translate_error(hf_err, "test/model", "bad-branch")
        assert isinstance(domain_err, RevisionNotFoundError)
        assert domain_err.error_code == "REVISION_NOT_FOUND"
