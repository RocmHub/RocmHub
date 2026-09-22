"""ModelInspector: static analysis of model architecture, metadata, and configuration."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from rocmhub.core.errors import InvalidModelMetadataError, RemoteCodeRequiredError
from rocmhub.core.types import ModelSpec
from rocmhub.models.base import ModelSource, RepositoryMetadata


class ModelInspector:
    """Inspects remote or local models to construct an immutable, validated ModelSpec.

    Safety & Reliability Rules:
    - Never executes remote Python code (`trust_remote_code=False`).
    - Inspects lightweight metadata without downloading weight tensors.
    - Resolves all revisions to immutable Git commit SHAs.
    - Never guesses context length or parameter count: sets to None if ambiguous.
    """

    def __init__(self, source: ModelSource) -> None:
        self._source = source

    @property
    def source(self) -> ModelSource:
        return self._source

    def _detect_weights_format(self, files: List[str]) -> str:
        """Determine primary weights format from repository file list."""
        has_safetensors = any(f.endswith(".safetensors") for f in files)
        has_bin = any(f.endswith(".bin") for f in files)
        has_gguf = any(f.endswith(".gguf") for f in files)

        if has_safetensors:
            return "safetensors"
        if has_bin:
            return "pytorch_bin"
        if has_gguf:
            return "gguf"
        return "unknown"

    def _extract_architecture(self, config: Dict[str, Any]) -> Optional[str]:
        """Extract model architecture class name statically without executing code."""
        # Check for remote code requirement
        auto_map = config.get("auto_map")
        architectures = config.get("architectures")

        # If architecture is explicitly provided in standard format, return it
        if isinstance(architectures, list) and architectures:
            first_arch = architectures[0]
            if isinstance(first_arch, str) and first_arch.strip():
                return first_arch.strip()

        # If standard architectures are missing but auto_map specifies custom remote modeling code
        if isinstance(auto_map, dict) and any("." in str(v) for v in auto_map.values()):
            raise RemoteCodeRequiredError(
                "Model configuration specifies custom remote code in 'auto_map' without standard "
                "architecture declarations. Inspection requires remote code execution which is "
                "forbidden (trust_remote_code=False).",
                details={"auto_map": auto_map},
            )

        # Fallback to model_type if available
        model_type = config.get("model_type")
        if isinstance(model_type, str) and model_type.strip():
            return model_type.strip()

        return None

    def _extract_context_length(self, config: Dict[str, Any]) -> Optional[int]:
        """Extract context length from known standard configuration fields without guessing."""
        candidates = [
            "max_position_embeddings",
            "max_sequence_length",
            "seq_length",
            "n_positions",
        ]
        for field in candidates:
            val = config.get(field)
            if isinstance(val, int) and val > 0:
                return val
        return None

    def _extract_parameter_count(
        self,
        model_id: str,
        commit_sha: str,
        repo_meta: RepositoryMetadata,
        config: Dict[str, Any],
    ) -> Optional[int]:
        """Extract parameter count from hub metadata or safetensors index without loading tensors."""
        # 1. Check Hub-level safetensors metadata
        st_meta = repo_meta.safetensors_metadata
        if isinstance(st_meta, dict):
            total = st_meta.get("total")
            if isinstance(total, int) and total >= 0:
                return total

        # 2. Check model.safetensors.index.json
        if "model.safetensors.index.json" in repo_meta.files:
            raw_index = self._source.fetch_metadata_file(
                model_id, "model.safetensors.index.json", commit_sha
            )
            if raw_index:
                try:
                    index_data = json.loads(raw_index)
                    metadata = index_data.get("metadata", {})
                    total_params = metadata.get("total_parameters")
                    if isinstance(total_params, int) and total_params >= 0:
                        return total_params
                except json.JSONDecodeError:
                    pass

        # If not determinable safely, return None (never guess)
        return None

    def inspect(self, model_id: str, revision: str = "main") -> ModelSpec:
        """Inspect a model repository and produce an immutable ModelSpec.

        Args:
            model_id: Model repository ID (e.g. 'Qwen/Qwen2.5-0.5B-Instruct')
            revision: Requested revision (branch, tag, or commit SHA)

        Returns:
            Validated ModelSpec instance.

        Raises:
            ModelNotFoundError: If model repository is not found.
            AuthRequiredError: If model is gated or private without authorization.
            RevisionNotFoundError: If revision does not exist.
            NetworkError: On connection/HTTP errors.
            InvalidModelMetadataError: If config file is corrupted.
            RemoteCodeRequiredError: If architecture requires executing untrusted code.
        """
        # 1. Resolve immutable Git commit SHA
        commit_sha = self._source.resolve_revision(model_id, revision)

        # 2. Fetch lightweight repository metadata
        repo_meta = self._source.get_repository_metadata(model_id, commit_sha)

        # 3. Determine weights format
        weights_format = self._detect_weights_format(repo_meta.files)

        # 4. Fetch and parse config.json
        config_data: Dict[str, Any] = {}
        architecture: Optional[str] = None
        context_length: Optional[int] = None
        default_dtype: Optional[str] = None

        raw_config = self._source.fetch_metadata_file(model_id, "config.json", commit_sha)
        if raw_config is not None:
            try:
                parsed = json.loads(raw_config)
                if not isinstance(parsed, dict):
                    raise InvalidModelMetadataError(
                        f"Expected JSON object in config.json for '{model_id}', got {type(parsed).__name__}"
                    )
                config_data = parsed
            except json.JSONDecodeError as exc:
                raise InvalidModelMetadataError(
                    f"Corrupted config.json in model '{model_id}@{commit_sha}': {exc}",
                    details={"model_id": model_id, "commit_sha": commit_sha, "error": str(exc)},
                ) from exc

            # Extract architecture (strictly static)
            architecture = self._extract_architecture(config_data)

            # Extract context length
            context_length = self._extract_context_length(config_data)

            # Extract default dtype
            torch_dtype = config_data.get("torch_dtype")
            if isinstance(torch_dtype, str) and torch_dtype.strip():
                default_dtype = torch_dtype.strip()

        # 5. Extract parameter count safely
        parameter_count = self._extract_parameter_count(
            model_id, commit_sha, repo_meta, config_data
        )

        return ModelSpec(
            model_id=model_id,
            source=self._source.source_name,
            requested_revision=revision or "main",
            commit_sha=commit_sha,
            architecture=architecture,
            parameter_count=parameter_count,
            context_length=context_length,
            default_dtype=default_dtype,
            weights_format=weights_format,
        )
