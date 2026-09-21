"""Base abstractions and protocols for model sources."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@dataclass(frozen=True)
class RepositoryMetadata:
    """Metadata describing a remote or local model repository without weight tensors."""

    model_id: str
    resolved_commit_sha: str
    files: List[str] = field(default_factory=list)
    card_data: Dict[str, Any] = field(default_factory=dict)
    safetensors_metadata: Optional[Dict[str, Any]] = None
    pipeline_tag: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@runtime_checkable
class ModelSource(Protocol):
    """Abstract interface for model sources (e.g. Hugging Face Hub, local directory, ModelScope)."""

    @property
    def source_name(self) -> str:
        """Name of this model source provider (e.g. 'huggingface')."""
        ...

    def resolve_revision(self, model_id: str, revision: str = "main") -> str:
        """Resolve a requested revision (branch, tag, or partial SHA) to an immutable 40-character Git commit SHA.

        Raises:
            ModelNotFoundError: If model repository does not exist.
            AuthRequiredError: If authentication is required (gated/private).
            RevisionNotFoundError: If the specified revision does not exist.
            NetworkError: If network communication fails.
        """
        ...

    def get_repository_metadata(self, model_id: str, revision: str = "main") -> RepositoryMetadata:
        """Fetch repository file listings and hub-level metadata without downloading weight tensors.

        Raises:
            ModelNotFoundError: If model repository does not exist.
            AuthRequiredError: If authentication is required.
            RevisionNotFoundError: If revision does not exist.
            NetworkError: If network communication fails.
        """
        ...

    def fetch_metadata_file(
        self, model_id: str, filename: str, revision: str = "main"
    ) -> Optional[str]:
        """Fetch the string content of a lightweight metadata or configuration file (e.g. config.json).

        Returns None if the file does not exist in the repository.

        Raises:
            ModelNotFoundError: If model repository does not exist.
            AuthRequiredError: If authentication is required.
            RevisionNotFoundError: If revision does not exist.
            NetworkError: If network communication fails.
        """
        ...
