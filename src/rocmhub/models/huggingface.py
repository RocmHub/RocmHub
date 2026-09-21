"""Hugging Face Hub model source implementation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from urllib.error import URLError

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import (
    EntryNotFoundError,
    GatedRepoError,
    HfHubHTTPError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

from rocmhub.core.errors import (
    AuthRequiredError,
    ModelNotFoundError,
    NetworkError,
)
from rocmhub.core.errors import (
    RevisionNotFoundError as ROCmHubRevisionNotFoundError,
)
from rocmhub.models.base import ModelSource, RepositoryMetadata


class HuggingFaceModelSource(ModelSource):
    """ModelSource adapter utilizing the Hugging Face Hub API.

    Guarantees:
    - Never executes remote model code (trust_remote_code=False).
    - Translates Hub-specific exceptions into ROCmHub domain exceptions.
    - Resolves requested revisions to immutable 40-character Git commit SHAs.
    - Lightweight: never downloads model weight tensors.
    """

    def __init__(self, api: Optional[HfApi] = None, token: Optional[str] = None) -> None:
        self._api = api or HfApi(token=token)
        self._token = token

    @property
    def source_name(self) -> str:
        return "huggingface"

    def _translate_error(self, exc: Exception, model_id: str, revision: str) -> Exception:
        """Map Hugging Face Hub exceptions into ROCmHub domain exceptions."""
        if isinstance(exc, GatedRepoError):
            return AuthRequiredError(
                f"Model '{model_id}' is gated or private and requires authentication.",
                details={"model_id": model_id, "revision": revision, "original_error": str(exc)},
            )
        if isinstance(exc, RepositoryNotFoundError):
            return ModelNotFoundError(
                f"Model repository '{model_id}' was not found on Hugging Face Hub.",
                details={"model_id": model_id, "revision": revision},
            )
        if isinstance(exc, RevisionNotFoundError):
            return ROCmHubRevisionNotFoundError(
                f"Revision '{revision}' for model '{model_id}' was not found.",
                details={"model_id": model_id, "revision": revision},
            )
        if isinstance(exc, HfHubHTTPError):
            status_code = getattr(exc.response, "status_code", None) if hasattr(exc, "response") else None
            if status_code in (401, 403):
                return AuthRequiredError(
                    f"Authentication required to access model '{model_id}' (HTTP {status_code}).",
                    details={"model_id": model_id, "revision": revision, "status_code": status_code},
                )
            if status_code == 404:
                return ModelNotFoundError(
                    f"Model or revision '{model_id}@{revision}' not found (HTTP 404).",
                    details={"model_id": model_id, "revision": revision},
                )
            return NetworkError(
                f"HTTP communication failure when contacting Hugging Face Hub: {exc}",
                details={"model_id": model_id, "status_code": status_code},
            )
        if isinstance(exc, (URLError, ConnectionError, OSError)) and "network" in str(exc).lower():
            return NetworkError(
                f"Network failure while reaching Hugging Face Hub: {exc}",
                details={"model_id": model_id},
            )
        return exc

    def resolve_revision(self, model_id: str, revision: str = "main") -> str:
        """Resolve requested branch/tag/revision to immutable 40-character commit SHA."""
        try:
            info = self._api.model_info(repo_id=model_id, revision=revision)
            sha: Optional[str] = getattr(info, "sha", None)
            if not sha or len(sha) != 40:
                raise ROCmHubRevisionNotFoundError(
                    f"Could not resolve an immutable 40-character commit SHA for '{model_id}@{revision}'",
                    details={"model_id": model_id, "revision": revision, "resolved_sha": sha},
                )
            return sha.lower()
        except Exception as exc:
            translated = self._translate_error(exc, model_id, revision)
            if translated is not exc:
                raise translated from exc
            raise

    def get_repository_metadata(self, model_id: str, revision: str = "main") -> RepositoryMetadata:
        """Fetch repository file listings and hub-level metadata without downloading weights."""
        try:
            info = self._api.model_info(repo_id=model_id, revision=revision)
            sha: str = getattr(info, "sha", revision).lower()
            files: List[str] = [s.rfilename for s in (info.siblings or [])]

            safetensors_meta: Optional[Dict[str, Any]] = None
            if hasattr(info, "safetensors") and info.safetensors is not None:
                st = info.safetensors
                if hasattr(st, "total"):
                    safetensors_meta = {
                        "total": st.total,
                        "parameters": getattr(st, "parameters", {}),
                    }
                elif isinstance(st, dict):
                    safetensors_meta = st

            card_data = getattr(info, "card_data", {})
            if not isinstance(card_data, dict):
                card_data = {}

            return RepositoryMetadata(
                model_id=model_id,
                resolved_commit_sha=sha,
                files=files,
                card_data=card_data,
                safetensors_metadata=safetensors_meta,
                pipeline_tag=getattr(info, "pipeline_tag", None),
                tags=list(getattr(info, "tags", []) or []),
            )
        except Exception as exc:
            translated = self._translate_error(exc, model_id, revision)
            if translated is not exc:
                raise translated from exc
            raise

    def fetch_metadata_file(
        self, model_id: str, filename: str, revision: str = "main"
    ) -> Optional[str]:
        """Fetch content of a lightweight metadata/config file without loading tensors."""
        try:
            local_path = hf_hub_download(
                repo_id=model_id,
                filename=filename,
                revision=revision,
                token=self._token,
            )
            with open(local_path, "r", encoding="utf-8") as f:
                return f.read()
        except EntryNotFoundError:
            return None
        except Exception as exc:
            translated = self._translate_error(exc, model_id, revision)
            if translated is not exc:
                raise translated from exc
            raise
