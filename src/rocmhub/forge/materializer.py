"""ModelMaterializer: downloads and verifies model files from Hub using standard caching."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import List, Optional

from huggingface_hub import snapshot_download
from huggingface_hub.errors import (
    GatedRepoError,
    HfHubHTTPError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

from rocmhub.core.errors import (
    AuthRequiredError,
    InsufficientDiskSpaceError,
    ModelMaterializationError,
    ModelNotFoundError,
    NetworkError,
)
from rocmhub.core.errors import (
    RevisionNotFoundError as ROCmHubRevisionNotFoundError,
)
from rocmhub.core.types import ModelSpec
from rocmhub.forge.base import MaterializedModel

WEIGHT_FILE_EXTENSIONS = {".safetensors", ".bin", ".pt", ".h5", ".pth", ".msgpack"}


class ModelMaterializer:
    """Safely materializes model weights and configuration at an immutable commit SHA."""

    def __init__(self, cache_dir: Optional[Path] = None, token: Optional[str] = None) -> None:
        self._cache_dir = cache_dir
        self._token = token

    def _determine_cache_target(self) -> Path:
        if self._cache_dir is not None:
            return self._cache_dir
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            return Path(hf_home) / "hub"
        return Path.home() / ".cache" / "huggingface" / "hub"

    def check_disk_space(self, target_path: Path, required_bytes: int) -> None:
        """Verify that sufficient free disk space exists at target location."""
        target_path.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(target_path)
        if usage.free < required_bytes:
            raise InsufficientDiskSpaceError(
                f"Insufficient disk space on '{target_path}': "
                f"requires {required_bytes / (1024**3):.2f} GB, "
                f"but only {usage.free / (1024**3):.2f} GB is available.",
                details={
                    "target_path": str(target_path),
                    "required_bytes": required_bytes,
                    "available_bytes": usage.free,
                },
            )

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
                f"Network HTTP error accessing model '{model_id}' (HTTP {status_code}): {exc}",
                details={"model_id": model_id, "revision": revision, "status_code": status_code},
            )
        return ModelMaterializationError(
            f"Failed to materialize model '{model_id}': {exc}",
            details={"model_id": model_id, "revision": revision, "error": str(exc)},
        )

    def materialize(
        self,
        model_spec: ModelSpec,
        required_disk_bytes: int = 0,
        download_weights: bool = True,
    ) -> MaterializedModel:
        """Download or retrieve from local cache the model at the immutable commit SHA.

        Guarantees:
        - Checks available disk space before initiating download.
        - Enforces trust_remote_code=False.
        - Uses immutable commit SHA (model_spec.revision).
        - Handles gated/missing repo errors cleanly.
        """
        cache_target = self._determine_cache_target()

        if required_disk_bytes > 0:
            self.check_disk_space(cache_target, required_disk_bytes)

        # Check if snapshot already existed prior to call
        model_folder_name = f"models--{model_spec.model_id.replace('/', '--')}"
        expected_snapshot_dir = cache_target / model_folder_name / "snapshots" / model_spec.commit_sha
        was_already_cached = expected_snapshot_dir.exists()

        ignore_patterns = None if download_weights else ["*.safetensors", "*.bin", "*.pt", "*.h5", "*.pth"]

        try:
            download_dir = snapshot_download(
                repo_id=model_spec.model_id,
                revision=model_spec.commit_sha,
                cache_dir=self._cache_dir,
                token=self._token,
                ignore_patterns=ignore_patterns,
            )
        except Exception as exc:
            raise self._translate_error(exc, model_spec.model_id, model_spec.commit_sha) from exc

        download_path = Path(download_dir)
        files: List[str] = []
        weights_size_bytes = 0

        for root, _, filenames in os.walk(download_path):
            for filename in filenames:
                full_path = Path(root) / filename
                rel_path = full_path.relative_to(download_path)
                files.append(str(rel_path))
                if full_path.suffix.lower() in WEIGHT_FILE_EXTENSIONS:
                    try:
                        weights_size_bytes += full_path.stat().st_size
                    except OSError:
                        pass

        files.sort()

        return MaterializedModel(
            local_path=str(download_path),
            files=files,
            license_name=getattr(model_spec, "license", None),
            weights_size_bytes=weights_size_bytes,
            cached=was_already_cached,
        )
