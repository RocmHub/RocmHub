"""ModelMaterializer: downloads and verifies model files from Hub using standard caching."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import shutil
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

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
from rocmhub.forge.base import MaterializationMode, MaterializedModel

WEIGHT_FILE_EXTENSIONS = {".safetensors", ".bin", ".pt", ".h5", ".pth", ".msgpack"}
_CACHE_LOCKS_GUARD = threading.Lock()
_CACHE_LOCKS: Dict[str, threading.Lock] = {}
_CACHE_MANIFEST = ".rocmhub-cache.json"
_MAX_MANAGED_CACHE_BYTES = 10 * 1024**3
_fcntl = importlib.import_module("fcntl") if importlib.util.find_spec("fcntl") else None
_CACHE_HIT_HASHED_FILES = {
    "config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "tokenizer.model",
    "model.safetensors.index.json", "pytorch_model.bin.index.json",
}


@contextmanager
def _managed_entry_lock(key: str, cache_root: Path) -> Iterator[None]:
    with _CACHE_LOCKS_GUARD:
        thread_lock = _CACHE_LOCKS.setdefault(key, threading.Lock())
    lock_dir = cache_root / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{hashlib.sha256(key.encode()).hexdigest()}.lock"
    with thread_lock, lock_path.open("a+b") as lock_file:
        if _fcntl is not None:
            _fcntl.flock(lock_file.fileno(), _fcntl.LOCK_EX)
        try:
            yield
        finally:
            if _fcntl is not None:
                _fcntl.flock(lock_file.fileno(), _fcntl.LOCK_UN)


class ModelMaterializer:
    """Safely materializes model weights and configuration at an immutable commit SHA."""

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        token: Optional[str] = None,
        *,
        managed_cache: bool = False,
        max_cache_bytes: int = _MAX_MANAGED_CACHE_BYTES,
        progress_callback: Optional[Callable[[str, Dict[str, object]], None]] = None,
        cancellation_check: Optional[Callable[[], bool]] = None,
    ) -> None:
        self._cache_dir = cache_dir
        self._token = token
        self._managed_cache = managed_cache
        self._max_cache_bytes = max_cache_bytes
        self._progress_callback = progress_callback
        self._cancellation_check = cancellation_check

    def _progress(self, phase: str, **details: object) -> None:
        if self._progress_callback:
            self._progress_callback(phase, details)

    def _check_cancelled(self) -> None:
        if self._cancellation_check and self._cancellation_check():
            raise ModelMaterializationError("Model materialization was cancelled before cache publication.")

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _cache_key(model_id: str, revision: str) -> str:
        identity = hashlib.sha256(f"{model_id}@{revision}".encode("utf-8")).hexdigest()
        safe_model = model_id.replace("/", "--").replace("\\", "--")
        return f"{safe_model}-{identity[:16]}"

    def _manifest_files(self, snapshot: Path, full: bool) -> List[Dict[str, object]]:
        inventory: List[Dict[str, object]] = []
        for path in sorted(snapshot.rglob("*")):
            if path.is_file():
                item: Dict[str, object] = {
                    "path": path.relative_to(snapshot).as_posix(),
                    "size_bytes": path.stat().st_size,
                }
                if full or path.name in _CACHE_HIT_HASHED_FILES:
                    item["sha256"] = self._sha256(path)
                inventory.append(item)
        return inventory

    @staticmethod
    def _directory_bytes(path: Path) -> int:
        if not path.exists():
            return 0
        total = 0
        for root, _, filenames in os.walk(path, followlinks=False):
            for filename in filenames:
                try:
                    total += (Path(root) / filename).lstat().st_size
                except OSError:
                    continue
        return total

    def _read_verified_cache(self, entry: Path, model_spec: ModelSpec) -> Optional[Tuple[Path, int, List[str]]]:
        manifest_path = entry / _CACHE_MANIFEST
        if not manifest_path.is_file():
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                manifest.get("schema_version") != 1
                or manifest.get("state") != "COMPLETE"
                or manifest.get("model_id") != model_spec.model_id
                or manifest.get("revision") != model_spec.commit_sha
                or manifest.get("weights_present") is not True
            ):
                return None
            snapshot = entry / "hub" / f"models--{model_spec.model_id.replace('/', '--')}" / "snapshots" / model_spec.commit_sha
            expected = manifest.get("files")
            if not isinstance(expected, list) or not expected:
                return None
            files: List[str] = []
            total = 0
            for item in expected:
                rel = item.get("path")
                if not isinstance(rel, str) or Path(rel).is_absolute() or ".." in Path(rel).parts:
                    return None
                path = snapshot / rel
                if not path.is_file() or path.stat().st_size != item.get("size_bytes") or path.stat().st_size <= 0:
                    return None
                if item.get("sha256") and Path(rel).name in _CACHE_HIT_HASHED_FILES and self._sha256(path) != item["sha256"]:
                    return None
                files.append(rel)
                total += path.stat().st_size
            actual_files = {path.relative_to(snapshot).as_posix() for path in snapshot.rglob("*") if path.is_file()}
            if actual_files != set(files):
                return None
            weight_bytes, has_weights = self._verify_weights(snapshot)
            if not has_weights or total != manifest.get("total_bytes"):
                return None
            return snapshot, weight_bytes, files
        except (OSError, ValueError, TypeError, KeyError, ModelMaterializationError):
            return None

    def _materialize_managed(self, model_spec: ModelSpec, required_disk_bytes: int) -> MaterializedModel:
        if not self._cache_dir:
            raise ModelMaterializationError("Managed model cache requires an Agent workspace cache directory.")
        if len(model_spec.commit_sha) != 40 or any(c not in "0123456789abcdefABCDEF" for c in model_spec.commit_sha):
            raise ModelMaterializationError("Managed full-weight materialization requires an immutable commit SHA.")
        if required_disk_bytes <= 0:
            required_disk_bytes = 2 * 1024**3
        bounded_estimate = int(required_disk_bytes * 1.25)
        if bounded_estimate > self._max_cache_bytes:
            raise InsufficientDiskSpaceError("Estimated model storage exceeds this Agent's managed cache limit.")

        root = self._cache_dir / "models"
        entry = root / self._cache_key(model_spec.model_id, model_spec.commit_sha)
        lock_key = str(entry.resolve())
        with _managed_entry_lock(lock_key, self._cache_dir):
            if root.is_symlink() or entry.is_symlink():
                raise ModelMaterializationError("Managed cache entry is not a ROCmHub-owned directory.")
            if entry.exists():
                verified = self._read_verified_cache(entry, model_spec)
                if verified is None:
                    raise ModelMaterializationError("Existing managed cache entry failed integrity verification.")
                snapshot, weight_bytes, files = verified
                self._progress("CACHE_HIT", materialized_bytes=sum((snapshot / f).stat().st_size for f in files))
                return MaterializedModel(
                    local_path=str(snapshot), mode=MaterializationMode.FULL_WEIGHTS, files=files,
                    has_weights=True, license_name=getattr(model_spec, "license", None),
                    weights_size_bytes=weight_bytes, cached=True, cache_status="HIT_VERIFIED",
                    integrity="MANIFEST_AND_SELECTED_SHA256_VERIFIED",
                    materialized_bytes=sum((snapshot / f).stat().st_size for f in files),
                )

            current_cache_bytes = self._directory_bytes(root)
            current_staging_bytes = self._directory_bytes(self._cache_dir / "staging")
            if current_cache_bytes + current_staging_bytes + bounded_estimate > self._max_cache_bytes:
                raise InsufficientDiskSpaceError("Managed Agent model cache has reached its configured size limit.")
            root.mkdir(parents=True, exist_ok=True)
            staging_parent = self._cache_dir / "staging"
            if staging_parent.is_symlink():
                raise ModelMaterializationError("Managed staging root is not a ROCmHub-owned directory.")
            stage_root = staging_parent / self._cache_key(model_spec.model_id, model_spec.commit_sha)
            # Stage is ROCmHub-owned and deterministic, so HF can resume its partial blobs after interruption.
            stage_marker = stage_root / ".rocmhub-staging.json"
            if stage_root.is_symlink():
                raise ModelMaterializationError("Managed staging path is not a ROCmHub-owned directory.")
            if stage_root.exists():
                if stage_marker.is_symlink() or not stage_marker.is_file():
                    raise ModelMaterializationError("Existing staging directory is not marked as ROCmHub-owned.")
                try:
                    marker = json.loads(stage_marker.read_text(encoding="utf-8"))
                except (OSError, ValueError) as exc:
                    raise ModelMaterializationError("Existing staging marker is invalid.") from exc
                if marker != {"model_id": model_spec.model_id, "revision": model_spec.commit_sha}:
                    raise ModelMaterializationError("Existing staging directory belongs to a different model revision.")
            else:
                stage_root.mkdir(parents=True, exist_ok=False)
                stage_marker.write_text(
                    json.dumps({"model_id": model_spec.model_id, "revision": model_spec.commit_sha}), encoding="utf-8"
                )
            self.check_disk_space(stage_root, bounded_estimate + 512 * 1024**2)
            self._progress("CACHE_MISS", estimated_bytes=required_disk_bytes)
            self._check_cancelled()
            try:
                self._progress("DOWNLOADING_MODEL", estimated_bytes=required_disk_bytes)
                downloaded = snapshot_download(
                    repo_id=model_spec.model_id,
                    revision=model_spec.commit_sha,
                    cache_dir=stage_root / "hub",
                    token=self._token,
                    allow_patterns=["*.json", "*.model", "*.txt", "*.tiktoken", "*.safetensors", "*.bin", "*.pt", "*.h5", "*.pth", "*.msgpack"],
                )
            except Exception as exc:
                raise self._translate_error(exc, model_spec.model_id, model_spec.commit_sha) from exc
            self._check_cancelled()
            snapshot = Path(downloaded)
            if not (snapshot / "config.json").is_file():
                raise ModelMaterializationError("Downloaded model is missing its required configuration.")
            if not any((snapshot / name).is_file() for name in {"tokenizer.json", "tokenizer_config.json", "vocab.json", "tokenizer.model"}):
                raise ModelMaterializationError("Downloaded model is missing required tokenizer assets.")
            weight_bytes, has_weights = self._verify_weights(snapshot)
            if not has_weights:
                raise ModelMaterializationError("Downloaded snapshot does not contain real model weights.")
            inventory = self._manifest_files(snapshot, full=True)
            total_bytes = sum(
                size for item in inventory if isinstance((size := item.get("size_bytes")), int)
            )
            if total_bytes > self._max_cache_bytes:
                raise InsufficientDiskSpaceError("Downloaded model exceeds this Agent's managed cache limit.")
            self._progress("VERIFYING_CACHE", materialized_bytes=total_bytes)
            manifest = {
                "schema_version": 1,
                "model_id": model_spec.model_id,
                "revision": model_spec.commit_sha,
                "materialized_at": datetime.now(timezone.utc).isoformat(),
                "files": inventory,
                "weights_present": True,
                "weights_bytes": weight_bytes,
                "total_bytes": total_bytes,
                "integrity_evidence": "huggingface_hub_snapshot_download_plus_sha256_file_inventory",
                "cache_hit_integrity_policy": "all manifest files and sizes plus SHA-256 for config, tokenizer, and weight index; weights are size-checked",
                "state": "COMPLETE",
            }
            (stage_root / _CACHE_MANIFEST).write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
            self._check_cancelled()
            # Atomic directory rename publishes only a fully verified snapshot and manifest.
            os.replace(stage_root, entry)
            final_snapshot = entry / "hub" / f"models--{model_spec.model_id.replace('/', '--')}" / "snapshots" / model_spec.commit_sha
            self._progress("MODEL_MATERIALIZED", materialized_bytes=total_bytes)
            return MaterializedModel(
                local_path=str(final_snapshot), mode=MaterializationMode.FULL_WEIGHTS, files=[str(f["path"]) for f in inventory],
                has_weights=True, license_name=getattr(model_spec, "license", None),
                weights_size_bytes=weight_bytes, cached=False, cache_status="MISS_DOWNLOADED",
                integrity="FULL_SHA256_MANIFEST_VERIFIED", materialized_bytes=total_bytes,
            )

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

    def _verify_weights(self, download_path: Path) -> Tuple[int, bool]:
        """Verify presence and completeness of weight files and shards.

        Raises ModelMaterializationError if shard index specifies missing or empty shards,
        or if no valid weight files exist.
        """
        index_files = ["model.safetensors.index.json", "pytorch_model.bin.index.json"]
        found_index: Optional[Path] = None
        for idx_name in index_files:
            idx_path = download_path / idx_name
            if idx_path.exists():
                found_index = idx_path
                break

        if found_index is not None:
            try:
                with open(found_index, "r", encoding="utf-8") as f:
                    data = json.load(f)
                weight_map = data.get("weight_map", {})
                needed_shards = set(weight_map.values())
            except Exception as exc:
                raise ModelMaterializationError(
                    f"Failed to parse weight index '{found_index.name}': {exc}",
                    details={"index_path": str(found_index), "error": str(exc)},
                ) from exc

            if not needed_shards:
                raise ModelMaterializationError(
                    f"Weight index '{found_index.name}' contains empty weight_map.",
                    details={"index_path": str(found_index)},
                )

            total_size = 0
            missing_shards: List[str] = []
            empty_shards: List[str] = []
            for shard_name in needed_shards:
                shard_name = str(shard_name)
                shard_relative = Path(shard_name)
                if shard_relative.is_absolute() or ".." in shard_relative.parts:
                    raise ModelMaterializationError("Weight index contains an unsafe shard path.")
                shard_path = download_path / shard_name
                if not shard_path.exists():
                    missing_shards.append(shard_name)
                else:
                    sz = shard_path.stat().st_size
                    if sz == 0:
                        empty_shards.append(shard_name)
                    total_size += sz

            if missing_shards or empty_shards:
                err_parts = []
                if missing_shards:
                    err_parts.append(f"missing shards: {', '.join(sorted(missing_shards))}")
                if empty_shards:
                    err_parts.append(f"empty shards: {', '.join(sorted(empty_shards))}")
                raise ModelMaterializationError(
                    f"Model weights verification failed: {'; '.join(err_parts)}",
                    details={
                        "missing_shards": sorted(missing_shards),
                        "empty_shards": sorted(empty_shards),
                        "index_path": str(found_index),
                    },
                )
            return total_size, True

        # Standalone weight files
        total_size = 0
        found_weight_files = False
        for root, _, filenames in os.walk(download_path):
            for filename in filenames:
                full_path = Path(root) / filename
                if full_path.suffix.lower() in WEIGHT_FILE_EXTENSIONS:
                    sz = full_path.stat().st_size
                    if sz > 0:
                        found_weight_files = True
                        total_size += sz

        if not found_weight_files:
            raise ModelMaterializationError(
                f"No valid weight files found in '{download_path}'.",
                details={"target_path": str(download_path)},
            )

        return total_size, True

    def materialize(
        self,
        model_spec: ModelSpec,
        required_disk_bytes: int = 0,
        download_weights: bool = True,
        mode: Optional[MaterializationMode] = None,
    ) -> MaterializedModel:
        """Download or retrieve from local cache the model at the immutable commit SHA.

        Guarantees:
        - Checks available disk space before initiating download.
        - Enforces trust_remote_code=False.
        - Uses immutable commit SHA (model_spec.revision).
        - Handles gated/missing repo errors cleanly.
        - Verifies completeness of weight shards if mode is FULL_WEIGHTS.
        """
        if mode is not None:
            effective_mode = mode
            download_weights = (mode == MaterializationMode.FULL_WEIGHTS)
        else:
            effective_mode = (
                MaterializationMode.FULL_WEIGHTS if download_weights else MaterializationMode.METADATA_ONLY
            )

        if self._managed_cache and effective_mode == MaterializationMode.FULL_WEIGHTS:
            return self._materialize_managed(model_spec, required_disk_bytes)

        cache_target = self._determine_cache_target()

        if required_disk_bytes > 0:
            self.check_disk_space(cache_target, required_disk_bytes)

        # Check if snapshot already existed prior to call
        model_folder_name = f"models--{model_spec.model_id.replace('/', '--')}"
        expected_snapshot_dir = cache_target / model_folder_name / "snapshots" / model_spec.commit_sha
        was_already_cached = expected_snapshot_dir.exists()

        ignore_patterns = (
            None
            if download_weights
            else ["*.safetensors", "*.bin", "*.pt", "*.h5", "*.pth", "*.msgpack"]
        )

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
        for root, _, filenames in os.walk(download_path):
            for filename in filenames:
                full_path = Path(root) / filename
                rel_path = full_path.relative_to(download_path)
                files.append(str(rel_path))

        files.sort()

        if effective_mode == MaterializationMode.FULL_WEIGHTS:
            # 1. Config verification
            if not (download_path / "config.json").exists():
                raise ModelMaterializationError(
                    f"Model configuration 'config.json' missing in '{download_path}'.",
                    details={"model_id": model_spec.model_id, "path": str(download_path)},
                )

            # 2. Tokenizer verification
            tokenizer_files = {"tokenizer.json", "tokenizer_config.json", "vocab.json", "tokenizer.model"}
            if not any((download_path / tf).exists() for tf in tokenizer_files):
                raise ModelMaterializationError(
                    f"Model tokenizer files missing in '{download_path}'.",
                    details={"model_id": model_spec.model_id, "path": str(download_path)},
                )

            # 3. Weights verification
            weights_size_bytes, has_weights = self._verify_weights(download_path)
        else:
            weights_size_bytes = 0
            has_weights = False

        return MaterializedModel(
            local_path=str(download_path),
            mode=effective_mode,
            files=files,
            has_weights=has_weights,
            license_name=getattr(model_spec, "license", None),
            weights_size_bytes=weights_size_bytes,
            cached=was_already_cached,
        )
