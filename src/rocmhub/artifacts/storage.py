"""Local artifact store handling atomic deployment, retrieval, and integrity verification."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from rocmhub.artifacts.integrity import (
    compute_file_sha256,
)
from rocmhub.core.errors import (
    ArtifactConflictError,
    ArtifactPackagingError,
)
from rocmhub.core.types import (
    ArtifactManifest,
    ArtifactVerificationResult,
)


class LocalArtifactStore:
    """Manages on-disk storage, atomic directory publication, and verification of artifact bundles."""

    def __init__(self, root_dir: Path | str = "artifacts") -> None:
        self.root_dir = Path(root_dir).resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def get_artifact_path(self, artifact_id: str) -> Path:
        """Get the full filesystem path to an artifact directory."""
        return self.root_dir / artifact_id

    def artifact_exists(self, artifact_id: str) -> bool:
        """Check whether an artifact bundle exists in the store."""
        target = self.get_artifact_path(artifact_id)
        return target.is_dir() and (target / "manifest.json").is_file()

    def list_artifacts(self) -> List[str]:
        """List all valid artifact identifiers present in the store."""
        if not self.root_dir.is_dir():
            return []
        artifacts: List[str] = []
        for entry in sorted(self.root_dir.iterdir()):
            if entry.is_dir() and not entry.name.startswith(".") and (entry / "manifest.json").is_file():
                artifacts.append(entry.name)
        return artifacts

    def commit_staged_bundle(
        self,
        staged_dir: Path,
        artifact_id: str,
    ) -> Path:
        """Atomically commit a staged artifact bundle into the store.

        Prevents silent overwrites:
        - If an artifact with artifact_id already exists:
          - If its content matches staged checksums exactly, commit is idempotent.
          - If content differs, raises ArtifactConflictError.
        - Directory move is atomic (POSIX rename on same filesystem).

        Args:
            staged_dir: Path to temporary directory holding complete artifact files.
            artifact_id: Canonical artifact identifier.

        Returns:
            Final filesystem path to the committed artifact directory.

        Raises:
            ArtifactConflictError: If artifact already exists with differing content.
            ArtifactPackagingError: If atomic rename or commit fails.
        """
        target_path = self.get_artifact_path(artifact_id)

        # Check existing target
        if target_path.exists():
            staged_checksums_file = staged_dir / "checksums.json"
            target_checksums_file = target_path / "checksums.json"

            if staged_checksums_file.is_file() and target_checksums_file.is_file():
                if staged_checksums_file.read_bytes() == target_checksums_file.read_bytes():
                    # Idempotent: identical artifact already present; clean up staging dir
                    shutil.rmtree(staged_dir, ignore_errors=True)
                    return target_path

            raise ArtifactConflictError(
                f"Artifact directory '{target_path}' already exists and contains differing data. "
                "Silent overwrites are prohibited.",
                details={"artifact_id": artifact_id, "path": str(target_path)},
            )

        try:
            # Atomic rename into final location
            os.replace(staged_dir, target_path)
            return target_path
        except Exception as exc:
            raise ArtifactPackagingError(
                f"Failed to atomically commit artifact bundle to '{target_path}': {exc}",
                details={"artifact_id": artifact_id, "error": str(exc)},
            ) from exc

    def verify_artifact(self, artifact_path: Path | str) -> ArtifactVerificationResult:
        """Verify the integrity, inventory, and cryptographic checksums of an artifact bundle.

        Checks:
        1. Directory exists.
        2. manifest.json and checksums.json are present and parse cleanly.
        3. manifest.json matches declared checksums.json['files']['manifest.json'].
        4. Every declared file is present on disk and matches both SHA-256 and size in bytes.
        5. No unexpected or undeclared files are present on disk.

        Args:
            artifact_path: Path to the artifact directory.

        Returns:
            ArtifactVerificationResult with full diagnostic findings.
        """
        target = Path(artifact_path).resolve()
        errors: List[str] = []
        missing_files: List[str] = []
        modified_files: List[str] = []
        unexpected_files: List[str] = []
        manifest_valid = False
        checksums_valid = False

        if not target.is_dir():
            return ArtifactVerificationResult(
                valid=False,
                artifact_id=target.name,
                manifest_valid=False,
                checksums_valid=False,
                missing_files=[],
                modified_files=[],
                unexpected_files=[],
                errors=[f"Artifact directory not found: '{target}'"],
            )

        manifest_path = target / "manifest.json"
        checksums_path = target / "checksums.json"

        # 1. Verify manifest.json exists and parses
        manifest_obj: Optional[ArtifactManifest] = None
        if not manifest_path.is_file():
            missing_files.append("manifest.json")
            errors.append("Required manifest.json is missing.")
        else:
            try:
                manifest_bytes = manifest_path.read_bytes()
                manifest_obj = ArtifactManifest.model_validate_json(manifest_bytes)
                manifest_valid = True
            except Exception as exc:
                errors.append(f"manifest.json failed schema validation: {exc}")

        # 2. Verify checksums.json exists and parses
        checksums_data: Optional[Dict[str, Any]] = None
        if not checksums_path.is_file():
            missing_files.append("checksums.json")
            errors.append("Required checksums.json is missing.")
        else:
            try:
                checksums_data = json.loads(checksums_path.read_text(encoding="utf-8"))
            except Exception as exc:
                errors.append(f"checksums.json is corrupted or invalid JSON: {exc}")

        artifact_id = (
            manifest_obj.artifact_id
            if manifest_obj
            else (checksums_data.get("artifact_id", target.name) if checksums_data else target.name)
        )

        # 3. Verify manifest checksum against checksums.json
        if manifest_valid and checksums_data and "files" in checksums_data:
            declared_files = checksums_data["files"]
            expected_manifest_sha = declared_files.get("manifest.json")
            if not expected_manifest_sha:
                errors.append("checksums.json does not contain entry for 'manifest.json'.")
            else:
                actual_manifest_sha = compute_file_sha256(manifest_path)
                if actual_manifest_sha != expected_manifest_sha:
                    modified_files.append("manifest.json")
                    errors.append(
                        f"manifest.json SHA-256 mismatch (expected {expected_manifest_sha}, got {actual_manifest_sha})."
                    )

        # 4. Check all files declared in checksums.json
        declared_file_names: set[str] = set()
        if checksums_data and "files" in checksums_data and isinstance(checksums_data["files"], dict):
            checksums_valid = True
            for rel_path, expected_sha in checksums_data["files"].items():
                if rel_path == "manifest.json":
                    declared_file_names.add(rel_path)
                    continue

                declared_file_names.add(rel_path)
                file_path = target / rel_path

                if not file_path.is_file():
                    missing_files.append(rel_path)
                    checksums_valid = False
                    continue

                actual_sha = compute_file_sha256(file_path)
                if actual_sha != expected_sha:
                    modified_files.append(rel_path)
                    checksums_valid = False

        # 5. Check manifest inventory consistency
        if manifest_obj:
            for rel_path, entry in manifest_obj.files.items():
                declared_file_names.add(rel_path)
                file_path = target / rel_path
                if not file_path.is_file() and rel_path not in missing_files:
                    missing_files.append(rel_path)
                elif file_path.is_file():
                    actual_size = file_path.stat().st_size
                    if actual_size != entry.size_bytes and rel_path not in modified_files:
                        modified_files.append(rel_path)
                        errors.append(
                            f"File size mismatch for '{rel_path}': declared {entry.size_bytes}B, actual {actual_size}B."
                        )

        # 6. Check for unexpected files in the directory
        # Exclude checksums.json from declared check as it is the detached checksum file
        valid_known_names = declared_file_names.union({"checksums.json"})
        for item in target.iterdir():
            if item.name.startswith("."):
                continue
            rel_name = item.name
            if rel_name not in valid_known_names:
                unexpected_files.append(rel_name)

        if missing_files:
            errors.append(f"Missing declared artifact files: {missing_files}")
        if modified_files:
            errors.append(f"Modified or corrupted artifact files: {modified_files}")
        if unexpected_files:
            errors.append(f"Unexpected undeclared files found in artifact directory: {unexpected_files}")

        is_valid = (
            manifest_valid
            and checksums_valid
            and len(missing_files) == 0
            and len(modified_files) == 0
            and len(unexpected_files) == 0
            and len(errors) == 0
        )

        return ArtifactVerificationResult(
            valid=is_valid,
            artifact_id=artifact_id,
            manifest_valid=manifest_valid,
            checksums_valid=checksums_valid and len(modified_files) == 0,
            missing_files=sorted(missing_files),
            modified_files=sorted(modified_files),
            unexpected_files=sorted(unexpected_files),
            errors=errors,
        )
