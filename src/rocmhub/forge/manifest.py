"""BuildManifest data model and serialization utilities."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from rocmhub.core.errors import SecretDetectedError
from rocmhub.forge.base import BuildStatus, BuildStepRecord

CURRENT_FORGE_SCHEMA_VERSION = "1.0.0"

_SECRET_PATTERNS = [
    re.compile(r"hf_[A-Za-z0-9]{34,}", re.IGNORECASE),
    re.compile(r"(?:api[_-]?key|auth[_-]?token|bearer[_-]?token)[\s:=]+(['\"]?)([\w\-]{16,})\1", re.IGNORECASE),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_secrets_in_obj(obj: Any) -> Any:
    """Recursively scrub known sensitive tokens and keys from dictionaries/lists/strings."""
    if isinstance(obj, str):
        result = obj
        for pat in _SECRET_PATTERNS:
            result = pat.sub("[REDACTED_SECRET]", result)
        return result
    if isinstance(obj, dict):
        sanitized = {}
        for k, v in obj.items():
            if k == "secret_scan_clean":
                sanitized[k] = v
            elif any(sub in k.lower() for sub in ("token", "secret", "password", "auth", "credential", "api_key")):
                sanitized[k] = "[REDACTED]"
            else:
                sanitized[k] = sanitize_secrets_in_obj(v)
        return sanitized
    if isinstance(obj, list):
        return [sanitize_secrets_in_obj(item) for item in obj]
    return obj


def assert_no_secrets(content: str) -> None:
    """Raise SecretDetectedError if content contains any credential or private key."""
    for pat in _SECRET_PATTERNS:
        if pat.search(content):
            raise SecretDetectedError("Sensitive secret or token detected in manifest output payload.")


class BuildManifest(BaseModel):
    """Manifest recording the complete and reproducible outcome of a Forge build."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(default=CURRENT_FORGE_SCHEMA_VERSION, description="Forge schema version.")
    build_id: str = Field(..., description="Unique build identifier.")
    plan_id: str = Field(..., description="Referenced plan ID.")
    model_id: str = Field(..., description="Model identifier.")
    revision: str = Field(..., min_length=40, max_length=40, description="Immutable 40-character commit SHA.")
    target_gpu: Optional[str] = Field(default=None, description="Target AMD GPU architecture.")
    precision: str = Field(..., description="Target floating-point precision.")
    recipe_id: str = Field(..., description="Build recipe identifier.")
    recipe_version: str = Field(..., description="Build recipe version.")
    runtime: str = Field(..., description="Target runtime environment.")
    status: BuildStatus = Field(..., description="Overall build lifecycle status.")
    build_dir: str = Field(..., description="Local build output directory.")
    weights_path: str = Field(..., description="Local path to materialized model weights.")
    steps: List[BuildStepRecord] = Field(default_factory=list, description="Execution records of all build steps.")
    created_at: str = Field(default_factory=_utc_now_iso, description="UTC build start timestamp.")
    completed_at: Optional[str] = Field(default=None, description="UTC build completion timestamp.")
    amd_validated: bool = Field(default=False, description="Whether execution was verified on actual AMD ROCm hardware.")
    secret_scan_clean: bool = Field(default=True, description="Whether secret scanning verified no leaked credentials.")
    artifacts: Dict[str, str] = Field(
        default_factory=dict,
        description="Map of relative artifact filenames to their SHA-256 digests.",
    )

    def to_json(self, indent: int = 2) -> str:
        """Serialize manifest to canonical formatted JSON."""
        data = self.model_dump(mode="json")
        sanitized = sanitize_secrets_in_obj(data)
        serialized = json.dumps(sanitized, indent=indent, sort_keys=True)
        assert_no_secrets(serialized)
        return serialized


def write_manifest(manifest: BuildManifest, target_path: Path) -> None:
    """Safely write build manifest to target JSON file."""
    content = manifest.to_json(indent=2)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        f.write(content)
        f.write("\n")
    temp_path.replace(target_path)
