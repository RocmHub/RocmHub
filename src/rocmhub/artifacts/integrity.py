"""Canonical serialization, hashing, and security scanning utilities for artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Set

from rocmhub.core.errors import SecretDetectedError

# Whitelist of legitimate dictionary keys that contain the substring 'token' or 'key'
SAFE_KEYS: Set[str] = {
    "generated_tokens",
    "input_tokens",
    "max_new_tokens",
    "prompt_tokens",
    "token_count",
    "tokens",
    "tokens_per_sec",
    "token_timestamps",
    "total_tokens",
    "key",
    "gfx_target",
    "device_name",
    "sort_keys",
}

# Forbidden sensitive key patterns (checked against lower-case stripped key names)
FORBIDDEN_EXACT_KEYS: Set[str] = {
    "token",
    "secret",
    "password",
    "passwd",
    "pwd",
    "api_key",
    "apikey",
    "access_key",
    "access_key_id",
    "secret_access_key",
    "private_key",
    "private_key_id",
    "ssh_key",
    "hf_token",
    "huggingface_token",
    "auth_token",
    "access_token",
    "bearer_token",
    "github_token",
    "gh_token",
    "aws_secret_access_key",
    "aws_access_key_id",
}

# Forbidden regex patterns for keys
FORBIDDEN_KEY_REGEX = re.compile(
    r"^.*(password|passwd|secret|api_key|apikey|private_key|access_key).*$",
    re.IGNORECASE,
)

# Forbidden prefix signatures in string values (known credential formats)
SECRET_VALUE_PREFIXES = (
    "hf_",  # Hugging Face User Access Token
    "ghp_",  # GitHub Personal Access Token
    "gho_",  # GitHub OAuth Token
    "AKIA",  # AWS Access Key ID
    "ASIA",  # AWS Temporary Access Key ID
    "-----BEGIN RSA PRIVATE KEY-----",
    "-----BEGIN PRIVATE KEY-----",
    "-----BEGIN OPENSSH PRIVATE KEY-----",
)


def scan_for_secrets(data: Any, path: str = "") -> None:
    """Recursively scan data structures to prevent credentials and secrets from entering artifacts.

    Fails closed by raising SecretDetectedError if any sensitive key or credential prefix is detected.
    Explicitly whitelists inference metrics like 'generated_tokens', 'input_tokens', 'max_new_tokens'.

    Args:
        data: Arbitrary object, dictionary, list, or primitive.
        path: Path string for contextual error reporting.

    Raises:
        SecretDetectedError: If a forbidden key or credential pattern is detected.
    """
    if isinstance(data, dict):
        for key, val in data.items():
            key_str = str(key).strip().lower()
            current_path = f"{path}.{key}" if path else str(key)

            # 1. Check if key is explicitly whitelisted as safe
            if key_str in SAFE_KEYS:
                scan_for_secrets(val, current_path)
                continue

            # 2. Check exact forbidden keys
            if key_str in FORBIDDEN_EXACT_KEYS:
                raise SecretDetectedError(
                    f"Prohibited secret key detected at '{current_path}': '{key}'. "
                    "Artifact build failed closed to prevent credential leakage.",
                    details={"path": current_path, "key": str(key)},
                )

            # 3. Check keys ending in _token that are not safe
            if key_str.endswith("_token") or key_str.endswith("token"):
                raise SecretDetectedError(
                    f"Prohibited token key detected at '{current_path}': '{key}'. "
                    "Artifact build failed closed to prevent credential leakage.",
                    details={"path": current_path, "key": str(key)},
                )

            # 4. Check regex pattern for password / secret / key
            if FORBIDDEN_KEY_REGEX.match(key_str):
                raise SecretDetectedError(
                    f"Prohibited credential key pattern detected at '{current_path}': '{key}'. "
                    "Artifact build failed closed to prevent credential leakage.",
                    details={"path": current_path, "key": str(key)},
                )

            scan_for_secrets(val, current_path)

    elif isinstance(data, (list, tuple, set)):
        for idx, item in enumerate(data):
            current_path = f"{path}[{idx}]"
            scan_for_secrets(item, current_path)

    elif isinstance(data, str):
        trimmed = data.strip()
        for prefix in SECRET_VALUE_PREFIXES:
            if trimmed.startswith(prefix) and len(trimmed) > len(prefix) + 4:
                raise SecretDetectedError(
                    f"Prohibited secret credential pattern detected in value at '{path}'. "
                    "Artifact build failed closed to prevent credential leakage.",
                    details={"path": path},
                )


def canonical_json_dumps(data: Any) -> str:
    """Serialize data into a deterministic, canonical JSON string.

    Guarantees:
    - Deterministic key ordering (sort_keys=True)
    - Compact stable separators (",", ":") without arbitrary whitespace
    - Strict rejection of NaN and Infinity (allow_nan=False)
    - UTF-8 compatible unicode representation (ensure_ascii=False)
    - Pydantic models automatically serialized via model_dump(mode='json')

    Args:
        data: Primitive, dictionary, list, or Pydantic model.

    Returns:
        Canonical JSON string.
    """
    if hasattr(data, "model_dump"):
        raw_data = data.model_dump(mode="json")
    else:
        raw_data = data

    return json.dumps(
        raw_data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_json_bytes(data: Any) -> bytes:
    """Serialize data into deterministic canonical UTF-8 bytes."""
    return canonical_json_dumps(data).encode("utf-8")


def compute_sha256(data: bytes | str) -> str:
    """Compute the SHA-256 hex digest of bytes or a string."""
    if isinstance(data, str):
        data_bytes = data.encode("utf-8")
    else:
        data_bytes = data
    return hashlib.sha256(data_bytes).hexdigest()


def compute_file_sha256(path: Path | str) -> str:
    """Compute the SHA-256 hex digest of an on-disk file in chunks."""
    target_path = Path(path)
    hasher = hashlib.sha256()
    with target_path.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()
