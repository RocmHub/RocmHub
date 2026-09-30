"""Redaction helpers for durable job evidence exposed by the public API."""

from __future__ import annotations

import re
from typing import Any

_PRIVATE_FIELDS = {
    "agent_id",
    "agent_name",
    "executor_id",
    "hostname",
    "host_name",
    "machine_name",
    "user",
    "user_name",
    "username",
}
_LOCAL_PATH_FIELDS = {
    "artifact_root",
    "build_dir",
    "cache_dir",
    "cache_path",
    "local_path",
    "output_dir",
    "output_path",
    "weights_path",
    "workspace",
}
_ABSOLUTE_LOCAL_PATH = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    r"/(?:Users|home|root|private|tmp|var|Volumes|mnt|opt|etc|srv|app|data|workspace|agent-workspace|"
    r"System|Library|Applications|Network|run|dev|proc|work)"
    r"(?:/[^\s\"'<>;,]*)?"
    r"|[A-Za-z]:\\(?:Users|home|Windows|Temp)(?:\\[^\s\"'<>;,]*)?"
    r")",
    re.IGNORECASE,
)


def sanitize_public_value(value: Any, *, preserve_null_path_fields: bool = False) -> Any:
    """Remove machine identity and local filesystem locations recursively."""
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            safe_key = _ABSOLUTE_LOCAL_PATH.sub("[local path withheld]", str(key))
            normalized_key = safe_key.lower()
            is_path_field = normalized_key in _LOCAL_PATH_FIELDS or (
                normalized_key != "download_path"
                and normalized_key.endswith(("_path", "_dir", "_directory"))
            )
            if (
                normalized_key in _PRIVATE_FIELDS
                or (is_path_field and not (preserve_null_path_fields and item is None))
            ):
                continue
            sanitized[safe_key] = sanitize_public_value(item, preserve_null_path_fields=preserve_null_path_fields)
        return sanitized
    if isinstance(value, list):
        return [sanitize_public_value(item, preserve_null_path_fields=preserve_null_path_fields) for item in value]
    if isinstance(value, tuple):
        return [sanitize_public_value(item, preserve_null_path_fields=preserve_null_path_fields) for item in value]
    if isinstance(value, str):
        return _ABSOLUTE_LOCAL_PATH.sub("[local path withheld]", value)
    return value
