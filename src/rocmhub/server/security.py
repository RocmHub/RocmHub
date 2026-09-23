"""Security boundaries and middleware for ROCmHub API."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, List

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from rocmhub.core.errors import SecurityBoundaryError

SECRET_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key\s*[:=]\s*['\"]?)([\w\-]{8,})(['\"]?)"),
    re.compile(r"(?i)(bearer\s+)([a-zA-Z0-9_\-\.]{10,})"),
    re.compile(r"(?i)(hf_[a-zA-Z0-9]{20,})"),
    re.compile(r"(?i)(token\s*[:=]\s*['\"]?)([\w\-]{8,})(['\"]?)"),
    re.compile(r"(?i)(secret\s*[:=]\s*['\"]?)([\w\-]{8,})(['\"]?)"),
]

FORBIDDEN_ROOTS = [
    Path("/etc"),
    Path("/bin"),
    Path("/sbin"),
    Path("/usr"),
    Path("/System"),
    Path("/Library"),
    Path("/var/root"),
    Path("/private/etc"),
    Path("/private/var/root"),
]


def redact_secrets(text: str) -> str:
    """Scrub sensitive credentials, tokens, and keys from text."""
    redacted = text
    # HF tokens
    redacted = re.sub(r"hf_[a-zA-Z0-9]{20,}", "***REDACTED***", redacted)
    # Bearer tokens
    redacted = re.sub(r"(?i)(bearer\s+)[a-zA-Z0-9_\-\.]{10,}", r"\g<1>***REDACTED***", redacted)
    # API key / token / secret key-value assignments
    redacted = re.sub(r"(?i)(api[_-]?key\s*[:=]\s*['\"]?)[\w\-]{8,}(['\"]?)", r"\g<1>***REDACTED***\g<2>", redacted)
    redacted = re.sub(r"(?i)(token\s*[:=]\s*['\"]?)[\w\-]{8,}(['\"]?)", r"\g<1>***REDACTED***\g<2>", redacted)
    redacted = re.sub(r"(?i)(secret\s*[:=]\s*['\"]?)[\w\-]{8,}(['\"]?)", r"\g<1>***REDACTED***\g<2>", redacted)
    return redacted


def sanitize_payload(obj: Any) -> Any:
    """Recursively scrub secrets from dictionaries, lists, and strings."""
    if isinstance(obj, str):
        return redact_secrets(obj)
    if isinstance(obj, dict):
        sanitized = {}
        for k, v in obj.items():
            if any(s in str(k).lower() for s in ("token", "secret", "password", "api_key", "apikey")):
                if isinstance(v, str):
                    sanitized[k] = "***REDACTED***"
                else:
                    sanitized[k] = sanitize_payload(v)
            else:
                sanitized[k] = sanitize_payload(v)
        return sanitized
    if isinstance(obj, list):
        return [sanitize_payload(v) for v in obj]
    return obj


def validate_job_path(path: str | Path, allowed_roots: List[Path]) -> Path:
    """Validate that requested job directory is inside an allowed root and not a system directory."""
    resolved = Path(path).resolve()

    # Disallow exact root or system directories
    if resolved == Path("/") or resolved == Path.home():
        raise SecurityBoundaryError(
            f"Directory '{resolved}' cannot be used as an output directory (root/home forbidden).",
            details={"path": str(resolved)},
        )

    for forbidden in FORBIDDEN_ROOTS:
        try:
            resolved.relative_to(forbidden.resolve())
            raise SecurityBoundaryError(
                f"Access to system directory '{forbidden}' is strictly forbidden.",
                details={"path": str(resolved), "forbidden_root": str(forbidden)},
            )
        except ValueError:
            pass

    # Ensure path is strictly within at least one allowed root (must be a subdirectory, not root itself)
    matched = False
    for allowed in allowed_roots:
        allowed_resolved = allowed.resolve()
        try:
            rel = resolved.relative_to(allowed_resolved)
            if rel == Path("."):
                raise SecurityBoundaryError(
                    f"Directory '{resolved}' matches an allowed workspace root exactly. An output directory must be a subdirectory.",
                    details={"path": str(resolved), "allowed_root": str(allowed_resolved)},
                )
            matched = True
            break
        except ValueError:
            continue

    if not matched:
        raise SecurityBoundaryError(
            f"Path '{resolved}' is outside allowed workspaces: {[str(p) for p in allowed_roots]}",
            details={"path": str(resolved)},
        )

    return resolved


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Middleware enforcing a maximum request body size limit."""

    def __init__(self, app: Any, max_request_bytes: int = 1_048_576) -> None:
        super().__init__(app)
        self.max_request_bytes = max_request_bytes

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                length = int(content_length)
                if length > self.max_request_bytes:
                    return Response(
                        content=f'{{"detail":"Request payload exceeds maximum allowed size of {self.max_request_bytes} bytes."}}',
                        status_code=413,
                        media_type="application/json",
                    )
            except ValueError:
                pass

        return await call_next(request)
