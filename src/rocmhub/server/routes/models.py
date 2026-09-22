"""Model inspection and resolution routes for ROCmHub API."""

from __future__ import annotations

import sys
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Query, status

from rocmhub.capabilities.evaluator import CapabilityEvaluator
from rocmhub.core.errors import (
    AuthRequiredError,
    InvalidModelMetadataError,
    ModelNotFoundError,
    NetworkError,
    RemoteCodeRequiredError,
    RevisionNotFoundError,
)
from rocmhub.hardware.detector import SystemObserver
from rocmhub.models.huggingface import HuggingFaceModelSource
from rocmhub.models.inspector import ModelInspector

router = APIRouter(prefix="/api/v1/models", tags=["Models"])


@router.get("/{model_id:path}")
async def get_model_info(
    model_id: str,
    revision: str = Query(default="main", description="Branch, tag, or 40-char commit SHA"),
) -> Dict[str, Any]:
    """Inspect remote Hugging Face model metadata, resolve immutable commit SHA, and check architecture."""
    source = HuggingFaceModelSource()
    inspector = ModelInspector(source)

    try:
        model_spec = inspector.inspect(model_id=model_id, revision=revision)
        repo_meta = source.get_repository_metadata(model_id=model_id, revision=revision)

        observer = SystemObserver()
        detection_report = observer.observe()
        gpus = detection_report.gpus

        evaluator = CapabilityEvaluator()
        cap_report = evaluator.evaluate(model_spec, detection_report)

        rocm_gpus = [g for g in gpus if g.gpu_vendor and g.gpu_vendor.lower() == "amd"]
        rocm_available = len(rocm_gpus) > 0 and sys.platform != "darwin"

        license_name = None
        if hasattr(repo_meta, "card_data") and isinstance(repo_meta.card_data, dict):
            license_name = repo_meta.card_data.get("license")
        if not license_name:
            for tag in repo_meta.tags:
                if tag.startswith("license:"):
                    license_name = tag.split("license:", 1)[1]
                    break

        return {
            "model_id": model_spec.model_id,
            "requested_revision": model_spec.requested_revision,
            "commit_sha": model_spec.commit_sha,
            "architecture": model_spec.architecture,
            "parameter_count": model_spec.parameter_count,
            "context_length": model_spec.context_length,
            "weights_format": model_spec.weights_format,
            "license": license_name,
            "pipeline_tag": repo_meta.pipeline_tag,
            "tags": repo_meta.tags,
            "files_count": len(repo_meta.files),
            "safetensors_metadata": repo_meta.safetensors_metadata,
            "rocm_available": rocm_available,
            "host_gpus_detected": len(rocm_gpus),
            "host_gpu_summary": [
                {
                    "device_name": g.device_name,
                    "gfx_target": g.gfx_target,
                    "vram_total_mb": g.vram_total_mb,
                    "gpu_present": g.gpu_present,
                }
                for g in rocm_gpus
            ],
            "compatibility": {
                "verdict": cap_report.verdict.value,
                "reasons": [
                    {
                        "code": r.code,
                        "message": r.message,
                        "severity": r.severity.value,
                    }
                    for r in cap_report.reasons
                ],
                "warnings": cap_report.warnings,
            },
        }

    except ModelNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_id}' was not found on Hugging Face Hub.",
        ) from exc
    except RevisionNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Revision '{revision}' for model '{model_id}' was not found.",
        ) from exc
    except AuthRequiredError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Authentication required to access model '{model_id}'.",
        ) from exc
    except RemoteCodeRequiredError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Model requires custom remote code execution which is strictly disabled.",
        ) from exc
    except (InvalidModelMetadataError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid model metadata: {exc}",
        ) from exc
    except NetworkError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to communicate with Hugging Face Hub: {exc}",
        ) from exc
