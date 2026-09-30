"""Forge planning routes for ROCmHub API."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, cast

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

from rocmhub.core.errors import (
    AuthRequiredError,
    ModelNotFoundError,
    NetworkError,
    RemoteCodeRequiredError,
    RevisionNotFoundError,
    SecurityBoundaryError,
    UnsupportedModelArchitectureError,
)
from rocmhub.forge.planner import ForgePlanner
from rocmhub.models.huggingface import HuggingFaceModelSource
from rocmhub.models.inspector import ModelInspector
from rocmhub.server.privacy import sanitize_public_value
from rocmhub.server.security import validate_job_path

router = APIRouter(prefix="/api/v1/forge", tags=["Forge"])


class ForgePlanRequest(BaseModel):
    """Payload for requesting a deterministic Forge build plan."""

    model_id: str = Field(..., min_length=1, max_length=256, description="Hugging Face model ID")
    revision: Optional[str] = Field(default="main", description="Git revision/branch/commit SHA")
    precision: Optional[str] = Field(default="fp16", description="Target floating point precision")
    target_gpu: Optional[str] = Field(default=None, description="Target GPU device/gfx")
    recipe: Optional[str] = Field(default=None, description="Build recipe name")
    output_dir: Optional[str] = Field(default=None, description="Target output build directory")


@router.post("/plan")
async def create_forge_plan(request_body: ForgePlanRequest, req: Request) -> Dict[str, Any]:
    """Generate a deterministic Forge build plan without downloading model weights."""
    manager = req.app.state.job_manager
    allowed_roots = manager.config.allowed_workspaces

    out_path = Path("builds") / f"forge--{request_body.model_id.replace('/', '--')}--{request_body.precision or 'fp16'}"
    if request_body.output_dir:
        try:
            out_path = validate_job_path(request_body.output_dir, allowed_roots)
        except SecurityBoundaryError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Security error for output directory: {sanitize_public_value(str(exc))}",
            ) from exc

    source = HuggingFaceModelSource()
    inspector = ModelInspector(source)
    planner = ForgePlanner(model_inspector=inspector)

    try:
        plan = planner.create_plan(
            model_id=request_body.model_id,
            revision=request_body.revision or "main",
            precision=request_body.precision or "fp16",
            target_gpu=request_body.target_gpu,
            output_dir=out_path,
            recipe_id=request_body.recipe,
        )
        return cast(Dict[str, Any], sanitize_public_value(plan.model_dump(mode="json")))

    except ModelNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except RevisionNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except AuthRequiredError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc
    except UnsupportedModelArchitectureError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except (RemoteCodeRequiredError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except NetworkError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
