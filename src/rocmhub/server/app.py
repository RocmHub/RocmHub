"""FastAPI application factory for ROCmHub."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from rocmhub.core.errors import ROCmHubError
from rocmhub.server.config import ServerConfig
from rocmhub.server.orchestrator.manager import JobManager
from rocmhub.server.routes import agents_router, forge_router, health_router, jobs_router, models_router
from rocmhub.server.security import RequestSizeLimitMiddleware


def create_app(config: Optional[ServerConfig] = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    cfg = config or ServerConfig()
    manager = JobManager(cfg)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Startup
        manager.start()
        app.state.job_manager = manager
        app.state.config = cfg
        try:
            yield
        finally:
            # Shutdown
            manager.stop()

    app = FastAPI(
        title="ROCmHub Backend API",
        version="0.1.0",
        description="Local backend API & Job Orchestration for preparing, optimizing, and running models on AMD GPUs.",
        lifespan=lifespan,
    )

    # Attach to app state immediately so test clients or early middleware can access
    app.state.job_manager = manager
    app.state.config = cfg

    # 1. Security: Request body size limit middleware
    app.add_middleware(RequestSizeLimitMiddleware, max_request_bytes=cfg.max_request_bytes)

    # 2. CORS middleware restricted to local development origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # 3. Global domain exception handler
    @app.exception_handler(ROCmHubError)
    async def rocmhub_error_handler(request: Request, exc: ROCmHubError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": exc.__class__.__name__,
                "detail": str(exc),
                "details": exc.details,
            },
        )

    # 4. Unhandled server exception handler (prevents trace leakage)
    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "InternalServerError",
                "detail": "An internal server error occurred while processing the request.",
            },
        )

    # 4. Include Routers
    app.include_router(health_router)
    app.include_router(models_router)
    app.include_router(forge_router)
    app.include_router(jobs_router)
    app.include_router(agents_router)

    return app
