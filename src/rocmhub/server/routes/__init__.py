"""ROCmHub API routes package."""

from rocmhub.server.routes.agents import router as agents_router
from rocmhub.server.routes.forge import router as forge_router
from rocmhub.server.routes.health import router as health_router
from rocmhub.server.routes.jobs import router as jobs_router
from rocmhub.server.routes.models import router as models_router

__all__ = [
    "health_router",
    "agents_router",
    "models_router",
    "forge_router",
    "jobs_router",
]
