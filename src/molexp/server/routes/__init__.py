"""API Routes package.

This package contains all the modularized API routes organized by domain.
"""

from fastapi import APIRouter

from . import asset, execution, experiment, registry, project, run, workspace


def create_api_router() -> APIRouter:
    """Create and configure the main API router with all sub-routes.

    Returns:
        Configured APIRouter with all domain routes included
    """
    api_router = APIRouter()

    # Include all domain routers
    api_router.include_router(project.router)
    api_router.include_router(experiment.router)
    api_router.include_router(run.router)
    api_router.include_router(asset.router)
    api_router.include_router(workspace.router)
    api_router.include_router(registry.router)
    api_router.include_router(execution.router)

    return api_router


__all__ = [
    "create_api_router",
    "project",
    "experiment",
    "run",
    "asset",
    "workspace",
    "registry",
    "execution",
]
