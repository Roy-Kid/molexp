"""API Routes package.

This package contains all the modularized API routes organized by domain.
"""

from fastapi import APIRouter

from . import assets, execution, experiments, nodes, projects, runs, workspace


def create_api_router() -> APIRouter:
    """Create and configure the main API router with all sub-routes.

    Returns:
        Configured APIRouter with all domain routes included
    """
    api_router = APIRouter()

    # Include all domain routers
    api_router.include_router(projects.router)
    api_router.include_router(experiments.router)
    api_router.include_router(runs.router)
    api_router.include_router(assets.router)
    api_router.include_router(workspace.router)
    api_router.include_router(nodes.router)
    api_router.include_router(execution.router)

    return api_router


__all__ = [
    "create_api_router",
    "projects",
    "experiments",
    "runs",
    "assets",
    "workspace",
    "nodes",
    "execution",
]
