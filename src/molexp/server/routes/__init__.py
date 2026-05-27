"""API Routes package.

This package contains all the modularized API routes organized by domain.
"""

from fastapi import APIRouter

from . import (
    agent,
    agent_admin,
    agent_tasks,
    asset,
    catalog,
    execution,
    experiment,
    molq,
    project,
    registry,
    reviews,
    run,
    targets,
    tensorboard,
    workspace,
)


def create_api_router() -> APIRouter:
    """Create and configure the main API router with all sub-routes.

    Returns:
        Configured APIRouter with all domain routes included
    """
    api_router = APIRouter()

    api_router.include_router(agent.router)
    api_router.include_router(agent_admin.router)
    api_router.include_router(agent_tasks.router)
    api_router.include_router(project.router)
    api_router.include_router(experiment.router)
    api_router.include_router(run.router)
    api_router.include_router(asset.router)
    api_router.include_router(catalog.router)
    api_router.include_router(workspace.router)
    api_router.include_router(registry.router)
    api_router.include_router(reviews.router)
    api_router.include_router(execution.router)
    api_router.include_router(molq.router)
    api_router.include_router(targets.router)
    api_router.include_router(tensorboard.router)

    return api_router


__all__ = [
    "agent",
    "agent_admin",
    "agent_tasks",
    "asset",
    "catalog",
    "create_api_router",
    "execution",
    "experiment",
    "molq",
    "project",
    "registry",
    "reviews",
    "run",
    "targets",
    "tensorboard",
    "workspace",
]
