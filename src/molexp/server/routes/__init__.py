"""API Routes package.

This package contains all the modularized API routes organized by domain.
"""

from fastapi import APIRouter, Depends, Request

from ..dependencies import assert_served_workspace, assert_workspace_writable
from . import (
    agent,
    agent_admin,
    agent_tasks,
    asset,
    catalog,
    execution,
    experiment,
    library,
    molq,
    preview,
    project,
    registry,
    run,
    targets,
    tensorboard,
    workflow,
    workspace,
    workspaces,
)


# Domain routers that address entities *inside* one workspace. They are
# mounted twice: flat (active/default workspace, back-compat) and again under
# the ``/workspaces/{ws}`` prefix (the aggregate surface). Process-global or
# active-workspace-management routers (agent*, molq, registry, targets,
# workspace, workspaces) are NOT namespaced.
def _workspace_scoped_modules() -> tuple:
    return (
        project,
        experiment,
        run,
        execution,
        asset,
        preview,
        catalog,
        library,
        workflow,
        tensorboard,
    )


def _bind_ws(ws: str, request: Request) -> str:
    """Router-level dependency for the ``/workspaces/{ws}`` mount.

    Declares the ``{ws}`` path param, 404s early when it names no served
    workspace, and 405s a mutating request against a remote workspace (before
    body validation). Resolved ahead of the endpoint body.
    """
    assert_served_workspace(ws)
    assert_workspace_writable(ws, request.method)
    return ws


def _create_workspace_scoped_router() -> APIRouter:
    """A mirror of the per-workspace domain routers, gated on a valid ``{ws}``."""
    scoped = APIRouter(dependencies=[Depends(_bind_ws)])
    for module in _workspace_scoped_modules():
        scoped.include_router(module.router)
    return scoped


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
    api_router.include_router(preview.router)
    api_router.include_router(catalog.router)
    api_router.include_router(library.router)
    api_router.include_router(workspace.router)
    api_router.include_router(workspaces.router)
    api_router.include_router(registry.router)
    api_router.include_router(execution.router)
    api_router.include_router(molq.router)
    api_router.include_router(targets.router)
    api_router.include_router(tensorboard.router)
    api_router.include_router(workflow.router)

    # Aggregate surface: the same domain routers, namespaced by workspace.
    api_router.include_router(_create_workspace_scoped_router(), prefix="/workspaces/{ws}")

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
    "library",
    "molq",
    "preview",
    "project",
    "registry",
    "run",
    "targets",
    "tensorboard",
    "workflow",
    "workspace",
    "workspaces",
]
