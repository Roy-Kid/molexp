"""Plugin and task-type registry routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from molexp.plugins import discover_ui_plugin_dirs
from molexp.workflow.registry import default_registry

from ..schemas import (
    TaskTypeListResponse,
    TaskTypeResponse,
    UiPluginListResponse,
    UiPluginResponse,
)

router = APIRouter(tags=["registry"])


@router.get("/tasks", response_model=TaskTypeListResponse)
def list_task_types() -> TaskTypeListResponse:
    """Return every task-type slug the agent / UI can compose into IR."""
    items = [
        TaskTypeResponse(slug=slug, description=description)
        for slug, description in default_registry.items()
    ]
    return TaskTypeListResponse(task_types=items, total=len(items))


@router.get("/tasks/{slug:path}", response_model=TaskTypeResponse)
def get_task_type(slug: str) -> TaskTypeResponse:
    """Return one task type by slug, or 404 if not registered."""
    if not default_registry.has(slug):
        raise HTTPException(
            status_code=404,
            detail=f"Task type {slug!r} is not registered.",
        )
    return TaskTypeResponse(slug=slug, description=default_registry.describe(slug))


@router.get("/plugins", response_model=UiPluginListResponse, tags=["plugins"])
def list_plugins() -> UiPluginListResponse:
    """List entry-point–discovered UI bundles.

    Built-in plugins (``core``, ``metrics``, ``molq``, ``molvis``) are
    statically imported by the frontend and do **not** appear here. The
    response carries no UI semantics — those live in each bundle's own
    ``manifest.json``, fetched by the browser-side loader.
    """  # noqa: RUF002
    plugins = [
        UiPluginResponse(
            id=plugin_id,
            manifestUrl=f"/api/plugins/{plugin_id}/manifest.json",
            entryUrl=f"/api/plugins/{plugin_id}/index.js",
        )
        for plugin_id in discover_ui_plugin_dirs()
    ]
    return UiPluginListResponse(plugins=plugins, total=len(plugins))
