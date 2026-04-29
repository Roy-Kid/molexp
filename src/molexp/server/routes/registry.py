"""Plugin and task-type registry routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from molexp.plugins import discover_ui_plugins
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
    plugins = [UiPluginResponse.from_descriptor(plugin) for plugin in discover_ui_plugins()]
    return UiPluginListResponse(plugins=plugins, total=len(plugins))
