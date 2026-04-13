"""Task plugin routes for MolExp API.

Note: Task registry is being re-implemented as part of Phase 3.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from molexp.plugins import discover_ui_plugins

from ..schemas import UiPluginListResponse, UiPluginResponse

router = APIRouter(tags=["registry"])


@router.get("/tasks")
def list_nodes() -> JSONResponse:
    return JSONResponse(
        status_code=501,
        content={
            "error": "not_implemented",
            "message": "Task registry is being re-implemented in Phase 3.",
            "tasks": [],
        },
    )


@router.get("/tasks/{node_id}")
def get_node(node_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=501,
        content={
            "error": "not_implemented",
            "message": f"Task registry is being re-implemented in Phase 3. Task '{node_id}' not found.",
        },
    )


@router.get("/plugins", response_model=UiPluginListResponse, tags=["plugins"])
def list_plugins() -> UiPluginListResponse:
    plugins = [UiPluginResponse.from_descriptor(plugin) for plugin in discover_ui_plugins()]
    return UiPluginListResponse(plugins=plugins, total=len(plugins))
