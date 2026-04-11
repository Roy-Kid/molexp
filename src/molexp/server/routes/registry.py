"""Task plugin routes for MolExp API.

Note: Task registry is being re-implemented as part of Phase 3.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.get("")
def list_nodes() -> JSONResponse:
    return JSONResponse(
        status_code=501,
        content={
            "error": "not_implemented",
            "message": "Task registry is being re-implemented in Phase 3.",
            "tasks": [],
        },
    )


@router.get("/{node_id}")
def get_node(node_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=501,
        content={
            "error": "not_implemented",
            "message": f"Task registry is being re-implemented in Phase 3. Task '{node_id}' not found.",
        },
    )
