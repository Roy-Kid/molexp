"""Task plugin routes for MolExp API."""

from __future__ import annotations

from fastapi import APIRouter

from molexp.workflow.plugin import get_task_registry

from ..exceptions import TaskNotFoundError

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.get("")
def list_nodes():
    """List all available node types from plugins.

    Returns:
        Dictionary with all node definitions including metadata and config schemas
    """
    registry = get_task_registry()
    return registry.to_dict()


@router.get("/{node_id}")
def get_node(node_id: str):
    """Get details for a specific node type.

    Args:
        node_id: Task identifier (e.g., "io.write_file")

    Returns:
        Task definition with metadata and config schema
    """
    registry = get_task_registry()
    registration = registry.get(node_id)

    if not registration:
        raise TaskNotFoundError(node_id)

    return registration.to_dict()
