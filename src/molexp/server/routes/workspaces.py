"""``GET /api/workspaces`` — the set of workspaces this server is hosting.

``molexp serve`` can be pointed at one or more workspaces (local or remote).
This route exposes that set so the UI can list them and switch between them via
``POST /api/workspace/open`` (singular — active-workspace operations live in
``routes/workspace.py``). With a single served workspace this returns one row,
matching the unchanged single-workspace behaviour.
"""

from fastapi import APIRouter
from pydantic import BaseModel, Field

from molexp.server.dependencies import get_served_workspaces, get_workspace_by_key
from molexp.server.exceptions import RemoteWorkspaceUnreachableError

router = APIRouter(prefix="/workspaces", tags=["workspaces"])


class ServedWorkspaceResponse(BaseModel):
    """One workspace the server is hosting."""

    key: str = Field(..., description="Stable switch handle, unique per server process")
    label: str = Field(..., description="Human-facing label (path or user@host:/path)")
    isRemote: bool = Field(..., description="True for an SSH-backed remote workspace")
    path: str | None = Field(default=None, description="Absolute local root, null when remote")
    unreachable: bool = Field(
        default=False,
        description="True when a remote workspace's transport could not be reached",
    )


def _is_unreachable(key: str) -> bool:
    """Probe a remote workspace; True when its transport fails.

    Local workspaces are always reachable. Reachable remotes are cached by
    :func:`get_workspace_by_key`, so a successful probe is paid once; a failed
    probe is retried on each list call (no negative cache in v1).
    """
    try:
        get_workspace_by_key(key)
        return False
    except RemoteWorkspaceUnreachableError:
        return True


@router.get("", response_model=list[ServedWorkspaceResponse])
def list_workspaces() -> list[ServedWorkspaceResponse]:
    """List the workspaces ``molexp serve`` was started with.

    A remote workspace whose transport is currently unreachable is still
    listed, flagged ``unreachable`` so the UI can degrade gracefully rather
    than failing the whole list.
    """
    return [
        ServedWorkspaceResponse(
            key=w.key,
            label=w.label,
            isRemote=w.is_remote,
            path=w.path,
            unreachable=_is_unreachable(w.key) if w.is_remote else False,
        )
        for w in get_served_workspaces()
    ]
