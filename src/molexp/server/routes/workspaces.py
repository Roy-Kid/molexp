"""``GET /api/workspaces`` — the set of workspaces this server is hosting.

``molexp serve`` can be pointed at one or more workspaces (local or remote).
This route exposes that set so the UI can list them and switch between them via
``POST /api/workspace/open`` (singular — active-workspace operations live in
``routes/workspace.py``). With a single served workspace this returns one row,
matching the unchanged single-workspace behaviour.
"""

from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from molexp.server.dependencies import (
    active_served_key,
    get_served_workspaces,
    get_workspace_by_key,
)
from molexp.server.deps.auth import get_optional_user
from molexp.server.exceptions import RemoteWorkspaceUnreachableError
from molexp.services.auth import AuthUser, get_auth_service, is_auth_enabled

router = APIRouter(prefix="/workspaces", tags=["workspaces"])


class ServedWorkspaceResponse(BaseModel):
    """One workspace the server is hosting."""

    key: str = Field(..., description="Stable switch handle, unique per server process")
    label: str = Field(..., description="Human-facing label (path or user@host:/path)")
    isRemote: bool = Field(..., description="True for an SSH-backed remote workspace")
    path: str | None = Field(default=None, description="Absolute local root, null when remote")
    active: bool = Field(
        default=False,
        description="True for the workspace the flat routes / active tree address",
    )
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
def list_workspaces(
    user: Annotated[AuthUser | None, Depends(get_optional_user)] = None,
) -> list[ServedWorkspaceResponse]:
    """List the workspaces ``molexp serve`` was started with.

    A remote workspace whose transport is currently unreachable is still
    listed, flagged ``unreachable`` so the UI can degrade gracefully rather
    than failing the whole list.

    When auth is enabled, the list is filtered by the user's workspace allowlist.
    """
    active_key = active_served_key()
    served = get_served_workspaces()
    if is_auth_enabled() and user is not None:
        allowed = set(get_auth_service().filter_workspaces(user, [w.key for w in served]))
        served = [w for w in served if w.key in allowed]
    return [
        ServedWorkspaceResponse(
            key=w.key,
            label=w.label,
            isRemote=w.is_remote,
            path=w.path,
            active=w.key == active_key,
            unreachable=_is_unreachable(w.key) if w.is_remote else False,
        )
        for w in served
    ]
