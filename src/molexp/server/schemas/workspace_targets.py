"""API schemas for the workspace-target registry (``/api/workspace/targets``).

A *workspace target* describes a remote root that the server can open
as the active :class:`~molexp.workspace.Workspace` — distinct from a
:class:`~molexp.workspace.ComputeTarget` (which describes a scratch
host for run execution).  These two concepts share the connectivity
probe envelope; :class:`TargetTestCheck` / :class:`TargetTestResponse`
are re-imported from ``.targets`` rather than duplicated.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from molexp.server.workspace_targets import WorkspaceTarget

from .targets import TargetTestCheck, TargetTestResponse


class WorkspaceTargetCreateRequest(BaseModel):
    """Payload for ``POST /api/workspace/targets``."""

    name: str = Field(..., description="Unique slug-shaped identifier")
    host: str = Field(..., description="``user@host`` or bare hostname for SSH")
    root_path: str = Field(..., description="Absolute POSIX path on the remote host")
    port: int | None = Field(default=None, description="SSH port")
    identity_file: str | None = Field(
        default=None,
        description="Absolute path to an SSH identity file",
    )
    ssh_opts: list[str] = Field(
        default_factory=list,
        description="Extra ``ssh`` argv tokens",
    )


class WorkspaceTargetResponse(BaseModel):
    """Wire form for a :class:`WorkspaceTarget`."""

    name: str
    host: str
    root_path: str
    port: int | None = None
    identity_file: str | None = None
    ssh_opts: list[str] = Field(default_factory=list)

    @classmethod
    def from_model(cls, target: WorkspaceTarget) -> WorkspaceTargetResponse:
        return cls(
            name=target.name,
            host=target.host,
            root_path=target.root_path,
            port=target.port,
            identity_file=target.identity_file,
            ssh_opts=list(target.ssh_opts),
        )


class WorkspaceTargetListResponse(BaseModel):
    """Response for ``GET /api/workspace/targets``."""

    targets: list[WorkspaceTargetResponse]
    total: int


__all__ = [
    "TargetTestCheck",
    "TargetTestResponse",
    "WorkspaceTargetCreateRequest",
    "WorkspaceTargetListResponse",
    "WorkspaceTargetResponse",
]
