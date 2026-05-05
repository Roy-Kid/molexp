"""API schemas for the workspace ComputeTarget registry.

A :class:`~molexp.workspace.ComputeTarget` is the cross-product of the
*transport* axis (local vs SSH) and the *scheduler* axis (local / slurm /
pbs / lsf).  These models are the wire format for ``/api/targets``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from molexp.workspace.models import ComputeTarget


class TargetCreateRequest(BaseModel):
    """Payload for ``POST /api/targets``."""

    name: str = Field(..., description="Unique target name within the workspace")
    scratch_root: str = Field(
        ...,
        alias="scratchRoot",
        description="Absolute scratch root on the target's filesystem",
    )
    scheduler: Literal["local", "slurm", "pbs", "lsf"] = Field(
        default="local", description="Dispatch axis"
    )
    host: str | None = Field(default=None, description="user@host for SSH; omit for local")
    port: int | None = Field(default=None, description="SSH port")
    identity_file: str | None = Field(
        default=None,
        alias="identityFile",
        description="Path to SSH identity file",
    )
    ssh_opts: list[str] = Field(
        default_factory=list,
        alias="sshOpts",
        description="Extra ssh argv tokens",
    )

    model_config = {"populate_by_name": True}


class TargetResponse(BaseModel):
    """Wire form for a :class:`ComputeTarget`."""

    name: str
    scratch_root: str = Field(..., alias="scratchRoot")
    scheduler: Literal["local", "slurm", "pbs", "lsf"]
    host: str | None = None
    port: int | None = None
    identity_file: str | None = Field(default=None, alias="identityFile")
    ssh_opts: list[str] = Field(default_factory=list, alias="sshOpts")
    is_remote: bool = Field(..., alias="isRemote")
    default_resources: dict[str, Any] = Field(default_factory=dict, alias="defaultResources")
    default_scheduling: dict[str, Any] = Field(default_factory=dict, alias="defaultScheduling")

    model_config = {"populate_by_name": True}

    @classmethod
    def from_model(cls, target: ComputeTarget) -> TargetResponse:
        return cls(
            name=target.name,
            scratchRoot=target.scratch_root,
            scheduler=target.scheduler,
            host=target.host,
            port=target.port,
            identityFile=target.identity_file,
            sshOpts=list(target.ssh_opts),
            isRemote=target.is_remote,
            defaultResources=dict(target.default_resources),
            defaultScheduling=dict(target.default_scheduling),
        )


class TargetListResponse(BaseModel):
    """Response for ``GET /api/targets``."""

    targets: list[TargetResponse]
    total: int


class TargetTestResponse(BaseModel):
    """Response for ``POST /api/targets/{name}/test``."""

    name: str
    ok: bool
    checks: list["TargetTestCheck"]
    error: str | None = None


class TargetTestCheck(BaseModel):
    """One step of the target connectivity probe."""

    label: str = Field(..., description="Human-readable check name")
    ok: bool
    detail: str | None = None


TargetTestResponse.model_rebuild()
