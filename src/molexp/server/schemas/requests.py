"""Pydantic request models for MolExp API.

Aligned with workspace.models — field names match domain models.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# ── Workspace ───────────────────────────────────────────────────────────────


class WorkspaceOpenRequest(BaseModel):
    path: str = Field(..., description="Absolute path to the workspace")
    create_if_missing: bool = Field(False, description="Create if missing")


# ── Project ─────────────────────────────────────────────────────────────────


class ProjectCreateRequest(BaseModel):
    name: str = Field(..., description="Human-readable project name")
    description: str = Field("", description="Project description")
    owner: str = Field("", description="Project owner")
    tags: list[str] = Field(default_factory=list, description="Project tags")


class ProjectUpdateRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    owner: str | None = None
    tags: list[str] | None = None
    config: dict[str, Any] | None = None


# ── Experiment ──────────────────────────────────────────────────────────────


class ExperimentCreateRequest(BaseModel):
    name: str = Field(..., description="Human-readable experiment name")
    workflow_source: str | None = Field(None, description="Path to workflow file")
    description: str = Field("", description="Experiment description")
    parameter_space: dict[str, Any] = Field(
        default_factory=dict, description="Parameter space definition"
    )


# ── Run ─────────────────────────────────────────────────────────────────────


class RunCreateRequest(BaseModel):
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Run parameters"
    )


class RunStatusUpdateRequest(BaseModel):
    status: str = Field(..., description="New status value")


# ── Execution ───────────────────────────────────────────────────────────────


class ExecutionCreateRequest(BaseModel):
    workflow_json: dict[str, Any] = Field(..., description="Serialized workflow graph")
    project_id: str = Field(..., description="Target project ID")
    experiment_id: str = Field(..., description="Target experiment ID")
    parameters: dict[str, Any] = Field(default_factory=dict)


# ── Asset ───────────────────────────────────────────────────────────────────


class AssetUpdateRequest(BaseModel):
    tags: list[str] | None = None
    metadata: dict[str, Any] | None = None


# ── Agent ───────────────────────────────────────────────────────────────────


class GoalCreateRequest(BaseModel):
    description: str = Field(..., description="Natural language goal description")
    constraints: dict[str, Any] = Field(default_factory=dict)
    success_criteria: list[str] = Field(default_factory=list)


class ApprovalRespondRequest(BaseModel):
    request_id: str = Field(..., description="Approval request ID")
    approved: bool = Field(..., description="Whether to approve")
