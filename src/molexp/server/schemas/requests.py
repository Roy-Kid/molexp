"""Pydantic request models for MolExp API.

This module defines standardized request models used across all API endpoints.
Consolidating request models provides:
- Single source of truth for request validation
- Consistent naming conventions
- Better API documentation
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# ============================================================================
# Workspace Requests
# ============================================================================


class WorkspaceFolderAddRequest(BaseModel):
    """Request to add a workspace folder."""

    path: str = Field(..., description="Absolute path to the folder")
    name: str | None = Field(None, description="Display name (defaults to folder name)")


class WorkspaceOpenRequest(BaseModel):
    """Request to open a workspace path."""

    path: str = Field(..., description="Absolute path to the workspace")
    create_if_missing: bool = Field(
        False, description="Create workspace metadata if missing"
    )


# ============================================================================
# File Requests
# ============================================================================


class FileContentUpdateRequest(BaseModel):
    """Request to update file content."""

    folder_id: str = Field(..., description="Workspace folder ID or 'workspace'")
    path: str = Field(..., description="Relative path within the folder")
    content: str = Field(..., description="New file content")


class DirectoryCreateRequest(BaseModel):
    """Request to create a directory."""

    folder_id: str = Field(..., description="Workspace folder ID or 'workspace'")
    path: str = Field(..., description="Relative path for new directory")


# ============================================================================
# Project Requests
# ============================================================================


class ProjectCreateRequest(BaseModel):
    """Request to create a project."""

    id: str = Field(
        ...,
        description="Unique project identifier (slug)",
        min_length=3,
        max_length=50,
        pattern=r"^[a-z0-9-]+$",
    )
    name: str = Field(..., description="Human-readable project name")
    description: str = Field("", description="Project description")
    owner: str = Field("", description="Project owner")
    tags: list[str] = Field(default_factory=list, description="Project tags")


class ProjectUpdateRequest(BaseModel):
    """Request to update a project."""

    name: str | None = None
    description: str | None = None
    owner: str | None = None
    tags: list[str] | None = None
    config: dict[str, Any] | None = None


# ============================================================================
# Experiment Requests
# ============================================================================


class ExperimentCreateRequest(BaseModel):
    """Request to create an experiment."""

    id: str = Field(
        ...,
        description="Unique experiment identifier (slug)",
        min_length=3,
        max_length=50,
        pattern=r"^[a-z0-9-]+$",
    )
    name: str = Field(..., description="Human-readable experiment name")
    workflow_source: str = Field(..., description="Path to workflow file")
    description: str = Field("", description="Experiment description")
    parameter_space: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameter space definition",
    )


class ExperimentUpdateRequest(BaseModel):
    """Request to update an experiment."""

    name: str | None = None
    description: str | None = None
    parameter_space: dict[str, Any] | None = None


# ============================================================================
# Run Requests
# ============================================================================


class RunCreateRequest(BaseModel):
    """Request to create a run."""

    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Run parameters",
    )
    workflow_file: str = Field(..., description="Workflow file path")
    git_commit: str | None = Field(None, description="Git commit hash")


class RunStatusUpdateRequest(BaseModel):
    """Request to update run status."""

    status: str = Field(..., description="New status value")


# ============================================================================
# Execution Requests
# ============================================================================


class GenericExecutionRequest(BaseModel):
    """Request for generic execution (playground)."""

    name: str | None = Field(None, description="Execution name")
    workflowSnapshot: dict[str, Any] | None = Field(
        None,
        description="Serialized workflow graph",
    )


class ExecutionCreateRequest(BaseModel):
    """Request to create a new execution in a specific context."""

    workflow_json: dict[str, Any] = Field(..., description="Serialized workflow graph")
    project_id: str = Field(..., description="Target Project ID")
    experiment_id: str = Field(..., description="Target Experiment ID")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Execution parameters")


class ExecutionPlanRequest(BaseModel):
    """Request for execution plan."""

    workflow_json: str = Field(..., description="Workflow definition as JSON string")
    targets: list[str] | None = Field(
        None,
        description="Target node IDs (defaults to workflow targets)",
    )


# ============================================================================
# Asset Requests
# ============================================================================


class AssetUpdateRequest(BaseModel):
    """Request to update asset metadata."""

    tags: list[str] | None = None
    metadata: dict[str, Any] | None = None
