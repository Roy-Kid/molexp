"""Metadata models for workspace entities.

This module defines plain data models for persistence. These models:
- Contain only serializable fields
- Have no Python object references
- Perform no side effects during validation
- Support round-trip JSON serialization

The metadata models represent the minimal declarative state required to
persist workspace entities across process boundaries.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class WorkspaceMetadata(BaseModel):
    """Serializable workspace metadata.

    Contains only essential fields that must persist across process boundaries.
    No runtime references or computed properties.
    """

    id: str = Field(..., description="Workspace identifier")
    name: str = Field(..., description="Human-readable workspace name")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    projects: list[str] = Field(default_factory=list, description="List of project IDs")


class ProjectMetadata(BaseModel):
    """Serializable project metadata.

    Contains only essential fields that must persist across process boundaries.
    No runtime references or computed properties.
    """

    id: str = Field(..., description="Project identifier")
    name: str = Field(..., description="Human-readable project name")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    description: str = Field(default="", description="Project description")
    owner: str = Field(default="", description="Project owner")
    tags: list[str] = Field(default_factory=list, description="Project tags")
    config: dict[str, Any] = Field(default_factory=dict, description="Project configuration")
    experiments: list[str] = Field(default_factory=list, description="List of experiment IDs")
    assets: list[str] = Field(default_factory=list, description="List of asset names")


class ExperimentMetadata(BaseModel):
    """Serializable experiment metadata.
    
    Contains only essential fields that must persist across process boundaries.
    No runtime references or computed properties.
    """
    
    id: str = Field(..., description="Experiment identifier (UUID)")
    name: str = Field(..., description="Human-readable experiment name")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    description: str = Field(default="", description="Experiment description")
    tags: list[str] = Field(default_factory=list, description="Experiment tags")
    config: dict[str, Any] = Field(default_factory=dict, description="Experiment configuration")


class RunMetadata(BaseModel):
    """Serializable run metadata.
    
    Contains only essential fields that must persist across process boundaries.
    No runtime references or computed properties.
    
    Includes run-specific fields like parameters and assets.
    """
    
    id: str = Field(..., description="Run identifier (UUID)")
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Execution parameters"
    )
    assets: dict[str, Any] = Field(
        default_factory=dict,
        description="Input/output assets and artifacts"
    )
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    status: str = Field(default="pending", description="Run status")
    finished_at: datetime | None = Field(default=None, description="Completion timestamp")
    
    # Error summary (for quick indexing)
    error: dict[str, str] | None = Field(
        default=None,
        description="Error summary: type, message, timestamp"
    )

