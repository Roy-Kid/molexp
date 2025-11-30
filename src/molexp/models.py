"""Core data models for Project-Experiment-Run architecture."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ============ Enums ============


class RunStatus(str, Enum):
    """Status of a Run execution."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AssetType(str, Enum):
    """Type of Asset."""

    STRUCTURE = "structure"
    TRAJECTORY = "trajectory"
    TOPOLOGY = "topology"
    FORCEFIELD = "forcefield"
    IMAGE = "image"
    TABLE = "table"
    MODEL = "model"
    LOG = "log"
    OTHER = "other"


# ============ Project ============


class Project(BaseModel):
    """Top-level research container."""

    project_id: str = Field(..., description="Unique project identifier (slug)")
    name: str
    description: str = ""
    owner: str = ""
    tags: list[str] = Field(default_factory=list)
    created_at: datetime
    config: dict[str, Any] = Field(default_factory=dict)

    @field_validator("project_id")
    @classmethod
    def validate_project_id(cls, v: str) -> str:
        """Validate project_id is a valid slug."""
        if not v or not 3 <= len(v) <= 50:
            raise ValueError("project_id must be 3-50 characters")
        if not all(c.islower() or c.isdigit() or c == "-" for c in v):
            raise ValueError("project_id must contain only lowercase, digits, and hyphens")
        return v

    @property
    def path(self) -> str:
        """Relative path within workspace."""
        return f"projects/{self.project_id}"


# ============ Experiment ============


class WorkflowTemplate(BaseModel):
    """Reference to a workflow definition."""

    type: str = "taskgraph_v1"
    source: str = Field(..., description="Path to workflow file or module")
    git_commit: str | None = None


class Experiment(BaseModel):
    """Repeatable research question with parameter space."""

    experiment_id: str = Field(..., description="Unique experiment identifier (slug)")
    project_id: str
    name: str
    description: str = ""
    created_at: datetime
    workflow_template: WorkflowTemplate
    parameter_space: dict[str, Any] = Field(default_factory=dict)
    default_inputs: list[AssetRef] = Field(default_factory=list)

    @field_validator("experiment_id")
    @classmethod
    def validate_experiment_id(cls, v: str) -> str:
        """Validate experiment_id is a valid slug."""
        if not v or not 3 <= len(v) <= 50:
            raise ValueError("experiment_id must be 3-50 characters")
        if not all(c.islower() or c.isdigit() or c == "-" for c in v):
            raise ValueError("experiment_id must contain only lowercase, digits, and hyphens")
        return v

    @property
    def path(self) -> str:
        """Relative path within workspace."""
        return f"projects/{self.project_id}/experiments/{self.experiment_id}"


# ============ Run ============


class WorkflowSnapshot(BaseModel):
    """Snapshot of workflow at execution time."""

    git_commit: str | None = None
    workflow_file: str
    serialized_graph: str | None = None


class Run(BaseModel):
    """Single execution instance of a workflow."""

    run_id: str = Field(..., description="Unique run identifier (timestamp_shortid)")
    project_id: str
    experiment_id: str
    created_at: datetime
    finished_at: datetime | None = None
    status: RunStatus = RunStatus.PENDING
    parameters: dict[str, Any] = Field(default_factory=dict)
    workflow_snapshot: WorkflowSnapshot
    executor_info: dict[str, Any] = Field(default_factory=dict)
    working_dir: str
    logs_dir: str = "logs/"

    @property
    def path(self) -> str:
        """Relative path within workspace."""
        return f"projects/{self.project_id}/experiments/{self.experiment_id}/runs/{self.run_id}"


# ============ Asset ============


class AssetFile(BaseModel):
    """File within an Asset."""

    path: str
    size: int
    hash: str


class Asset(BaseModel):
    """Reusable digital artifact."""

    asset_id: str = Field(..., description="UUID or content hash")
    type: AssetType
    format: str = Field(..., description="File format, e.g., 'pdb', 'xtc', 'png'")
    created_at: datetime
    producer_run_id: str | None = None
    size_bytes: int
    content_hash: str = Field(..., description="SHA256 hash of content")
    mime_type: str = ""
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    files: list[AssetFile] = Field(default_factory=list)

    @property
    def path(self) -> str:
        """Relative path within workspace."""
        return f"assets/{self.asset_id}"


# ============ AssetRef ============


class AssetRef(BaseModel):
    """Lightweight reference to an Asset."""

    asset_id: str
    role: str = Field(..., description="Role in workflow, e.g., 'input_structure'")
    producer_run_id: str | None = None
    accessed_at: datetime | None = None
    produced_at: datetime | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


# ============ Context Snapshot ============


class RunContextSnapshot(BaseModel):
    """Snapshot of execution environment."""

    environment: dict[str, str] = Field(default_factory=dict)
    dependencies: dict[str, str] = Field(default_factory=dict)
    hardware: dict[str, Any] = Field(default_factory=dict)


# ============ Asset References Collection ============


class AssetRefsCollection(BaseModel):
    """Collection of input and output asset references for a run."""

    inputs: list[AssetRef] = Field(default_factory=list)
    outputs: list[AssetRef] = Field(default_factory=list)
