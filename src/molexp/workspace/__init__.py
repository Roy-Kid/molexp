"""Workspace module — file-system-backed experiment management.

Hierarchy: Workspace -> Project -> Experiment -> Run
Each level owns an AssetLibrary for scoped artifact storage.
"""

from .asset import Asset, AssetLibrary, AssetWorkflow
from .context import Context
from .experiment import Experiment
from .models import (
    ErrorInfo,
    ExecutionRecord,
    ExperimentMetadata,
    ProjectMetadata,
    RunMetadata,
    WorkflowSnapshotRef,
    WorkspaceMetadata,
)
from .param import GridSpace, Params, ParamSpace, UniformSpace
from .project import Project
from .run import Run, RunContext, RunStatus
from .workspace import Workspace

__all__ = [
    # Entities
    "Workspace",
    "Project",
    "Experiment",
    "Run",
    "RunContext",
    "RunStatus",
    # Metadata models
    "WorkspaceMetadata",
    "ProjectMetadata",
    "ExperimentMetadata",
    "RunMetadata",
    "ExecutionRecord",
    "ErrorInfo",
    "WorkflowSnapshotRef",
    # Assets
    "Asset",
    "AssetLibrary",
    "AssetWorkflow",
    # Parameters
    "ParamSpace",
    "Params",
    "GridSpace",
    "UniformSpace",
    # Context
    "Context",
]
