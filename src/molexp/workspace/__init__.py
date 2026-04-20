"""Workspace module — file-system-backed experiment management.

Hierarchy: Workspace -> Project -> Experiment -> Run

Each scope exposes:

- ``{scope}.assets``       — read-only catalog view (typed Asset queries)
- ``{scope}.data_assets``  — ``DataAssetLibrary`` for importing user inputs
- ``workspace.catalog``    — full workspace-level ``AssetCatalog``
"""

from .assets import (
    ArtifactAsset,
    Asset,
    AssetCatalog,
    AssetManifest,
    AssetScope,
    AssetsView,
    CheckpointAsset,
    DataAsset,
    DataAssetLibrary,
    ErrorTraceAsset,
    ExecutionStateAsset,
    LogAsset,
    OutputAsset,
    Producer,
)
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
    "ArtifactAsset",
    "AssetCatalog",
    "AssetManifest",
    "AssetScope",
    "AssetsView",
    "CheckpointAsset",
    "DataAsset",
    "DataAssetLibrary",
    "ErrorTraceAsset",
    "ExecutionStateAsset",
    "LogAsset",
    "OutputAsset",
    "Producer",
    # Parameters
    "ParamSpace",
    "Params",
    "GridSpace",
    "UniformSpace",
    # Context
    "Context",
]
