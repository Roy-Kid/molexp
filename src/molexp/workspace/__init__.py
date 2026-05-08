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
    ComputeTarget,
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
from .targets import (
    add_target,
    get_target,
    has_target,
    list_targets,
    remove_target,
    target_run_dir,
    to_transport,
)
from .workspace import Workspace

__all__ = [
    "ArtifactAsset",
    # Assets
    "Asset",
    "AssetCatalog",
    "AssetManifest",
    "AssetScope",
    "AssetsView",
    "CheckpointAsset",
    "ComputeTarget",
    # Context
    "Context",
    "DataAsset",
    "DataAssetLibrary",
    "ErrorInfo",
    "ErrorTraceAsset",
    "ExecutionRecord",
    "ExecutionStateAsset",
    "Experiment",
    "ExperimentMetadata",
    "GridSpace",
    "LogAsset",
    "OutputAsset",
    # Parameters
    "ParamSpace",
    "Params",
    "Producer",
    "Project",
    "ProjectMetadata",
    "Run",
    "RunContext",
    "RunMetadata",
    "RunStatus",
    "UniformSpace",
    "WorkflowSnapshotRef",
    # Entities
    "Workspace",
    # Metadata models
    "WorkspaceMetadata",
    # Compute target helpers
    "add_target",
    "get_target",
    "has_target",
    "list_targets",
    "remove_target",
    "target_run_dir",
    "to_transport",
]
