"""Workspace module — file-system-backed storage primitive.

Hierarchy: Workspace -> Project -> Experiment -> Run

Workspace is the bottom of the molexp dependency DAG: it knows about
filesystem layout, atomic JSON I/O, content-addressed assets, and
generic subsystem storage — and nothing about workflows, sessions,
agents, or LLMs. The workflow layer uses workspace for caching and
persistence; the agent layer uses workspace for session storage.
Cross-layer payloads are stored as opaque JSON dicts here; the
upstream layers own the typed shape and own the typed parsing on
read-back.

Each scope exposes:

- ``{scope}.assets``       — read-only catalog view (typed Asset queries)
- ``{scope}.data_assets``  — ``DataAssetLibrary`` for importing user inputs
- ``workspace.catalog``    — full workspace-level ``AssetCatalog``
- ``workspace.subsystem_store(kind)`` — generic per-kind private dir
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
from .base import atomic_write_json, atomic_write_text
from .context import Context
from .errors import (
    ExperimentExistsError,
    ExperimentNotFoundError,
    FolderMoveCollisionError,
    ProjectExistsError,
    ProjectNotFoundError,
    RunExistsError,
    RunNotFoundError,
)
from .experiment import Experiment
from .folder import Folder
from .models import (
    ComputeTarget,
    ErrorInfo,
    ExecutionRecord,
    ExperimentMetadata,
    FolderMetadata,
    ProjectMetadata,
    RunMetadata,
    WorkspaceMetadata,
)
from .param import GridSpace, Params, ParamSpace, UniformSpace
from .project import Project
from .run import Run, RunContext, RunStatus
from .subsystem import SubsystemStore
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
    # Workspace error hierarchy
    "ExperimentExistsError",
    "ExperimentMetadata",
    "ExperimentNotFoundError",
    # Folder abstraction (unify-folder-abstraction-01)
    "Folder",
    "FolderMetadata",
    "FolderMoveCollisionError",
    "GridSpace",
    "LogAsset",
    "OutputAsset",
    # Parameters
    "ParamSpace",
    "Params",
    "Producer",
    "Project",
    "ProjectExistsError",
    "ProjectMetadata",
    "ProjectNotFoundError",
    "Run",
    "RunContext",
    "RunExistsError",
    "RunMetadata",
    "RunNotFoundError",
    "RunStatus",
    # Subsystem storage primitive
    "SubsystemStore",
    "UniformSpace",
    # Entities
    "Workspace",
    # Metadata models
    "WorkspaceMetadata",
    # Compute target helpers
    "add_target",
    # Atomic JSON I/O — used by workflow layer's persistence + agent
    # layer's session storage.
    "atomic_write_json",
    # Atomic plain-text I/O — companion to atomic_write_json for
    # markdown reports / generated source previews / log snapshots.
    "atomic_write_text",
    "get_target",
    "has_target",
    "list_targets",
    "remove_target",
    "target_run_dir",
    "to_transport",
]
