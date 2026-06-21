"""Workspace module — file-system-backed storage primitive.

Hierarchy: Workspace -> Project -> Experiment -> Run

Workspace is the bottom of the molexp dependency DAG: it knows about
filesystem layout, atomic JSON I/O, content-addressed assets, and
typed system folders — and nothing about workflows, sessions, agents,
or LLMs. The workflow layer uses workspace for caching and
persistence; the agent layer uses workspace for session storage.
Cross-layer payloads are stored as opaque JSON dicts here; the
upstream layers own the typed shape and own the typed parsing on
read-back.

Each scope exposes:

- ``{scope}.assets``       — read-only catalog view (typed Asset queries)
- ``{scope}.data_assets``  — ``DataAssetLibrary`` for importing user inputs
- ``workspace.catalog``    — full workspace-level ``AssetCatalog`` (singleton property)
- ``workspace.cache``      — ``CacheFolder`` (singleton property; exposes ``as_cache_store()``)

Upstream layers extend the workspace tree by importing the public
``Folder`` base class and mounting their own subclasses via the
generic five-verb CRUD — see ``molexp.agent.folders`` for the
``Agent`` / ``AgentSession`` pair.
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
    LogAsset,
    NoteAsset,
    Producer,
)
from .base import atomic_write_json, atomic_write_text
from .cache import WORKSPACE_CACHE_KIND, CacheFolder
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
from .folder import (
    WORKSPACE_EXPERIMENT_KIND,
    WORKSPACE_PROJECT_KIND,
    WORKSPACE_ROOT_KIND,
    WORKSPACE_RUN_KIND,
    Folder,
)
from .library import Library, LibraryIndex, NoteEntry, Reference, ReferenceStore
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
from .run import RETRYABLE_STATUSES, Run, RunContext, RunStatus
from .target import (
    LocalTarget,
    RemoteTarget,
    Session,
    SessionManager,
    Target,
    TargetNotFound,
    parse_target,
    resolve_target,
    target_to_transport,
)
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
    # Retryable-status domain (resume / rerun verb selection)
    "RETRYABLE_STATUSES",
    # Folder kind taxonomy (unify-folder-abstraction-02)
    "WORKSPACE_CACHE_KIND",
    "WORKSPACE_EXPERIMENT_KIND",
    "WORKSPACE_PROJECT_KIND",
    "WORKSPACE_ROOT_KIND",
    "WORKSPACE_RUN_KIND",
    "ArtifactAsset",
    # Assets
    "Asset",
    "AssetCatalog",
    "AssetManifest",
    "AssetScope",
    "AssetsView",
    # System folders (unify-folder-abstraction-03)
    "CacheFolder",
    "CheckpointAsset",
    "ComputeTarget",
    # Context
    "Context",
    "DataAsset",
    "DataAssetLibrary",
    "ErrorInfo",
    "ErrorTraceAsset",
    "ExecutionRecord",
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
    # Library — notes + references per scope
    "Library",
    "LibraryIndex",
    # Target types + session management (unified workspace CLI)
    "LocalTarget",
    "LogAsset",
    "NoteAsset",
    "NoteEntry",
    # Parameters
    "ParamSpace",
    "Params",
    "Producer",
    "Project",
    "ProjectExistsError",
    "Reference",
    "ReferenceStore",
    "ProjectMetadata",
    "ProjectNotFoundError",
    "RemoteTarget",
    "Run",
    "RunContext",
    "RunExistsError",
    "RunMetadata",
    "RunNotFoundError",
    "RunStatus",
    "Session",
    "SessionManager",
    "Target",
    "TargetNotFound",
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
    "parse_target",
    "remove_target",
    "resolve_target",
    "target_run_dir",
    "target_to_transport",
    "to_transport",
]
