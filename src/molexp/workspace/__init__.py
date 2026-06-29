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

Notes + literature are owned by the OKF Concepts (``Note`` /
``ReferenceConcept`` + its typed ``ReferenceMeta``), reached via the
``Bundle`` façade / ``concept_from_dir`` — directories whose path is
their identity. ``ZoteroItem`` / ``read_zotero_items`` are the
read-only Zotero importer that produces ``ReferenceConcept`` records
(PDFs pointed at, never copied).

Each scope exposes:

- ``{scope}.assets``       — read-only asset view (typed Asset queries over the manifests)
- ``{scope}.data_assets``  — ``DataAssetLibrary`` for importing user inputs
- ``workspace.cache``      — ``CacheFolder`` (singleton property; exposes ``as_cache_store()``)

Upstream layers extend the workspace tree by importing the public
``Folder`` base class and mounting their own subclasses via the
generic five-verb CRUD — see ``molexp.agent.folders`` for the
``Agent`` / ``AgentSession`` pair.
"""

from .assets import (
    ArtifactAsset,
    Asset,
    AssetManifest,
    AssetScope,
    AssetsView,
    CheckpointAsset,
    DataAsset,
    DataAssetLibrary,
    ErrorTraceAsset,
    LogAsset,
    Producer,
)
from .base import atomic_write_json, atomic_write_text
from .bundle import Bundle
from .bundle_index import BundleIndex, ConceptIndexEntry
from .cache import WORKSPACE_CACHE_KIND, CacheFolder
from .concepts import Note, ReferenceConcept
from .context import Context
from .errors import (
    ConceptNotFoundError,
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
from .reference_meta import ReferenceMeta
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
from .zotero_concepts import ZoteroItem, read_zotero_items

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
    "AssetManifest",
    "AssetScope",
    "AssetsView",
    # OKF bundle façade (wsokf-04) — distinct from the per-scope Library
    "Bundle",
    "BundleIndex",
    # System folders (unify-folder-abstraction-03)
    "CacheFolder",
    "CheckpointAsset",
    "ComputeTarget",
    "ConceptIndexEntry",
    "ConceptNotFoundError",
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
    # Target types + session management (unified workspace CLI)
    "LocalTarget",
    "LogAsset",
    # OKF Note Concept (wsokf-05) — a directory whose path is its identity
    "Note",
    # Parameters
    "ParamSpace",
    "Params",
    "Producer",
    "Project",
    "ProjectExistsError",
    "ProjectMetadata",
    "ProjectNotFoundError",
    # OKF Reference Concept (wsokf-05) — a directory whose path is its
    # identity. Its typed meta.yaml payload is ReferenceMeta.
    "ReferenceConcept",
    "ReferenceMeta",
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
    # OKF read-only Zotero importer (wsokf-05) — produces ReferenceConcepts
    "ZoteroItem",
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
    "read_zotero_items",
    "remove_target",
    "resolve_target",
    "target_run_dir",
    "target_to_transport",
    "to_transport",
]
