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
from .bundle import Backlink, Bundle
from .bundle_index import BundleIndex, ConceptIndexEntry
from .cache import WORKSPACE_CACHE_KIND, CacheFolder
from .concepts import Note, ReferenceConcept
from .context import Context
from .doc_embed import EntitySummary, summarize_entity
from .edges import DEFAULT_EDGE_ROLE, Edge, EdgeRole
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
from .events import (
    WORKSPACE_EVENTS_DB,
    WorkspaceEvent,
    WorkspaceEventLog,
    WorkspaceEventType,
    emit_workspace_event,
    read_workspace_events,
)
from .experiment import Experiment
from .folder import (
    WORKSPACE_EXPERIMENT_KIND,
    WORKSPACE_PROJECT_KIND,
    WORKSPACE_ROOT_KIND,
    WORKSPACE_RUN_KIND,
    Folder,
)
from .knowledge_item import (
    KNOWLEDGE_ITEM_KIND,
    KnowledgeItem,
    KnowledgeKind,
    KnowledgeMeta,
    SourceKind,
    SourceRef,
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
from .note_meta import NoteMeta
from .param import GridSpace, Params, ParamSpace, UniformSpace
from .project import Project
from .reference_meta import ReferenceMeta
from .run import RETRYABLE_STATUSES, Run, RunContext, RunStatus
from .run_reaper import pid_alive, reap_zombie_run
from .runset import RunRecord, RunSet, RunSetResult
from .target import (
    LocalTarget,
    RemoteTarget,
    SessionManager,
    SSHSession,
    Target,
    TargetNotFound,
    parse_target,
    resolve_target,
    target_to_transport,
)
from .targets import (
    LOCAL_TARGET_NAME,
    add_target,
    builtin_local_target,
    effective_targets,
    get_target,
    has_target,
    list_targets,
    remove_target,
    resolve_compute_target,
    target_run_dir,
    to_transport,
)
from .workspace import Workspace
from .workspace_context import (
    ArtifactRef,
    ContextFocus,
    ExperimentRef,
    HealthFlag,
    KnowledgeRef,
    ProjectRef,
    RunRef,
    WorkflowRef,
    WorkspaceContext,
    WorkspaceRef,
    assemble_workspace_context,
)
from .zotero_concepts import ZoteroItem, read_zotero_items

__all__ = [
    "DEFAULT_EDGE_ROLE",
    # OKF KnowledgeItem Concept (integration P0.4) — typed, source-linked
    "KNOWLEDGE_ITEM_KIND",
    # Built-in ``local`` compute target (targets-merge)
    "LOCAL_TARGET_NAME",
    # Retryable-status domain (resume / rerun verb selection)
    "RETRYABLE_STATUSES",
    # Folder kind taxonomy (unify-folder-abstraction-02)
    "WORKSPACE_CACHE_KIND",
    # Workspace event spine (integration P0.3) — append-only cross-object timeline
    "WORKSPACE_EVENTS_DB",
    "WORKSPACE_EXPERIMENT_KIND",
    "WORKSPACE_PROJECT_KIND",
    "WORKSPACE_ROOT_KIND",
    "WORKSPACE_RUN_KIND",
    "ArtifactAsset",
    # WorkspaceContext read-model + assembler (integration P0.2)
    "ArtifactRef",
    # Assets
    "Asset",
    "AssetManifest",
    "AssetScope",
    "AssetsView",
    # OKF Note backlink (knowledge-docs-01) — a derived reverse-edge row
    "Backlink",
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
    "ContextFocus",
    "DataAsset",
    "DataAssetLibrary",
    # OKF typed knowledge-graph edge role (typed-provenance-edge P0.1)
    "Edge",
    "EdgeRole",
    # OKF document-embed entity summary (knowledge-docs-05) — read-only UI card
    "EntitySummary",
    "ErrorInfo",
    "ErrorTraceAsset",
    "ExecutionRecord",
    "Experiment",
    # Workspace error hierarchy
    "ExperimentExistsError",
    "ExperimentMetadata",
    "ExperimentNotFoundError",
    "ExperimentRef",
    # Folder abstraction (unify-folder-abstraction-01)
    "Folder",
    "FolderMetadata",
    "FolderMoveCollisionError",
    "GridSpace",
    "HealthFlag",
    "KnowledgeItem",
    "KnowledgeKind",
    "KnowledgeMeta",
    "KnowledgeRef",
    # Target types + session management (unified workspace CLI)
    "LocalTarget",
    "LogAsset",
    # OKF Note Concept (wsokf-05) — a directory whose path is its identity
    "Note",
    # OKF Note document meta.yaml payload (knowledge-docs-05) — tags + status
    "NoteMeta",
    # Parameters
    "ParamSpace",
    "Params",
    "Producer",
    "Project",
    "ProjectExistsError",
    "ProjectMetadata",
    "ProjectNotFoundError",
    "ProjectRef",
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
    "RunRecord",
    "RunRef",
    "RunSet",
    "RunSetResult",
    "RunStatus",
    "SSHSession",
    "SessionManager",
    "SourceKind",
    "SourceRef",
    "Target",
    "TargetNotFound",
    "UniformSpace",
    "WorkflowRef",
    # Entities
    "Workspace",
    "WorkspaceContext",
    "WorkspaceEvent",
    "WorkspaceEventLog",
    "WorkspaceEventType",
    # Metadata models
    "WorkspaceMetadata",
    "WorkspaceRef",
    # OKF read-only Zotero importer (wsokf-05) — produces ReferenceConcepts
    "ZoteroItem",
    # Compute target helpers
    "add_target",
    "assemble_workspace_context",
    # Atomic JSON I/O — used by workflow layer's persistence + agent
    # layer's session storage.
    "atomic_write_json",
    # Atomic plain-text I/O — companion to atomic_write_json for
    # markdown reports / generated source previews / log snapshots.
    "atomic_write_text",
    "builtin_local_target",
    "effective_targets",
    "emit_workspace_event",
    "get_target",
    "has_target",
    "list_targets",
    "parse_target",
    "pid_alive",
    "read_workspace_events",
    "read_zotero_items",
    "reap_zombie_run",
    "remove_target",
    "resolve_compute_target",
    "resolve_target",
    "summarize_entity",
    "target_run_dir",
    "target_to_transport",
    "to_transport",
]
