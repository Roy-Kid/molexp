"""``molexp.knowledge`` — Open Knowledge Format (OKF) storage substrate.

The bottom layer of the molexp dependency DAG (peer of ``molexp.workspace``
during the OKF rewrite). Represents every asset as an OKF *Concept*: a
directory whose path is its identity, physically split into ``meta.yaml``
(structured — :class:`ConceptMeta`) and ``index.md`` (narrative + the
markdown-link knowledge graph), with hot machine state isolated to a
``_ops/`` sidecar.

Allowed imports for this layer: stdlib, pydantic, pyyaml, and the
sanctioned cross-layer primitives (``molexp.atomicio``, ``molexp.ids``,
``molexp.path``, ``mollog``, ``molcfg``). It MUST NOT import
``molexp.workspace`` or any upstream layer — enforced by
``tests/test_knowledge/test_import_guard.py``.

The ``meta.yaml`` model + typed errors (okf-01-02), the ``Folder``
Concept-on-disk base (okf-01-03), the ``Library`` bundle façade
(okf-01-04 — ``walk`` / ``get`` / ``put`` / ``link``), and the typed
storage hierarchy ``Workspace`` / ``Project`` / ``Experiment`` / ``Run`` over
an open concept-type registry (okf-02 — ``@concept_type`` reconstructs the
right subclass from each Concept's ``meta.yaml`` ``type``) are in place.
"""

from .concepts import Experiment, Note, Project, Reference, Run, Workspace
from .errors import ConceptExistsError, ConceptNotFoundError
from .folder import Folder, LinkScan
from .fs import FileSystem, LocalFileSystem
from .index import ConceptIndexEntry, LibraryIndex
from .library import Library
from .models import ConceptMeta
from .ops import RETRYABLE_STATUSES, ExecutionRecord, RunOpsState, RunStatus
from .references import ReferenceMeta
from .run_lifecycle import (
    RunHeartbeat,
    RunNotRetryableError,
    cancel_run,
    claim_ownership,
    finish_run,
    make_execution_id,
    reap_run_if_stale,
    rerun_run,
    resumable_execution_id,
    resume_run,
    should_reap,
)
from .types import concept_type, register_concept_type, resolve_concept_type
from .zotero import ZoteroItem

__all__ = [
    "RETRYABLE_STATUSES",
    "ConceptExistsError",
    "ConceptIndexEntry",
    "ConceptMeta",
    "ConceptNotFoundError",
    "ExecutionRecord",
    "Experiment",
    "FileSystem",
    "Folder",
    "Library",
    "LibraryIndex",
    "LinkScan",
    "LocalFileSystem",
    "Note",
    "Project",
    "Reference",
    "ReferenceMeta",
    "Run",
    "RunHeartbeat",
    "RunNotRetryableError",
    "RunOpsState",
    "RunStatus",
    "Workspace",
    "ZoteroItem",
    "cancel_run",
    "claim_ownership",
    "concept_type",
    "finish_run",
    "make_execution_id",
    "reap_run_if_stale",
    "register_concept_type",
    "rerun_run",
    "resolve_concept_type",
    "resumable_execution_id",
    "resume_run",
    "should_reap",
]
