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

from .concepts import Experiment, Project, Run, Workspace
from .errors import ConceptExistsError, ConceptNotFoundError
from .folder import Folder, LinkScan
from .library import Library
from .models import ConceptMeta
from .ops import RETRYABLE_STATUSES, ExecutionRecord, RunOpsState, RunStatus
from .types import concept_type, register_concept_type, resolve_concept_type

__all__ = [
    "RETRYABLE_STATUSES",
    "ConceptExistsError",
    "ConceptMeta",
    "ConceptNotFoundError",
    "ExecutionRecord",
    "Experiment",
    "Folder",
    "Library",
    "LinkScan",
    "Project",
    "Run",
    "RunOpsState",
    "RunStatus",
    "Workspace",
    "concept_type",
    "register_concept_type",
    "resolve_concept_type",
]
