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

This sub-spec (okf-01-02) ships the ``meta.yaml`` model and the typed error
family; ``Folder`` (okf-01-03) and ``Library`` (okf-01-04) follow.
"""

from .errors import ConceptExistsError, ConceptNotFoundError
from .models import ConceptMeta

__all__ = [
    "ConceptExistsError",
    "ConceptMeta",
    "ConceptNotFoundError",
]
