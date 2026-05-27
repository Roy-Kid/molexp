"""Persistence backends for ``molexp.harness``.

Three stores, each a Protocol + concrete impl pair (mirroring
:mod:`molexp.workflow.cache_store`):

- :class:`ArtifactStore` Protocol / :class:`FileArtifactStore` — content +
  refs + per-kind index on the filesystem, atomic writes via
  :func:`molexp.workspace.atomic_write_json` / ``atomic_write_text``, hashing
  via :func:`molexp.workspace.utils.compute_content_hash`.
- :class:`EventLog` Protocol / :class:`SQLiteEventLog` — append-only audit
  timeline, ``UNIQUE(run_id, seq)`` enforced by SQLite. *(lands in T6)*
- :class:`ProvenanceStore` Protocol / :class:`SQLiteProvenanceStore` —
  ``artifact_edges`` table + BFS lineage traversals. *(lands in T8)*

The private ``_sqlite`` helper is shared between the two SQLite-backed
stores and is not part of the public surface.
"""

from __future__ import annotations

from molexp.harness.store.artifact_store import ArtifactStore
from molexp.harness.store.event_log import EventLog
from molexp.harness.store.file_artifact_store import FileArtifactStore
from molexp.harness.store.provenance_store import ProvenanceStore
from molexp.harness.store.sqlite_event_log import SQLiteEventLog
from molexp.harness.store.sqlite_provenance_store import SQLiteProvenanceStore

__all__ = [
    "ArtifactStore",
    "EventLog",
    "FileArtifactStore",
    "ProvenanceStore",
    "SQLiteEventLog",
    "SQLiteProvenanceStore",
]
