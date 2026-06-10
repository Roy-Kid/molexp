"""Persistence backends for ``molexp.harness``.

Three stores, each a Protocol + concrete impl pair (mirroring
:mod:`molexp.workflow.cache_store`):

- :class:`ArtifactStore` Protocol / :class:`FileArtifactStore` — content +
  refs + per-kind index on the filesystem, atomic writes via
  :func:`molexp.workspace.atomic_write_json` / ``atomic_write_text``, hashing
  via :func:`molexp.workspace.utils.compute_content_hash`.
- :class:`EventLog` Protocol / :class:`SQLiteEventLog` — append-only audit
  timeline, ``UNIQUE(run_id, seq)`` enforced by SQLite.
- :class:`ArtifactLineageStore` Protocol / :class:`SQLiteArtifactLineageStore` —
  ``artifact_edges`` table (``derived_from`` edges stamped with the producing
  stage + run id) + BFS lineage traversals. Scoped strictly to
  pipeline-artifact lineage; run-level provenance (params, config, env,
  workflow identity) is owned by :mod:`molexp.workspace`.

The private ``_sqlite`` helper is shared between the two SQLite-backed
stores and is not part of the public surface.
"""

from __future__ import annotations

from molexp.harness.store.artifact_store import ArtifactStore
from molexp.harness.store.event_log import EventLog
from molexp.harness.store.file_artifact_store import FileArtifactStore
from molexp.harness.store.lineage_store import ArtifactLineageStore
from molexp.harness.store.sqlite_event_log import SQLiteEventLog
from molexp.harness.store.sqlite_lineage_store import SQLiteArtifactLineageStore

__all__ = [
    "ArtifactLineageStore",
    "ArtifactStore",
    "EventLog",
    "FileArtifactStore",
    "SQLiteArtifactLineageStore",
    "SQLiteEventLog",
]
