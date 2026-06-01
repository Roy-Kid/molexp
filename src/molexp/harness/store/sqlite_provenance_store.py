"""SQLite implementation of :class:`ProvenanceStore`.

Shares the database file (and therefore the ``artifact_edges`` table) with
:class:`SQLiteEventLog` via :func:`molexp.harness.store._sqlite.open_db`.
Traversals are BFS:

- ``trace_backward(C)`` walks ``parent_id`` edges from ``C``, returning
  ancestors in level order.
- ``trace_forward(A)`` walks ``child_id`` edges from ``A``, returning
  descendants in level order.

Each visited id is hydrated into an :class:`ArtifactRef` via the supplied
:class:`ArtifactStore` so callers get the full metadata, not just an id.

The edge walk runs as a single ``WITH RECURSIVE`` CTE rather than one
``SELECT`` per node: ``validate_provenance`` calls ``trace_backward`` once per
artifact, so a per-node BFS was O(artifacts x lineage) round-trips. SQLite
processes a recursive CTE's working table as a **FIFO queue**, so a ``UNION``
recursion over the bare id emits rows in BFS discovery order — the same order
the old Python ``deque`` produced, with each id appearing once at its
shallowest depth. ``UNION`` (not ``UNION ALL``) dedups on the id, which is also
what terminates the recursion on a malformed cycle — a depth column would make
every ``(id, depth)`` row unique and defeat that guard, so it is deliberately
omitted. A first-occurrence pass in Python re-applies the old ``seen`` set,
excluding ``start`` itself.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from molexp.harness.schemas import ArtifactRef
from molexp.harness.store._sqlite import open_db
from molexp.harness.store.artifact_store import ArtifactStore

__all__ = ["SQLiteProvenanceStore"]


class SQLiteProvenanceStore:
    """SQLite-backed artifact-lineage store."""

    def __init__(self, path: Path, artifact_store: ArtifactStore) -> None:
        self._path = Path(path)
        # Shares the per-DB-file lock with the SQLiteEventLog on the same path
        # (see ``store._sqlite``); all connection access serializes through it
        # because ``StageRunner`` drives add_edge from worker threads.
        self._conn, self._lock = open_db(self._path)
        self._artifacts = artifact_store

    def add_edge(
        self,
        parent_id: str,
        child_id: str,
        relation: str = "derived_from",
    ) -> None:
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO artifact_edges (parent_id, child_id, relation, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (parent_id, child_id, relation, datetime.now(tz=UTC).isoformat()),
                )
            except sqlite3.IntegrityError:
                # PRIMARY KEY (parent_id, child_id, relation) — duplicate is a no-op.
                return

    def trace_backward(self, artifact_id: str) -> list[ArtifactRef]:
        order = self._bfs(artifact_id, direction="up")
        return [self._artifacts.get_ref(aid) for aid in order]

    def trace_forward(self, artifact_id: str) -> list[ArtifactRef]:
        order = self._bfs(artifact_id, direction="down")
        return [self._artifacts.get_ref(aid) for aid in order]

    def lineage_graph(self, artifact_id: str) -> dict[str, Any]:
        # Union of ancestors + descendants + the seed itself.
        ids = {
            artifact_id,
            *self._bfs(artifact_id, direction="up"),
            *self._bfs(artifact_id, direction="down"),
        }
        nodes: list[dict[str, Any]] = []
        for aid in sorted(ids):
            try:
                ref = self._artifacts.get_ref(aid)
                nodes.append({"id": aid, "kind": ref.kind, "uri": ref.uri})
            except Exception:
                nodes.append({"id": aid})
        # Pull every edge touching any node in the subgraph.
        placeholders = ",".join(["?"] * len(ids))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT parent_id, child_id, relation FROM artifact_edges "
                f"WHERE parent_id IN ({placeholders}) OR child_id IN ({placeholders})",
                (*ids, *ids),
            ).fetchall()
        edges = [
            {"parent_id": p, "child_id": c, "relation": r}
            for p, c, r in rows
            if p in ids and c in ids
        ]
        return {"nodes": nodes, "edges": edges}

    # ----------------------------------------------------------- internals

    def _bfs(self, start: str, *, direction: str) -> list[str]:
        """BFS from ``start`` (exclusive) returning visited ids in level order.

        ``direction="up"`` follows ``parent_id`` (ancestors); ``direction="down"``
        follows ``child_id`` (descendants). Runs one recursive-CTE query whose
        FIFO emission order matches a ``deque`` BFS; first-occurrence dedup in
        Python keeps each id once at its shallowest depth.
        """
        if direction == "up":
            # ancestors: seed = parents of ``start``; recurse child_id -> parent_id.
            cte = (
                "WITH RECURSIVE walk(id) AS ("
                "  SELECT parent_id FROM artifact_edges WHERE child_id = ?"
                "  UNION"
                "  SELECT e.parent_id FROM artifact_edges e"
                "  JOIN walk w ON e.child_id = w.id"
                ") SELECT id FROM walk"
            )
        elif direction == "down":
            # descendants: seed = children of ``start``; recurse parent_id -> child_id.
            cte = (
                "WITH RECURSIVE walk(id) AS ("
                "  SELECT child_id FROM artifact_edges WHERE parent_id = ?"
                "  UNION"
                "  SELECT e.child_id FROM artifact_edges e"
                "  JOIN walk w ON e.parent_id = w.id"
                ") SELECT id FROM walk"
            )
        else:
            raise ValueError(f"unknown direction: {direction!r}")

        # Single round-trip; serialize on the shared lock. Hydration (get_ref)
        # runs in the callers, outside this lock, so it never blocks on FS I/O.
        with self._lock:
            rows = self._conn.execute(cte, (start,)).fetchall()

        seen: set[str] = {start}
        order: list[str] = []
        for (neighbour,) in rows:
            if neighbour in seen:
                continue
            seen.add(neighbour)
            order.append(neighbour)
        return order
