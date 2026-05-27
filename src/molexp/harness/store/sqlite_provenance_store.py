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
"""

from __future__ import annotations

import sqlite3
from collections import deque
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
        self._conn: sqlite3.Connection = open_db(self._path)
        self._artifacts = artifact_store

    def add_edge(
        self,
        parent_id: str,
        child_id: str,
        relation: str = "derived_from",
    ) -> None:
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

        direction="up" follows parent_id; direction="down" follows child_id.
        """
        if direction == "up":
            query = "SELECT parent_id FROM artifact_edges WHERE child_id = ?"
        elif direction == "down":
            query = "SELECT child_id FROM artifact_edges WHERE parent_id = ?"
        else:
            raise ValueError(f"unknown direction: {direction!r}")

        seen: set[str] = {start}
        order: list[str] = []
        queue: deque[str] = deque([start])
        while queue:
            current = queue.popleft()
            for (neighbour,) in self._conn.execute(query, (current,)).fetchall():
                if neighbour in seen:
                    continue
                seen.add(neighbour)
                order.append(neighbour)
                queue.append(neighbour)
        return order
