"""Headline benchmark for perf-hardening-02 (ac-008).

Measures the edge-walk SQL statement count and wall-clock for a single
``SQLiteArtifactLineageStore.trace_backward`` over a deep synthetic lineage.

The store's BFS used to emit one SQL statement per visited node (O(N));
the recursive-CTE rewrite collapses the edge walk to a single statement
(or O(depth)). This script reports both numbers so the before/after drop
is directly comparable. It is a MEASUREMENT harness, not a test: it never
asserts, it only prints.

Run::

    python -m benches.bench_provenance_lineage

Against the current per-node BFS it reports ~N statements; after the
rewrite it reports ~1, while the returned id set is unchanged.
"""

from __future__ import annotations

import tempfile
import time
from itertools import pairwise
from pathlib import Path

from molexp.harness.store.file_artifact_store import FileArtifactStore
from molexp.harness.store.sqlite_lineage_store import SQLiteArtifactLineageStore

N_EDGES = 500
"""Depth of the synthetic chain — 500 edges, 501 artifact nodes."""


def _build_chain(
    store: SQLiteArtifactLineageStore, artifact_store: FileArtifactStore, n_edges: int
) -> str:
    """Build a linear lineage of ``n_edges`` edges and return the leaf id.

    Each node backs onto a real ArtifactRef so ``trace_backward`` can
    hydrate every visited id.
    """
    node_ids: list[str] = []
    for i in range(n_edges + 1):
        ref = artifact_store.put_json(
            kind="workflow_ir",
            obj={"label": f"N{i}"},
            created_by="bench",
            parent_ids=[],
        )
        node_ids.append(ref.id)
    for parent, child in pairwise(node_ids):
        store.add_edge(parent_id=parent, child_id=child)
    return node_ids[-1]


def _measure(store: SQLiteArtifactLineageStore, leaf: str) -> tuple[int, float, int]:
    """Return (edge_walk_statement_count, wall_clock_seconds, n_results).

    The trace callback is installed AFTER the graph is built so only the
    traversal's statements are counted. ``get_ref`` hydration hits the
    filesystem store, not this connection, so it never appears here.
    """
    statements: list[str] = []
    store._conn.set_trace_callback(statements.append)  # bench reaches into store internals
    try:
        start = time.perf_counter()
        result = store.trace_backward(leaf)
        elapsed = time.perf_counter() - start
    finally:
        store._conn.set_trace_callback(None)
    edge_walk = [s for s in statements if "artifact_edges" in s]
    return len(edge_walk), elapsed, len(result)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        artifact_store = FileArtifactStore(root=root / "artifacts")
        store = SQLiteArtifactLineageStore(
            path=root / "events.sqlite", artifact_store=artifact_store
        )

        leaf = _build_chain(store, artifact_store, N_EDGES)
        stmt_count, elapsed, n_results = _measure(store, leaf)

    print("perf-hardening-02 :: provenance lineage edge-walk benchmark")
    print(f"  synthetic chain depth (edges)   : {N_EDGES}")
    print(f"  nodes returned by trace_backward: {n_results}")
    print(f"  edge-walk SQL statements        : {stmt_count}")
    print(f"  trace_backward wall-clock (s)   : {elapsed:.6f}")
    print(
        "  interpretation: per-node BFS ~= one statement per node "
        f"(~{N_EDGES}); recursive-CTE target == 1 (or O(depth))."
    )


if __name__ == "__main__":
    main()
