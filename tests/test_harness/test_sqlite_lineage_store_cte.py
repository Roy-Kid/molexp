"""Traversal-contract tests for :class:`SQLiteArtifactLineageStore`.

The lineage store walks ``artifact_edges`` with a single ``WITH RECURSIVE``
CTE. These tests own the traversal semantics — BFS level order, shallowest-depth
dedup, cycle termination, the ``lineage_graph`` node/edge shape — plus the one
performance-regression guard that the walk stays a single statement rather than
one ``SELECT`` per node (the O(nodes) round-trip bug perf-hardening-02 fixed).

White-box convention: the query-count guard reaches into ``store._conn`` exactly
as the sibling stage-bracket tests reach into store internals.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import pytest

from molexp.harness.store.file_artifact_store import FileArtifactStore
from molexp.harness.store.sqlite_lineage_store import SQLiteArtifactLineageStore


@pytest.fixture()
def artifact_store(tmp_path: Path) -> FileArtifactStore:
    return FileArtifactStore(root=tmp_path / "artifacts")


@pytest.fixture()
def store(tmp_path: Path, artifact_store: FileArtifactStore) -> SQLiteArtifactLineageStore:
    return SQLiteArtifactLineageStore(
        path=tmp_path / "events.sqlite", artifact_store=artifact_store
    )


def _make_node(artifact_store: FileArtifactStore, label: str) -> str:
    """Create a distinct real artifact for ``label`` and return its id.

    Traversals hydrate every visited id through ``get_ref``, so each graph node
    must back onto a real ``PlanArtifactRef``; the label keeps content-addressed
    ids distinct.
    """
    ref = artifact_store.put_json(
        kind="workflow_ir",
        obj={"label": label},
        created_by="test",
        parent_ids=[],
    )
    return ref.id


class TestSQLiteArtifactLineageStore:
    def test_trace_backward_returns_ancestors_in_level_order(
        self, store: SQLiteArtifactLineageStore, artifact_store: FileArtifactStore
    ) -> None:
        a = _make_node(artifact_store, "A")
        b = _make_node(artifact_store, "B")
        c = _make_node(artifact_store, "C")
        d = _make_node(artifact_store, "D")
        store.add_edge(parent_id=a, child_id=b)
        store.add_edge(parent_id=b, child_id=c)
        store.add_edge(parent_id=c, child_id=d)

        assert [r.id for r in store.trace_backward(d)] == [c, b, a]

    def test_trace_forward_returns_descendants_in_level_order(
        self, store: SQLiteArtifactLineageStore, artifact_store: FileArtifactStore
    ) -> None:
        a = _make_node(artifact_store, "A")
        b = _make_node(artifact_store, "B")
        c = _make_node(artifact_store, "C")
        d = _make_node(artifact_store, "D")
        store.add_edge(parent_id=a, child_id=b)
        store.add_edge(parent_id=b, child_id=c)
        store.add_edge(parent_id=c, child_id=d)

        assert [r.id for r in store.trace_forward(a)] == [b, c, d]

    def test_trace_backward_dedups_shared_ancestor_at_shallowest_depth(
        self, store: SQLiteArtifactLineageStore, artifact_store: FileArtifactStore
    ) -> None:
        a = _make_node(artifact_store, "A")
        b = _make_node(artifact_store, "B")
        c = _make_node(artifact_store, "C")
        d = _make_node(artifact_store, "D")
        # Diamond: A->B, A->C, B->D, C->D. From D upward: depth1 = {B, C}, depth2 = {A}.
        store.add_edge(parent_id=a, child_id=b)
        store.add_edge(parent_id=a, child_id=c)
        store.add_edge(parent_id=b, child_id=d)
        store.add_edge(parent_id=c, child_id=d)

        result_ids = [r.id for r in store.trace_backward(d)]

        # Shared ancestor A appears exactly once; depth-1 frontier precedes it.
        assert len(result_ids) == 3
        assert result_ids.count(a) == 1
        assert set(result_ids[:2]) == {b, c}
        assert result_ids[2] == a

    def test_traversal_terminates_on_cycle_without_duplicates(
        self, store: SQLiteArtifactLineageStore, artifact_store: FileArtifactStore
    ) -> None:
        a = _make_node(artifact_store, "A")
        b = _make_node(artifact_store, "B")
        # Malformed cycle: A->B and B->A.
        store.add_edge(parent_id=a, child_id=b)
        store.add_edge(parent_id=b, child_id=a)

        backward = [r.id for r in store.trace_backward(a)]
        forward = [r.id for r in store.trace_forward(a)]

        # Reachable set from A (exclusive of A itself) is just {B}, once.
        assert backward == [b]
        assert forward == [b]

    def test_lineage_graph_returns_subgraph_nodes_and_stampless_edges(
        self, store: SQLiteArtifactLineageStore, artifact_store: FileArtifactStore
    ) -> None:
        a = _make_node(artifact_store, "A")
        b = _make_node(artifact_store, "B")
        c = _make_node(artifact_store, "C")
        d = _make_node(artifact_store, "D")
        store.add_edge(parent_id=a, child_id=b)
        store.add_edge(parent_id=a, child_id=c)
        store.add_edge(parent_id=b, child_id=d)
        store.add_edge(parent_id=c, child_id=d)

        graph = store.lineage_graph(d)

        # Nodes: sorted by id, each {id, kind, uri}.
        expected_nodes = [
            {
                "id": aid,
                "kind": artifact_store.get_ref(aid).kind,
                "uri": artifact_store.get_ref(aid).uri,
            }
            for aid in sorted([a, b, c, d])
        ]
        assert graph["nodes"] == expected_nodes
        # Edges written without pipeline context carry stage/run_id as None.
        assert {(e["parent_id"], e["child_id"], e["relation"]) for e in graph["edges"]} == {
            (a, b, "derived_from"),
            (a, c, "derived_from"),
            (b, d, "derived_from"),
            (c, d, "derived_from"),
        }
        assert all(
            set(e.keys()) == {"parent_id", "child_id", "relation", "stage", "run_id"}
            and e["stage"] is None
            and e["run_id"] is None
            for e in graph["edges"]
        )

    def test_trace_backward_walks_edges_in_a_single_recursive_cte(
        self, store: SQLiteArtifactLineageStore, artifact_store: FileArtifactStore
    ) -> None:
        """The edge walk emits one statement, not one per node (perf-hardening-02)."""
        chain_depth = 50
        node_ids = [_make_node(artifact_store, f"N{i}") for i in range(chain_depth + 1)]
        for parent, child in pairwise(node_ids):
            store.add_edge(parent_id=parent, child_id=child)

        # ``add_edge`` setup ran before the callback is installed, so those
        # statements are not counted; ``get_ref`` hydration hits the filesystem
        # store (a different connection), so it cannot appear here either.
        statements: list[str] = []
        store._conn.set_trace_callback(statements.append)
        try:
            result = store.trace_backward(node_ids[-1])
        finally:
            store._conn.set_trace_callback(None)

        edge_walk_statements = [s for s in statements if "artifact_edges" in s]
        assert len(result) == chain_depth  # correctness: all ancestors returned
        assert len(edge_walk_statements) == 1
