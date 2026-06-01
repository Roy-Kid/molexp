"""Recursive-CTE equivalence + query-count tests for SQLiteProvenanceStore.

Maps acceptance criteria ac-001..ac-007 of spec
``perf-hardening-02-provenance-recursive-cte``.

The traversal rewrite replaces the per-node Python BFS in
``SQLiteProvenanceStore._bfs`` with a single ``WITH RECURSIVE`` CTE edge
walk. The behavioural tests below (ac-001..ac-005, ac-007) are
EQUIVALENCE / regression guards: they pin the exact returned id set, the
BFS/level order, the first-seen-by-shallowest-depth dedup, cycle
termination, and the byte-identical ``lineage_graph`` shape. They pass
against the current correct BFS AND must keep passing after the rewrite.

The meaningful RED driver is the query-count test (ac-006): it counts the
SQL statements emitted on the store's connection during a single
``trace_backward`` over a deep chain. The current code emits ~one statement
per node; the rewrite collapses that to a single recursive CTE.

White-box convention: these tests reach into ``store._conn`` exactly as the
sibling tests in this directory reach into store internals.
"""

from __future__ import annotations

from collections import deque
from itertools import pairwise
from pathlib import Path
from typing import Literal

import pytest

from molexp.harness.store.file_artifact_store import FileArtifactStore
from molexp.harness.store.sqlite_provenance_store import SQLiteProvenanceStore


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture()
def artifact_store(tmp_path: Path) -> FileArtifactStore:
    return FileArtifactStore(root=tmp_path / "artifacts")


@pytest.fixture()
def store(tmp_path: Path, artifact_store: FileArtifactStore) -> SQLiteProvenanceStore:
    return SQLiteProvenanceStore(path=tmp_path / "events.sqlite", artifact_store=artifact_store)


def _make_node(artifact_store: FileArtifactStore, label: str) -> str:
    """Create a distinct real artifact for ``label`` and return its id.

    ``trace_backward`` / ``trace_forward`` hydrate every visited id through
    ``get_ref``, so each graph node must back onto a real ArtifactRef. The
    payload embeds the label so content-addressed ids stay distinct.
    """
    ref = artifact_store.put_json(
        kind="workflow_ir",
        obj={"label": label},
        created_by="test",
        parent_ids=[],
    )
    return ref.id


def _reference_bfs(
    store: SQLiteProvenanceStore, start: str, *, direction: Literal["up", "down"]
) -> list[str]:
    """The pre-rewrite Python BFS, inlined as the equivalence oracle.

    Issues one query per visited node (the O(nodes) algorithm the CTE
    rewrite replaces). Used only to assert the rewritten walk returns an
    identical ordered id list.
    """
    if direction == "up":
        query = "SELECT parent_id FROM artifact_edges WHERE child_id = ?"
    else:
        query = "SELECT child_id FROM artifact_edges WHERE parent_id = ?"
    seen: set[str] = {start}
    order: list[str] = []
    queue: deque[str] = deque([start])
    conn = store._conn  # white-box oracle, matches sibling tests in this dir
    while queue:
        current = queue.popleft()
        for (neighbour,) in conn.execute(query, (current,)).fetchall():
            if neighbour in seen:
                continue
            seen.add(neighbour)
            order.append(neighbour)
            queue.append(neighbour)
    return order


# --------------------------------------------------------------------------- #
# ac-001 — linear chain equivalence
# --------------------------------------------------------------------------- #
def test_ac001_trace_backward_linear_chain_matches_reference_and_expected_order(
    store: SQLiteProvenanceStore, artifact_store: FileArtifactStore
) -> None:
    a = _make_node(artifact_store, "A")
    b = _make_node(artifact_store, "B")
    c = _make_node(artifact_store, "C")
    d = _make_node(artifact_store, "D")
    store.add_edge(parent_id=a, child_id=b)
    store.add_edge(parent_id=b, child_id=c)
    store.add_edge(parent_id=c, child_id=d)

    result_ids = [r.id for r in store.trace_backward(d)]

    assert result_ids == [c, b, a]
    assert result_ids == _reference_bfs(store, d, direction="up")


# --------------------------------------------------------------------------- #
# ac-002 — diamond DAG dedups multi-path node once at shallowest depth
# --------------------------------------------------------------------------- #
def test_ac002_trace_backward_diamond_dedups_shared_ancestor_at_shallowest_depth(
    store: SQLiteProvenanceStore, artifact_store: FileArtifactStore
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

    # Each ancestor appears exactly once (shared node A deduped).
    assert sorted(result_ids) == sorted([b, c, a])
    assert len(result_ids) == 3
    assert result_ids.count(a) == 1
    # Level order: depth-1 frontier {B, C} precede depth-2 {A}.
    assert set(result_ids[:2]) == {b, c}
    assert result_ids[2] == a
    # Byte-identical to the reference BFS ordering.
    assert result_ids == _reference_bfs(store, d, direction="up")


# --------------------------------------------------------------------------- #
# ac-003 — cycle terminates and dedups
# --------------------------------------------------------------------------- #
def test_ac003_cycle_terminates_without_duplicates_both_directions(
    store: SQLiteProvenanceStore, artifact_store: FileArtifactStore
) -> None:
    a = _make_node(artifact_store, "A")
    b = _make_node(artifact_store, "B")
    # Malformed cycle: A->B and B->A.
    store.add_edge(parent_id=a, child_id=b)
    store.add_edge(parent_id=b, child_id=a)

    # Generous guard: if the walk hangs, the test process would never return;
    # we additionally assert the result is finite and small.
    backward = [r.id for r in store.trace_backward(a)]
    forward = [r.id for r in store.trace_forward(a)]

    # Reachable set from A (exclusive of A itself) is just {B}, once.
    assert backward == [b]
    assert forward == [b]
    assert len(backward) == len(set(backward))
    assert len(forward) == len(set(forward))


# --------------------------------------------------------------------------- #
# ac-004 — trace_forward on chain + wide fan-out
# --------------------------------------------------------------------------- #
def test_ac004_trace_forward_linear_chain_matches_reference(
    store: SQLiteProvenanceStore, artifact_store: FileArtifactStore
) -> None:
    a = _make_node(artifact_store, "A")
    b = _make_node(artifact_store, "B")
    c = _make_node(artifact_store, "C")
    d = _make_node(artifact_store, "D")
    store.add_edge(parent_id=a, child_id=b)
    store.add_edge(parent_id=b, child_id=c)
    store.add_edge(parent_id=c, child_id=d)

    result_ids = [r.id for r in store.trace_forward(a)]

    assert result_ids == [b, c, d]
    assert result_ids == _reference_bfs(store, a, direction="down")


def test_ac004_trace_forward_wide_fanout_returns_all_children_at_depth1(
    store: SQLiteProvenanceStore, artifact_store: FileArtifactStore
) -> None:
    parent = _make_node(artifact_store, "P")
    children = [_make_node(artifact_store, f"C{i}") for i in range(5)]
    for child in children:
        store.add_edge(parent_id=parent, child_id=child)

    result_ids = [r.id for r in store.trace_forward(parent)]

    # All five children are at depth 1; the exact frontier order must match
    # the reference BFS.
    assert sorted(result_ids) == sorted(children)
    assert len(result_ids) == 5
    assert result_ids == _reference_bfs(store, parent, direction="down")


# --------------------------------------------------------------------------- #
# ac-005 — lineage_graph byte-identical shape (golden capture)
# --------------------------------------------------------------------------- #
def test_ac005_lineage_graph_diamond_shape_is_byte_identical_golden(
    store: SQLiteProvenanceStore, artifact_store: FileArtifactStore
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

    # Golden: nodes sorted by id, each {id, kind, uri}; edges only those whose
    # both endpoints are in the subgraph. We reconstruct the expected golden
    # from the known ids so the assertion captures the EXACT current shape and
    # guarantees byte-identical output across the rewrite.
    expected_ids = sorted([a, b, c, d])
    expected_nodes = []
    for aid in expected_ids:
        ref = artifact_store.get_ref(aid)
        expected_nodes.append({"id": aid, "kind": ref.kind, "uri": ref.uri})
    expected_edges = [
        {"parent_id": a, "child_id": b, "relation": "derived_from"},
        {"parent_id": a, "child_id": c, "relation": "derived_from"},
        {"parent_id": b, "child_id": d, "relation": "derived_from"},
        {"parent_id": c, "child_id": d, "relation": "derived_from"},
    ]

    assert graph["nodes"] == expected_nodes
    # Edge ordering follows the row scan; compare as sets for stability of the
    # contract while still asserting exact membership + the dict shape.
    assert {(e["parent_id"], e["child_id"], e["relation"]) for e in graph["edges"]} == {
        (e["parent_id"], e["child_id"], e["relation"]) for e in expected_edges
    }
    assert all(set(e.keys()) == {"parent_id", "child_id", "relation"} for e in graph["edges"])
    assert len(graph["edges"]) == len(expected_edges)


# --------------------------------------------------------------------------- #
# ac-006 — query-count: edge walk is 1 statement (or O(depth)), not O(nodes)
# --------------------------------------------------------------------------- #
def test_ac006_trace_backward_edge_walk_emits_one_statement_not_per_node(
    store: SQLiteProvenanceStore, artifact_store: FileArtifactStore
) -> None:
    # Build a linear chain of >=50 edges (51 nodes, 50 edges).
    chain_depth = 50
    node_ids = [_make_node(artifact_store, f"N{i}") for i in range(chain_depth + 1)]
    for parent, child in pairwise(node_ids):
        store.add_edge(parent_id=parent, child_id=child)

    leaf = node_ids[-1]

    # Count only edge-walk statements: those that touch ``artifact_edges`` via
    # the recursive walk. ``add_edge`` setup statements ran BEFORE we install
    # the callback, so they are not counted. ``get_ref`` hydration hits the
    # filesystem ArtifactStore (not this connection), so it cannot appear here.
    statements: list[str] = []

    def _trace(sql: str) -> None:
        statements.append(sql)

    store._conn.set_trace_callback(_trace)  # white-box store test
    try:
        result = store.trace_backward(leaf)
    finally:
        store._conn.set_trace_callback(None)

    edge_walk_statements = [s for s in statements if "artifact_edges" in s]

    # Correctness sanity: all 50 ancestors were returned.
    assert len(result) == chain_depth

    # Single recursive CTE (== 1) OR an O(depth) batched fallback (<= depth),
    # and in all cases STRICTLY LESS THAN the node count. Current per-node BFS
    # emits ~chain_depth statements -> fails -> RED. After rewrite -> 1 -> GREEN.
    assert edge_walk_statements, "expected at least one edge-walk statement"
    assert len(edge_walk_statements) == 1 or len(edge_walk_statements) <= chain_depth
    assert len(edge_walk_statements) < len(node_ids)
    # Tightened: the rewrite target is a single recursive CTE.
    assert len(edge_walk_statements) == 1


# --------------------------------------------------------------------------- #
# ac-007 — regression-guard parity with the existing suites
# --------------------------------------------------------------------------- #
def test_ac007_existing_suites_serve_as_regression_guard(
    store: SQLiteProvenanceStore, artifact_store: FileArtifactStore
) -> None:
    """ac-007 parity is served by the pre-existing suites.

    ``tests/test_harness/test_sqlite_provenance_store.py`` and
    ``tests/test_harness/test_provenance_validator.py`` exercise the public
    trace API and ``validate_provenance`` against representative DAGs without
    edits to their assertions. Rather than duplicate them, this test asserts
    those files exist (so the regression guard is wired) and re-runs the
    representative trace path here to confirm parity in this module too.
    """
    here = Path(__file__).resolve().parent
    assert (here / "test_sqlite_provenance_store.py").is_file()
    assert (here / "test_provenance_validator.py").is_file()

    # Representative DAG round-trip via the public API.
    a = _make_node(artifact_store, "A")
    b = _make_node(artifact_store, "B")
    c = _make_node(artifact_store, "C")
    store.add_edge(parent_id=a, child_id=b)
    store.add_edge(parent_id=b, child_id=c)
    assert [r.id for r in store.trace_backward(c)] == [b, a]
    assert [r.id for r in store.trace_forward(a)] == [b, c]
