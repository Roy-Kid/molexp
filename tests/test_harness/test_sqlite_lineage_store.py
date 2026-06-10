"""Tests for SQLiteArtifactLineageStore (Phase 1 lineage graph).

Locks the contract per spec §SQLiteArtifactLineageStore:
- add_edge(parent, child, relation) writes one row
- trace_backward(C) on A→B→C returns [B, A] (BFS up the chain)
- trace_forward(A) symmetric returns [B, C]
- lineage_graph(B) returns the whole subgraph as a dict
- backed by ArtifactStore.get_ref to materialize ArtifactRef results
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture()
def store_root(tmp_path: Path) -> Path:
    return tmp_path / "artifacts"


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "events.sqlite"


@pytest.fixture()
def artifact_store(store_root: Path):
    from molexp.harness.store.file_artifact_store import FileArtifactStore

    return FileArtifactStore(root=store_root)


@pytest.fixture()
def provenance(db_path: Path, artifact_store):
    from molexp.harness.store.sqlite_lineage_store import SQLiteArtifactLineageStore

    return SQLiteArtifactLineageStore(path=db_path, artifact_store=artifact_store)


@pytest.fixture()
def chain_abc(artifact_store, provenance):
    """A → B → C chain (`derived_from` edges)."""
    a = artifact_store.put_json(kind="user_plan", obj={"a": 1}, created_by="user", parent_ids=[])
    b = artifact_store.put_json(
        kind="experiment_report", obj={"b": 1}, created_by="harness", parent_ids=[a.id]
    )
    c = artifact_store.put_json(
        kind="workflow_ir", obj={"c": 1}, created_by="harness", parent_ids=[b.id]
    )
    provenance.add_edge(parent_id=a.id, child_id=b.id)
    provenance.add_edge(parent_id=b.id, child_id=c.id)
    return a, b, c


def test_trace_backward_on_three_layer_chain(chain_abc, provenance) -> None:
    a, b, c = chain_abc
    result = provenance.trace_backward(c.id)
    assert [r.id for r in result] == [b.id, a.id]


def test_trace_forward_on_three_layer_chain(chain_abc, provenance) -> None:
    a, b, c = chain_abc
    result = provenance.trace_forward(a.id)
    assert [r.id for r in result] == [b.id, c.id]


def test_trace_backward_terminates_at_root(chain_abc, provenance) -> None:
    a, _b, _c = chain_abc
    assert provenance.trace_backward(a.id) == []


def test_trace_forward_terminates_at_leaf(chain_abc, provenance) -> None:
    _a, _b, c = chain_abc
    assert provenance.trace_forward(c.id) == []


def test_lineage_graph_contains_full_subgraph(chain_abc, provenance) -> None:
    a, b, c = chain_abc
    graph = provenance.lineage_graph(b.id)
    nodes = {n["id"] for n in graph["nodes"]}
    edges = {(e["parent_id"], e["child_id"], e["relation"]) for e in graph["edges"]}
    assert nodes == {a.id, b.id, c.id}
    assert edges == {(a.id, b.id, "derived_from"), (b.id, c.id, "derived_from")}


def test_default_relation_is_derived_from(artifact_store, provenance) -> None:
    a = artifact_store.put_json(kind="user_plan", obj={}, created_by="user", parent_ids=[])
    b = artifact_store.put_json(
        kind="experiment_report", obj={}, created_by="harness", parent_ids=[a.id]
    )
    provenance.add_edge(parent_id=a.id, child_id=b.id)
    graph = provenance.lineage_graph(a.id)
    assert graph["edges"][0]["relation"] == "derived_from"


def test_custom_relation_preserved(artifact_store, provenance) -> None:
    a = artifact_store.put_json(kind="user_plan", obj={}, created_by="user", parent_ids=[])
    b = artifact_store.put_json(
        kind="experiment_report", obj={}, created_by="harness", parent_ids=[]
    )
    provenance.add_edge(parent_id=a.id, child_id=b.id, relation="repairs")
    graph = provenance.lineage_graph(a.id)
    assert graph["edges"][0]["relation"] == "repairs"


def test_add_edge_idempotent_on_same_triple(artifact_store, provenance) -> None:
    a = artifact_store.put_json(kind="user_plan", obj={}, created_by="user", parent_ids=[])
    b = artifact_store.put_json(
        kind="experiment_report", obj={}, created_by="harness", parent_ids=[]
    )
    provenance.add_edge(parent_id=a.id, child_id=b.id)
    provenance.add_edge(parent_id=a.id, child_id=b.id)  # second call is a no-op
    assert len(provenance.lineage_graph(a.id)["edges"]) == 1


# ───────────────────────── pipeline context on edges (stage + run_id)


def test_add_edge_records_stage_and_run_id(artifact_store, provenance) -> None:
    """An edge written by the pipeline carries the producing stage + run id."""
    a = artifact_store.put_json(kind="user_plan", obj={"s": 1}, created_by="user", parent_ids=[])
    b = artifact_store.put_json(
        kind="experiment_report", obj={"s": 2}, created_by="harness", parent_ids=[a.id]
    )
    provenance.add_edge(
        parent_id=a.id, child_id=b.id, stage="generate_experiment_report", run_id="run-1"
    )
    edge = provenance.lineage_graph(a.id)["edges"][0]
    assert edge["stage"] == "generate_experiment_report"
    assert edge["run_id"] == "run-1"


def test_add_edge_without_pipeline_context_stores_none(artifact_store, provenance) -> None:
    """stage / run_id are optional — a bare derived_from edge stays valid."""
    a = artifact_store.put_json(kind="user_plan", obj={"n": 1}, created_by="user", parent_ids=[])
    b = artifact_store.put_json(
        kind="experiment_report", obj={"n": 2}, created_by="harness", parent_ids=[]
    )
    provenance.add_edge(parent_id=a.id, child_id=b.id)
    edge = provenance.lineage_graph(a.id)["edges"][0]
    assert edge["stage"] is None
    assert edge["run_id"] is None


def test_add_edge_backfills_stage_and_run_id_on_duplicate(artifact_store, provenance) -> None:
    """Re-adding an existing edge with pipeline context fills missing fields.

    Mirrors FileArtifactStore's parent_ids merging: an idempotent re-derivation
    must not lose newly-known lineage context, and must not clobber what the
    first writer recorded.
    """
    a = artifact_store.put_json(kind="user_plan", obj={"d": 1}, created_by="user", parent_ids=[])
    b = artifact_store.put_json(
        kind="experiment_report", obj={"d": 2}, created_by="harness", parent_ids=[]
    )
    provenance.add_edge(parent_id=a.id, child_id=b.id)  # no context yet
    provenance.add_edge(parent_id=a.id, child_id=b.id, stage="report", run_id="run-9")
    edges = provenance.lineage_graph(a.id)["edges"]
    assert len(edges) == 1  # still one edge — idempotent on the triple
    assert edges[0]["stage"] == "report"
    assert edges[0]["run_id"] == "run-9"
    # First-writer-wins: a later conflicting context does not overwrite.
    provenance.add_edge(parent_id=a.id, child_id=b.id, stage="other", run_id="run-x")
    edges = provenance.lineage_graph(a.id)["edges"]
    assert edges[0]["stage"] == "report"
    assert edges[0]["run_id"] == "run-9"


def test_v1_schema_db_is_migrated_in_place(tmp_path: Path, artifact_store) -> None:
    """A pre-existing v1 harness.sqlite (no stage/run_id columns) still opens.

    ``open_db`` adds the missing ``artifact_edges`` columns; rows written by
    the old schema read back with ``stage`` / ``run_id`` as None.
    """
    import sqlite3

    from molexp.harness.store.sqlite_lineage_store import SQLiteArtifactLineageStore

    db = tmp_path / "legacy.sqlite"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE schema_version (version INTEGER PRIMARY KEY);
        INSERT INTO schema_version (version) VALUES (1);
        CREATE TABLE events (
            id TEXT PRIMARY KEY, run_id TEXT NOT NULL, seq INTEGER NOT NULL,
            type TEXT NOT NULL, actor TEXT NOT NULL, created_at TEXT NOT NULL,
            payload_json TEXT NOT NULL, artifact_ids_json TEXT NOT NULL
        );
        CREATE TABLE artifact_edges (
            parent_id TEXT NOT NULL, child_id TEXT NOT NULL,
            relation TEXT NOT NULL, created_at TEXT NOT NULL,
            PRIMARY KEY (parent_id, child_id, relation)
        );
        INSERT INTO artifact_edges VALUES ('p1', 'c1', 'derived_from', '2026-01-01T00:00:00');
        """
    )
    conn.commit()
    conn.close()

    store = SQLiteArtifactLineageStore(path=db, artifact_store=artifact_store)
    a = artifact_store.put_json(kind="user_plan", obj={"m": 1}, created_by="user", parent_ids=[])
    b = artifact_store.put_json(
        kind="experiment_report", obj={"m": 2}, created_by="harness", parent_ids=[]
    )
    store.add_edge(parent_id=a.id, child_id=b.id, stage="report", run_id="run-2")

    graph = store.lineage_graph("p1")
    legacy_edge = next(e for e in graph["edges"] if e["parent_id"] == "p1")
    assert legacy_edge["stage"] is None and legacy_edge["run_id"] is None
    new_edge = store.lineage_graph(a.id)["edges"][0]
    assert new_edge["stage"] == "report" and new_edge["run_id"] == "run-2"
