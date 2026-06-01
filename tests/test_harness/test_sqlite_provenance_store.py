"""Tests for SQLiteProvenanceStore (Phase 1 lineage graph).

Locks the contract per spec §SQLiteProvenanceStore:
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
    from molexp.harness.store.sqlite_provenance_store import SQLiteProvenanceStore

    return SQLiteProvenanceStore(path=db_path, artifact_store=artifact_store)


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
