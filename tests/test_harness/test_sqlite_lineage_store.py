"""Tests for ``SQLiteArtifactLineageStore`` (pipeline-artifact lineage graph).

Locks the contract per spec §SQLiteArtifactLineageStore:
- ``trace_backward`` terminates (empty) at a root artifact
- ``lineage_graph(mid)`` returns the whole ancestor+descendant subgraph
- ``add_edge`` is idempotent on ``(parent_id, child_id, relation)``
- an edge records its pipeline context (``stage`` + ``run_id``), backfilling
  missing fields on a duplicate write but never clobbering the first writer
- a legacy v1 DB (no ``stage`` / ``run_id`` columns) is migrated in place
"""

from __future__ import annotations

from pathlib import Path

import pytest

from molexp.harness.store.file_artifact_store import FileArtifactStore
from molexp.harness.store.sqlite_lineage_store import SQLiteArtifactLineageStore


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "events.sqlite"


@pytest.fixture()
def artifact_store(tmp_path: Path) -> FileArtifactStore:
    return FileArtifactStore(root=tmp_path / "artifacts")


@pytest.fixture()
def provenance(db_path: Path, artifact_store: FileArtifactStore) -> SQLiteArtifactLineageStore:
    return SQLiteArtifactLineageStore(path=db_path, artifact_store=artifact_store)


@pytest.fixture()
def chain_abc(artifact_store, provenance):
    """A → B → C chain (``derived_from`` edges)."""
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


class TestSQLiteArtifactLineageStore:
    def test_trace_backward_terminates_at_root(self, chain_abc, provenance) -> None:
        a, _b, _c = chain_abc
        assert provenance.trace_backward(a.id) == []

    def test_lineage_graph_contains_full_subgraph(self, chain_abc, provenance) -> None:
        a, b, c = chain_abc
        graph = provenance.lineage_graph(b.id)
        nodes = {n["id"] for n in graph["nodes"]}
        edges = {(e["parent_id"], e["child_id"], e["relation"]) for e in graph["edges"]}
        assert nodes == {a.id, b.id, c.id}
        assert edges == {(a.id, b.id, "derived_from"), (b.id, c.id, "derived_from")}

    def test_add_edge_idempotent_on_same_triple(self, artifact_store, provenance) -> None:
        a = artifact_store.put_json(kind="user_plan", obj={}, created_by="user", parent_ids=[])
        b = artifact_store.put_json(
            kind="experiment_report", obj={}, created_by="harness", parent_ids=[]
        )
        provenance.add_edge(parent_id=a.id, child_id=b.id)
        provenance.add_edge(parent_id=a.id, child_id=b.id)  # second call is a no-op
        assert len(provenance.lineage_graph(a.id)["edges"]) == 1

    def test_add_edge_records_stage_and_run_id(self, artifact_store, provenance) -> None:
        """A fresh edge written by the pipeline carries the producing stage + run id."""
        a = artifact_store.put_json(
            kind="user_plan", obj={"s": 1}, created_by="user", parent_ids=[]
        )
        b = artifact_store.put_json(
            kind="experiment_report", obj={"s": 2}, created_by="harness", parent_ids=[a.id]
        )
        provenance.add_edge(
            parent_id=a.id, child_id=b.id, stage="generate_experiment_report", run_id="run-1"
        )
        edge = provenance.lineage_graph(a.id)["edges"][0]
        assert edge["stage"] == "generate_experiment_report"
        assert edge["run_id"] == "run-1"

    def test_add_edge_backfills_stage_and_run_id_on_duplicate(
        self, artifact_store, provenance
    ) -> None:
        """A duplicate re-derivation fills missing lineage context via COALESCE, then
        first-writer-wins protects it against a later conflicting context."""
        a = artifact_store.put_json(
            kind="user_plan", obj={"d": 1}, created_by="user", parent_ids=[]
        )
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

    def test_v1_schema_db_is_migrated_in_place(self, tmp_path: Path, artifact_store) -> None:
        """A pre-existing v1 harness.sqlite (no stage/run_id columns) still opens.

        ``open_db`` adds the missing ``artifact_edges`` columns; rows written by
        the old schema read back with ``stage`` / ``run_id`` as None.
        """
        import sqlite3

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
        a = artifact_store.put_json(
            kind="user_plan", obj={"m": 1}, created_by="user", parent_ids=[]
        )
        b = artifact_store.put_json(
            kind="experiment_report", obj={"m": 2}, created_by="harness", parent_ids=[]
        )
        store.add_edge(parent_id=a.id, child_id=b.id, stage="report", run_id="run-2")

        graph = store.lineage_graph("p1")
        legacy_edge = next(e for e in graph["edges"] if e["parent_id"] == "p1")
        assert legacy_edge["stage"] is None and legacy_edge["run_id"] is None
        new_edge = store.lineage_graph(a.id)["edges"][0]
        assert new_edge["stage"] == "report" and new_edge["run_id"] == "run-2"
