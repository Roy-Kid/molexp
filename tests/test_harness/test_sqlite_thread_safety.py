"""Concurrency-integration guard for the harness's shared-path SQLite stores.

``SQLiteEventLog`` and ``SQLiteArtifactLineageStore`` opened on the *same* DB
path share one connection + lock (via ``store._sqlite.open_db``), and
``StageRunner`` drives ``append`` / ``add_edge`` from ``asyncio.to_thread``
workers. This guards the harness composition: concurrent worker-thread writes
across both stores raise no same-thread ``sqlite3.ProgrammingError`` and corrupt
neither the ``seq`` run nor the edge set.

The Layer-0 primitive (``check_same_thread=False`` + shared-lock identity +
monotonic ``seq``) is owned by ``tests/test_sqlitelog.py``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "events.sqlite"


@pytest.fixture()
def artifact_store(tmp_path: Path):
    from molexp.harness.store.file_artifact_store import FileArtifactStore

    return FileArtifactStore(root=tmp_path / "artifacts")


@pytest.mark.asyncio
async def test_concurrent_append_and_add_edge_on_shared_path_stay_consistent(
    db_path: Path,
    artifact_store,
) -> None:
    from molexp.harness.store.sqlite_event_log import SQLiteEventLog
    from molexp.harness.store.sqlite_lineage_store import SQLiteArtifactLineageStore

    elog = SQLiteEventLog(path=db_path)
    pstore = SQLiteArtifactLineageStore(path=db_path, artifact_store=artifact_store)

    n = 25
    # Pre-create artifacts so add_edge has real ids to reference.
    parents = [
        artifact_store.put_json(kind="user_plan", obj={"p": i}, created_by="t", parent_ids=[])
        for i in range(n)
    ]
    children = [
        artifact_store.put_json(
            kind="experiment_report", obj={"c": i}, created_by="t", parent_ids=[parents[i].id]
        )
        for i in range(n)
    ]

    append_tasks = [
        asyncio.to_thread(
            elog.append, run_id="r", type="stage_started", actor="t", payload={"i": i}
        )
        for i in range(n)
    ]
    edge_tasks = [
        asyncio.to_thread(pstore.add_edge, parent_id=parents[i].id, child_id=children[i].id)
        for i in range(n)
    ]

    # No same-thread ProgrammingError is raised through asyncio.gather.
    await asyncio.gather(*append_tasks, *edge_tasks)

    # Events: gap-free unique seq 1..n.
    events = elog.list_events("r")
    assert sorted(e.seq for e in events) == list(range(1, n + 1))
    # Edges: each parent traces forward to its child.
    for i in range(n):
        descendants = pstore.trace_forward(parents[i].id)
        assert children[i].id in {d.id for d in descendants}
