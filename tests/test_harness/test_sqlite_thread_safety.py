"""Thread-safety tests for the shared-lock SQLite stores (perf-hardening-01).

Maps to ac-001 / ac-002 (code: ``check_same_thread=False`` + shared lock
identity) and ac-005 (runtime: concurrent ``asyncio.to_thread`` ``append`` +
``add_edge`` raise no same-thread ``ProgrammingError`` and corrupt no
``UNIQUE(run_id, seq)`` index).

Design A (per spec): ``open_db`` opens the connection with
``check_same_thread=False`` and returns ``(conn, lock)`` where ``lock`` is a
``threading.Lock`` shared across every store that opens the *same* resolved
DB-file path. ``SQLiteEventLog`` and ``SQLiteArtifactLineageStore`` constructed on
the same path therefore serialize through one identical lock object.

They guard against two regressions: ``open_db`` dropping the lock tuple (so
stores would lose their ``_lock``), and the connection reverting to the
default ``check_same_thread=True`` — under which a worker thread spawned by
``asyncio.to_thread`` raises ``sqlite3.ProgrammingError`` the moment it
touches the connection.
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


# ── ac-005: concurrent append from to_thread workers ──────────────────────


@pytest.mark.asyncio
async def test_concurrent_appends_from_threads_no_corruption(db_path: Path) -> None:
    """50 concurrent ``append`` calls dispatched via ``asyncio.to_thread``.

    Asserts no ``sqlite3.ProgrammingError`` ("SQLite objects created in a
    thread can only be used in that same thread") and that the resulting
    ``list_events`` is a gap-free unique ``seq`` run 1..50.

    RED today: ``check_same_thread=True`` raises ``ProgrammingError`` from
    the worker threads, which ``asyncio.gather`` re-raises here.
    """
    from molexp.harness.store.sqlite_event_log import SQLiteEventLog

    elog = SQLiteEventLog(path=db_path)
    n = 50

    await asyncio.gather(
        *[
            asyncio.to_thread(
                elog.append,
                run_id="r",
                type="stage_started",
                actor="t",
                payload={"i": i},
            )
            for i in range(n)
        ]
    )

    events = elog.list_events("r")
    assert len(events) == n
    seqs = sorted(e.seq for e in events)
    assert seqs == list(range(1, n + 1)), "seq sequence is not gap-free / unique 1..n"


@pytest.mark.asyncio
async def test_concurrent_append_and_add_edge_same_path_no_error(
    db_path: Path,
    artifact_store,
) -> None:
    """Mixed concurrent ``append`` + ``add_edge`` on same-path stores.

    Both stores share one connection/lock; interleaving writes from many
    ``to_thread`` workers must not raise and must leave consistent final
    state (all events present, all edges present).

    RED today: same-thread ``ProgrammingError`` from worker threads.
    """
    from molexp.harness.store.sqlite_event_log import SQLiteEventLog
    from molexp.harness.store.sqlite_lineage_store import SQLiteArtifactLineageStore

    elog = SQLiteEventLog(path=db_path)
    pstore = SQLiteArtifactLineageStore(path=db_path, artifact_store=artifact_store)

    n = 25
    # Pre-create artifacts so add_edge has real ids to reference (parent_id
    # / child_id are opaque to the edges table — no FK to artifacts).
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

    await asyncio.gather(*append_tasks, *edge_tasks)

    # Events: gap-free unique 1..n.
    events = elog.list_events("r")
    assert sorted(e.seq for e in events) == list(range(1, n + 1))

    # Edges: each parent traces forward to its child.
    for i in range(n):
        descendants = pstore.trace_forward(parents[i].id)
        assert children[i].id in {d.id for d in descendants}
