"""Layer-0 append-only SQLite seq-log primitive (workspace-event-01-sqlitelog, P0.3).

RED-first: ``molexp.sqlitelog`` does not exist yet.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from molexp.sqlitelog import SeqConflictError, SeqEventStore, open_wal_connection


def _store(tmp_path: Path) -> SeqEventStore:
    conn, lock = open_wal_connection(tmp_path / "db.sqlite")
    store = SeqEventStore(
        conn, lock, table="events", scope_column="run_id", refs_column="refs_json"
    )
    store.ensure_schema()
    return store


def _append(store: SeqEventStore, scope_id: str, i: int, seq: int | None = None) -> int:
    return store.append(
        event_id=f"e-{scope_id}-{i}",
        scope_id=scope_id,
        type="x",
        actor="test",
        created_at_iso="2026-07-01T00:00:00+00:00",
        payload_json="{}",
        refs_json="[]",
        seq=seq,
    )


def test_open_wal_connection(tmp_path: Path) -> None:
    conn, lock = open_wal_connection(tmp_path / "db.sqlite")
    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    _conn2, lock2 = open_wal_connection(tmp_path / "db.sqlite")
    assert lock2 is lock  # same resolved path → same lock instance


def test_append_and_list_monotonic_seq(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert [_append(store, "r1", i) for i in range(3)] == [1, 2, 3]
    assert _append(store, "r2", 0) == 1  # independent seq per scope
    rows = store.list_rows("r1")
    assert [row[2] for row in rows] == [1, 2, 3]  # seq column, ordered
    assert store.list_rows("r1")[0][0] == "e-r1-0"  # id column
    assert len(store.list_rows("r2")) == 1


def test_duplicate_seq_conflict(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _append(store, "r1", 0, seq=5)
    with pytest.raises(SeqConflictError):
        _append(store, "r1", 1, seq=5)
