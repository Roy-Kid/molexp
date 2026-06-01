"""Tests for SQLiteEventLog (Phase 1 audit-timeline persistence).

Locks the contract per spec §SQLiteEventLog:
- append() returns a HarnessEvent with monotonic per-run_id seq starting at 1
- list_events(run_id) returns events in seq order, isolated per run_id
- duplicate (run_id, seq) raises EventSeqConflictError chaining sqlite3.IntegrityError
- get_timeline is an alias of list_events
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "events.sqlite"


@pytest.fixture()
def log(db_path: Path):
    from molexp.harness.store.sqlite_event_log import SQLiteEventLog

    return SQLiteEventLog(path=db_path)


def test_append_assigns_seq_starting_at_one(log) -> None:
    e = log.append(
        run_id="run-A",
        type="run_created",
        actor="harness",
        payload={"foo": "bar"},
    )
    assert e.seq == 1
    assert e.run_id == "run-A"
    assert e.type == "run_created"


def test_append_seq_is_monotonic_per_run_id(log) -> None:
    seqs = [log.append(run_id="run-A", type="stage_started", actor="harness").seq for _ in range(5)]
    assert seqs == [1, 2, 3, 4, 5]


def test_seq_is_isolated_across_run_ids(log) -> None:
    log.append(run_id="run-A", type="run_created", actor="harness")
    log.append(run_id="run-A", type="stage_started", actor="harness")
    b1 = log.append(run_id="run-B", type="run_created", actor="harness")
    a3 = log.append(run_id="run-A", type="stage_completed", actor="harness")
    b2 = log.append(run_id="run-B", type="stage_started", actor="harness")
    assert b1.seq == 1
    assert b2.seq == 2
    assert a3.seq == 3


def test_list_events_returns_seq_order(log) -> None:
    log.append(run_id="run-A", type="run_created", actor="harness")
    log.append(run_id="run-A", type="stage_started", actor="harness")
    log.append(run_id="run-A", type="stage_completed", actor="harness")
    events = log.list_events("run-A")
    assert [e.seq for e in events] == [1, 2, 3]
    assert [e.type for e in events] == ["run_created", "stage_started", "stage_completed"]


def test_list_events_isolated_per_run_id(log) -> None:
    log.append(run_id="run-A", type="run_created", actor="harness")
    log.append(run_id="run-B", type="run_created", actor="harness")
    a_events = log.list_events("run-A")
    b_events = log.list_events("run-B")
    assert len(a_events) == 1 and a_events[0].run_id == "run-A"
    assert len(b_events) == 1 and b_events[0].run_id == "run-B"


def test_get_timeline_is_list_events_alias(log) -> None:
    log.append(run_id="run-A", type="run_created", actor="harness")
    log.append(run_id="run-A", type="stage_started", actor="harness")
    assert log.get_timeline("run-A") == log.list_events("run-A")


def test_duplicate_seq_raises_event_seq_conflict_error(log, db_path: Path) -> None:
    """Manually insert a duplicate (run_id, seq) row via raw SQL to confirm
    the UNIQUE constraint is enforced and mapped to EventSeqConflictError.
    """
    import sqlite3
    from datetime import UTC, datetime

    from molexp.harness.errors import EventSeqConflictError

    log.append(run_id="run-A", type="run_created", actor="harness")
    # Direct insert with the same (run_id, seq=1) must fail.
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys=ON")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO events (id, run_id, seq, type, actor, created_at, "
                "payload_json, artifact_ids_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "manual",
                    "run-A",
                    1,  # duplicate seq for run-A
                    "run_created",
                    "test",
                    datetime.now(tz=UTC).isoformat(),
                    "{}",
                    "[]",
                ),
            )

    # The harness must also map IntegrityError → EventSeqConflictError when
    # the conflict comes through the SQLiteEventLog public API.
    with pytest.raises(EventSeqConflictError) as exc_info:
        # Force a conflict by appending with an explicit seq that collides.
        # If the API does not accept an explicit seq, force one via raw SQL
        # under the assumption that the wrapper class exposes a typed mapper.
        log._append_with_explicit_seq(  # type: ignore[attr-defined]
            run_id="run-A",
            seq=1,
            type="run_created",
            actor="harness",
        )
    assert isinstance(exc_info.value.__cause__, sqlite3.IntegrityError)


def test_payload_and_artifact_ids_round_trip(log) -> None:
    e = log.append(
        run_id="run-A",
        type="artifact_created",
        actor="harness",
        payload={"k": [1, 2, {"nested": True}]},
        artifact_ids=["a", "b"],
    )
    events = log.list_events("run-A")
    assert events == [e]
    assert events[0].payload == {"k": [1, 2, {"nested": True}]}
    assert events[0].artifact_ids == ["a", "b"]
