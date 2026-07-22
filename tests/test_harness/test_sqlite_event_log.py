"""Tests for ``SQLiteEventLog`` (audit-timeline persistence).

Locks the contract per spec §SQLiteEventLog:
- ``append()`` assigns a monotonic per-``run_id`` ``seq`` starting at 1, isolated across runs
- ``list_events(run_id)`` returns events in ``seq`` order, isolated per ``run_id``
- ``get_timeline`` is an alias of ``list_events``
- a duplicate ``(run_id, seq)`` maps sqlite3.IntegrityError → EventSeqConflictError
- ``payload`` / ``artifact_ids`` survive the persist → read round-trip
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from molexp.harness.errors import EventSeqConflictError
from molexp.harness.store.sqlite_event_log import SQLiteEventLog


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "events.sqlite"


@pytest.fixture()
def log(db_path: Path) -> SQLiteEventLog:
    return SQLiteEventLog(path=db_path)


class TestSQLiteEventLog:
    def test_append_assigns_monotonic_seq_isolated_per_run(self, log: SQLiteEventLog) -> None:
        """Interleaved runs each keep their own monotonic counter starting at 1."""
        log.append(run_id="run-A", type="run_created", actor="harness")
        log.append(run_id="run-A", type="stage_started", actor="harness")
        b1 = log.append(run_id="run-B", type="run_created", actor="harness")
        a3 = log.append(run_id="run-A", type="stage_completed", actor="harness")
        b2 = log.append(run_id="run-B", type="stage_started", actor="harness")
        assert b1.seq == 1
        assert b2.seq == 2
        assert a3.seq == 3

    def test_list_events_returns_ordered_events_isolated_per_run(self, log: SQLiteEventLog) -> None:
        log.append(run_id="run-A", type="run_created", actor="harness")
        log.append(run_id="run-A", type="stage_started", actor="harness")
        log.append(run_id="run-A", type="stage_completed", actor="harness")
        log.append(run_id="run-B", type="run_created", actor="harness")

        a_events = log.list_events("run-A")
        b_events = log.list_events("run-B")

        assert [e.seq for e in a_events] == [1, 2, 3]
        assert [e.type for e in a_events] == ["run_created", "stage_started", "stage_completed"]
        assert all(e.run_id == "run-A" for e in a_events)
        assert len(b_events) == 1 and b_events[0].run_id == "run-B"

    def test_get_timeline_aliases_list_events(self, log: SQLiteEventLog) -> None:
        log.append(run_id="run-A", type="run_created", actor="harness")
        log.append(run_id="run-A", type="stage_started", actor="harness")
        assert log.get_timeline("run-A") == log.list_events("run-A")

    def test_duplicate_seq_raises_event_seq_conflict_error(self, log: SQLiteEventLog) -> None:
        """A colliding ``(run_id, seq)`` maps sqlite3.IntegrityError → EventSeqConflictError,
        with the underlying IntegrityError chained as ``__cause__``."""
        log.append(run_id="run-A", type="run_created", actor="harness")  # takes seq 1

        with pytest.raises(EventSeqConflictError) as exc_info:
            log._append_with_explicit_seq(  # type: ignore[attr-defined]
                run_id="run-A",
                seq=1,  # collides with the append above
                type="run_created",
                actor="harness",
            )
        assert isinstance(exc_info.value.__cause__, sqlite3.IntegrityError)

    def test_payload_and_artifact_ids_round_trip(self, log: SQLiteEventLog) -> None:
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
