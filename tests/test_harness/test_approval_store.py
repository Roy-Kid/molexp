"""``SQLiteApprovalStore`` round-trip laws (spec vision-loop-01-approval-inbox).

The store turns approvals from per-process callbacks into a durable resource.
Its one non-obvious rule is the replay law:

* **Grants replay** — a stored grant is durable consent; ``granted_decision_for``
  returns it on every later re-entry.
* **Rejections never replay** — a rejection is recorded (history), but
  ``granted_decision_for`` returns ``None`` and a later ``record_pending`` for
  the same request id re-opens it as pending ("not now" must never deadlock
  re-entry forever).
* ``record_pending`` is an idempotent upsert that never downgrades a
  **granted** row back to pending. (Per the spec's replay law, "decided" here
  means *granted*: a rejected row is deliberately re-askable.)
* ``record_decision`` on an unknown ``request_id`` fails loud — no fallback.

The store shares the run's existing ``harness.sqlite`` (same ``db_path`` as
``SQLiteEventLog`` / ``SQLiteArtifactLineageStore``).
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from molexp.harness.schemas import ApprovalDecision, ApprovalRequest
from molexp.harness.store.approval_store import ApprovalStore, SQLiteApprovalStore

_RUN_ID = "run-approvals"


def _request(request_id: str = "req-1") -> ApprovalRequest:
    return ApprovalRequest(
        id=request_id,
        intent="experiment_spec",
        reason="approve the concrete experiment spec",
        triggered_by_policy="PlanMode",
        metadata={"execution_backend": "local"},
        created_at=datetime(2026, 7, 3, tzinfo=UTC),
    )


def _decision(
    request_id: str = "req-1",
    *,
    granted: bool,
    decided_by: str = "tester",
    reason: str | None = None,
) -> ApprovalDecision:
    return ApprovalDecision(
        request_id=request_id,
        granted=granted,
        decided_by=decided_by,
        decided_at=datetime(2026, 7, 3, 12, 0, tzinfo=UTC),
        reason=reason,
    )


@pytest.fixture()
def store(tmp_path: Path) -> SQLiteApprovalStore:
    return SQLiteApprovalStore(tmp_path / "harness.sqlite")


# ───────────────────────────────────────────────────────────── basics


class TestApprovalStoreBasics:
    def test_record_pending_round_trips_the_request(self, store: SQLiteApprovalStore) -> None:
        request = _request()
        store.record_pending(_RUN_ID, request)

        [pending] = store.pending(_RUN_ID)
        assert pending.id == request.id
        assert pending.intent == request.intent
        assert pending.reason == request.reason
        assert pending.triggered_by_policy == request.triggered_by_policy
        assert pending.metadata == request.metadata
        assert pending.created_at == request.created_at

    def test_pending_for_unknown_run_is_empty(self, store: SQLiteApprovalStore) -> None:
        assert store.pending("run-that-never-asked") == []

    def test_granted_decision_for_unknown_request_returns_none(
        self, store: SQLiteApprovalStore
    ) -> None:
        assert store.granted_decision_for("req-never-seen") is None

    def test_sqlite_store_satisfies_the_protocol(self, store: SQLiteApprovalStore) -> None:
        assert isinstance(store, ApprovalStore)

    def test_store_and_pending_error_are_harness_public_surface(self) -> None:
        # Spec: harness/__init__ exports both (public surface 19 → 21 symbols).
        import molexp.harness as harness

        assert harness.SQLiteApprovalStore is SQLiteApprovalStore
        assert issubclass(harness.ApprovalPendingError, harness.StageExecutionError)


# ─────────────────────────────────────────────────────── replay law


class TestReplayLaw:
    def test_grant_replays_and_clears_pending(self, store: SQLiteApprovalStore) -> None:
        request = _request()
        store.record_pending(_RUN_ID, request)
        store.record_decision(_decision(granted=True, decided_by="ui-operator", reason="ok"))

        stored = store.granted_decision_for(request.id)
        assert stored is not None
        assert stored.request_id == request.id
        assert stored.granted is True
        assert stored.decided_by == "ui-operator"
        assert stored.reason == "ok"
        assert store.pending(_RUN_ID) == []

    def test_rejection_is_recorded_but_never_replays(self, store: SQLiteApprovalStore) -> None:
        request = _request()
        store.record_pending(_RUN_ID, request)
        store.record_decision(_decision(granted=False, reason="not now"))

        # Decided (not pending anymore), but a rejection is not a replayable grant.
        assert store.granted_decision_for(request.id) is None
        assert store.pending(_RUN_ID) == []

    def test_reentry_after_rejection_reopens_pending(self, store: SQLiteApprovalStore) -> None:
        request = _request()
        store.record_pending(_RUN_ID, request)
        store.record_decision(_decision(granted=False, reason="not now"))

        # A later re-entered gate re-asks: record_pending re-opens the row.
        store.record_pending(_RUN_ID, request)
        [reopened] = store.pending(_RUN_ID)
        assert reopened.id == request.id
        assert store.granted_decision_for(request.id) is None


# ────────────────────────────────────────────── edge cases / immutability


class TestPendingUpsert:
    def test_record_pending_is_idempotent(self, store: SQLiteApprovalStore) -> None:
        request = _request()
        store.record_pending(_RUN_ID, request)
        store.record_pending(_RUN_ID, request)

        assert len(store.pending(_RUN_ID)) == 1

    def test_record_pending_never_downgrades_a_granted_row(
        self, store: SQLiteApprovalStore
    ) -> None:
        request = _request()
        store.record_pending(_RUN_ID, request)
        store.record_decision(_decision(granted=True, decided_by="ui-operator"))

        # A re-entering gate racing the decision must not erase durable consent.
        store.record_pending(_RUN_ID, request)

        stored = store.granted_decision_for(request.id)
        assert stored is not None
        assert stored.decided_by == "ui-operator"
        assert store.pending(_RUN_ID) == []

    def test_record_decision_on_unknown_request_id_raises(self, store: SQLiteApprovalStore) -> None:
        # "Fails loud on an unknown request_id — no fallback" (spec §1).
        with pytest.raises(ValueError, match="request_id"):
            store.record_decision(_decision("req-never-recorded", granted=True))


# ─────────────────────────────────────────────────────── integration


class TestSharedHarnessSqlite:
    def test_store_shares_the_runs_harness_sqlite(self, tmp_path: Path) -> None:
        """The approvals table lives in the same ``harness.sqlite`` as the
        event log — one run directory, one database file."""
        from molexp.harness.store.sqlite_event_log import SQLiteEventLog

        db = tmp_path / "harness.sqlite"
        event_log = SQLiteEventLog(path=db)
        store = SQLiteApprovalStore(db)

        event_log.append(run_id=_RUN_ID, type="stage_started", actor="test")
        store.record_pending(_RUN_ID, _request())

        assert db.exists()
        with sqlite3.connect(db) as conn:
            tables = {
                row[0]
                for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
            }
        assert "approvals" in tables
        assert "events" in tables
        assert len(event_log.list_events(_RUN_ID)) == 1
        assert len(store.pending(_RUN_ID)) == 1
