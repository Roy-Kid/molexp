"""``SQLiteApprovalStore`` round-trip + replay laws (spec vision-loop-01).

The store turns approvals from per-process callbacks into a durable resource.
Its laws:

* **Grants replay** — a stored grant is durable consent; ``granted_decision_for``
  returns it on every later re-entry.
* **Rejections never replay** — recorded as history, but ``granted_decision_for``
  returns ``None`` and a later ``record_pending`` re-opens the request pending
  ("not now" must never deadlock re-entry forever).
* ``record_pending`` is an idempotent upsert that never downgrades a *granted*
  row back to pending.
* ``record_decision`` on an unknown ``request_id`` fails loud — no fallback.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from molexp.harness.schemas import ApprovalDecision, ApprovalRequest
from molexp.harness.store.approval_store import SQLiteApprovalStore

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


def _intervention_request(request_id: str = "req-int") -> ApprovalRequest:
    return ApprovalRequest(
        id=request_id,
        intent="task_intervention",
        reason="loop stuck — need human guidance",
        triggered_by_policy="StepAuditLoop",
        metadata={},
        created_at=datetime(2026, 7, 22, tzinfo=UTC),
        scope="intervention_request",
        target_agent_id="codegen-task-T",
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


class TestSQLiteApprovalStore:
    def test_record_pending_round_trips_every_field(self, store: SQLiteApprovalStore) -> None:
        request = _request()
        store.record_pending(_RUN_ID, request)

        [pending] = store.pending(_RUN_ID)
        assert pending.id == request.id
        assert pending.intent == request.intent
        assert pending.reason == request.reason
        assert pending.triggered_by_policy == request.triggered_by_policy
        assert pending.metadata == request.metadata
        assert pending.created_at == request.created_at

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

    def test_record_pending_after_rejection_reopens_pending(
        self, store: SQLiteApprovalStore
    ) -> None:
        request = _request()
        store.record_pending(_RUN_ID, request)
        store.record_decision(_decision(granted=False, reason="not now"))

        # A later re-entered gate re-asks: record_pending re-opens the row.
        store.record_pending(_RUN_ID, request)
        [reopened] = store.pending(_RUN_ID)
        assert reopened.id == request.id
        assert store.granted_decision_for(request.id) is None

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


# ---- plan-emergent-07: scope + target_agent_id round-trip + additive migration ----


# Pre-generalization (legacy) ``approvals`` table: the exact column set before
# plan-emergent-07 added ``scope`` / ``target_agent_id``. Used to prove the
# idempotent additive migration (PRAGMA table_info -> ALTER TABLE ADD COLUMN)
# defaults a legacy row's scope to ``approval_gate``.
_LEGACY_SCHEMA_SQL = """
CREATE TABLE approvals (
    request_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    intent TEXT NOT NULL,
    reason TEXT NOT NULL,
    triggered_by_policy TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    requested_at TEXT NOT NULL,
    state TEXT NOT NULL,
    decided_by TEXT,
    decided_at TEXT,
    decision_reason TEXT
);
"""


class TestApprovalStoreScope:
    def test_intervention_request_round_trips_scope_and_target(
        self, store: SQLiteApprovalStore
    ) -> None:
        request = _intervention_request()
        store.record_pending(_RUN_ID, request)

        [pending] = store.pending(_RUN_ID)
        assert pending.id == request.id
        assert pending.intent == "task_intervention"
        assert pending.scope == "intervention_request"
        assert pending.target_agent_id == "codegen-task-T"

    def test_grant_still_durable_for_intervention_request(self, store: SQLiteApprovalStore) -> None:
        request = _intervention_request()
        store.record_pending(_RUN_ID, request)
        store.record_decision(_decision(request.id, granted=True, decided_by="ui-operator"))

        stored = store.granted_decision_for(request.id)
        assert stored is not None
        assert stored.granted is True
        assert store.pending(_RUN_ID) == []

    def test_rejection_reopens_intervention_request(self, store: SQLiteApprovalStore) -> None:
        request = _intervention_request()
        store.record_pending(_RUN_ID, request)
        store.record_decision(_decision(request.id, granted=False, reason="not now"))
        assert store.granted_decision_for(request.id) is None

        # reject-reopen law is unchanged: re-entry re-asks with scope preserved.
        store.record_pending(_RUN_ID, request)
        [reopened] = store.pending(_RUN_ID)
        assert reopened.scope == "intervention_request"
        assert reopened.target_agent_id == "codegen-task-T"

    def test_legacy_row_migrates_to_default_scope(self, tmp_path: Path) -> None:
        """A row written before the new columns reads back scope=='approval_gate'.

        The store's ``__init__`` runs an idempotent additive migration; a
        pre-existing legacy ``approvals`` table (no ``scope`` / ``target_agent_id``
        columns) must gain them and default legacy rows to ``approval_gate``.
        """
        db_path = tmp_path / "legacy.sqlite"
        legacy = sqlite3.connect(db_path)
        try:
            legacy.executescript(_LEGACY_SCHEMA_SQL)
            legacy.execute(
                """
                INSERT INTO approvals (
                    request_id, run_id, intent, reason, triggered_by_policy,
                    metadata_json, requested_at, state
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending')
                """,
                (
                    "legacy-req",
                    _RUN_ID,
                    "experiment_spec",
                    "legacy pending row",
                    "PlanMode",
                    "{}",
                    datetime(2026, 7, 1, tzinfo=UTC).isoformat(),
                ),
            )
            legacy.commit()
        finally:
            legacy.close()

        store = SQLiteApprovalStore(db_path)  # __init__ migration runs here
        [pending] = store.pending(_RUN_ID)
        assert pending.id == "legacy-req"
        assert pending.scope == "approval_gate"
        assert pending.target_agent_id is None
