"""Tests for record_approval_request + record_approval_decision (Phase 6).

Asserts the exact persisted payload shape and actor defaults for the request
and the granted-decision events (the rejected type + audit ordering are pinned
end-to-end in ``test_approval_gate.py``).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest


@pytest.fixture()
def event_log(tmp_path: Path):
    from molexp.harness.store.sqlite_event_log import SQLiteEventLog

    return SQLiteEventLog(path=tmp_path / "events.sqlite")


def _request():
    from molexp.harness.schemas.approval import ApprovalRequest

    return ApprovalRequest(
        id="req-abc",
        intent="hpc_submission",
        reason="Workflow targets slurm backend",
        triggered_by_policy="require_for_hpc_submission",
        metadata={"execution_backend": "slurm"},
        created_at=datetime(2026, 5, 26, tzinfo=UTC),
    )


# -------------------------------------------------- record_approval_request


class TestApprovalEventLog:
    def test_record_approval_request_writes_correct_event(self, event_log) -> None:
        from molexp.harness.policy.event_log import ApprovalEventRecorder

        req = _request()
        event = ApprovalEventRecorder.record_request(event_log, "run-001", req)

        assert event.type == "approval_requested"
        assert event.actor == "harness"
        assert event.payload == {
            "request_id": req.id,
            "intent": req.intent,
            "reason": req.reason,
            "triggered_by_policy": req.triggered_by_policy,
            "metadata": req.metadata,
        }
        # ApprovalRequest.id is NOT an artifact_store id — it lives in payload.
        assert event.artifact_ids == []

        # The event is persisted: list_events should surface it.
        listed = event_log.list_events("run-001")
        assert listed[-1] == event

    # ------------------------------------------------- record_approval_decision

    def test_record_approval_decision_granted(self, event_log) -> None:
        from molexp.harness.policy.event_log import ApprovalEventRecorder
        from molexp.harness.schemas.approval import ApprovalDecision

        req = _request()
        decision = ApprovalDecision(
            request_id=req.id,
            granted=True,
            decided_by="alice",
            decided_at=datetime(2026, 5, 26, tzinfo=UTC),
            reason="Reviewed and OK",
        )
        event = ApprovalEventRecorder.record_decision(event_log, "run-001", req, decision)
        assert event.type == "approval_granted"
        assert event.actor == "alice"  # defaults to decision.decided_by
        assert event.payload == {
            "request_id": req.id,
            "intent": req.intent,
            "decided_by": "alice",
            "reason": "Reviewed and OK",
            "decided_at": decision.decided_at.isoformat(),
        }
        assert event.artifact_ids == []
