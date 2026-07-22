"""``ApprovalEventRecorder`` — approval events threaded into the harness log.

Pins the exact persisted payload shape and actor defaults for the request and
the granted-decision events (the rejected type + audit ordering across all
three gate paths are pinned end-to-end in ``test_approval_gate.py``).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from molexp.harness.policy.event_log import ApprovalEventRecorder
from molexp.harness.schemas.approval import ApprovalDecision, ApprovalRequest


@pytest.fixture()
def event_log(tmp_path: Path):
    from molexp.harness.store.sqlite_event_log import SQLiteEventLog

    return SQLiteEventLog(path=tmp_path / "events.sqlite")


def _request() -> ApprovalRequest:
    return ApprovalRequest(
        id="req-abc",
        intent="hpc_submission",
        reason="Workflow targets slurm backend",
        triggered_by_policy="require_for_hpc_submission",
        metadata={"execution_backend": "slurm"},
        created_at=datetime(2026, 5, 26, tzinfo=UTC),
    )


class TestApprovalEventRecorder:
    def test_record_request_persists_requested_event_with_id_in_payload(self, event_log) -> None:
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
        # The event is persisted: list_events surfaces it.
        assert event_log.list_events("run-001")[-1] == event

    def test_record_decision_granted_defaults_actor_to_decided_by(self, event_log) -> None:
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
