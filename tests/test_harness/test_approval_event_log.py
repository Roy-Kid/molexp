"""Tests for record_approval_request + record_approval_decision (Phase 6).

Asserts:
- correct EventType (approval_requested / approval_granted / approval_rejected)
- correct payload fields
- artifact_ids == [request.id]
- actor defaults: record_approval_request defaults "harness";
  record_approval_decision defaults to decision.decided_by, override wins.
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


def test_record_approval_request_writes_correct_event(event_log) -> None:
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


def test_record_approval_request_custom_actor(event_log) -> None:
    from molexp.harness.policy.event_log import ApprovalEventRecorder

    req = _request()
    event = ApprovalEventRecorder.record_request(event_log, "run-001", req, actor="evaluator")
    assert event.actor == "evaluator"


# ------------------------------------------------- record_approval_decision


def test_record_approval_decision_granted(event_log) -> None:
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


def test_record_approval_decision_rejected(event_log) -> None:
    from molexp.harness.policy.event_log import ApprovalEventRecorder
    from molexp.harness.schemas.approval import ApprovalDecision

    req = _request()
    decision = ApprovalDecision(
        request_id=req.id,
        granted=False,
        decided_by="alice",
        decided_at=datetime(2026, 5, 26, tzinfo=UTC),
        reason="Resource too high",
    )
    event = ApprovalEventRecorder.record_decision(event_log, "run-001", req, decision)
    assert event.type == "approval_rejected"


class TestRecordApprovalDecisionTimestamp:
    """record_approval_decision serializes decided_at into the event payload."""

    def test_serializes_decided_at_for_granted_and_rejected(self, event_log) -> None:
        """decided_at reaches the persisted event payload as an ISO string for
        both granted and rejected decisions, without dropping existing keys."""
        from molexp.harness.policy.event_log import ApprovalEventRecorder
        from molexp.harness.schemas.approval import ApprovalDecision

        req = _request()
        decided_at = datetime(2026, 5, 26, 14, 30, tzinfo=UTC)
        for granted in (True, False):
            decision = ApprovalDecision(
                request_id=req.id,
                granted=granted,
                decided_by="alice",
                decided_at=decided_at,
                reason="r",
            )
            event = ApprovalEventRecorder.record_decision(event_log, "run-001", req, decision)
            assert event.payload["decided_at"] == decided_at.isoformat()
            # Existing keys remain present.
            for key in ("request_id", "intent", "decided_by", "reason"):
                assert key in event.payload


def test_record_approval_decision_actor_override(event_log) -> None:
    """Explicit actor= kwarg wins over decision.decided_by."""
    from molexp.harness.policy.event_log import ApprovalEventRecorder
    from molexp.harness.schemas.approval import ApprovalDecision

    req = _request()
    decision = ApprovalDecision(
        request_id=req.id,
        granted=True,
        decided_by="alice",
        decided_at=datetime(2026, 5, 26, tzinfo=UTC),
    )
    event = ApprovalEventRecorder.record_decision(
        event_log, "run-001", req, decision, actor="harness"
    )
    assert event.actor == "harness"


def test_record_approval_decision_reason_none(event_log) -> None:
    from molexp.harness.policy.event_log import ApprovalEventRecorder
    from molexp.harness.schemas.approval import ApprovalDecision

    req = _request()
    decision = ApprovalDecision(
        request_id=req.id,
        granted=True,
        decided_by="alice",
        decided_at=datetime(2026, 5, 26, tzinfo=UTC),
    )
    event = ApprovalEventRecorder.record_decision(event_log, "run-001", req, decision)
    assert event.payload["reason"] is None


# ------------------------------------------------- re-exports


def test_helpers_re_exported() -> None:
    from molexp.harness import (
        ApprovalEventRecorder as via_top_decision,
    )
    from molexp.harness import (
        ApprovalEventRecorder as via_top_request,
    )
    from molexp.harness.policy import (
        ApprovalEventRecorder as via_pkg_decision,
    )
    from molexp.harness.policy import (
        ApprovalEventRecorder as via_pkg_request,
    )

    assert via_top_request is via_pkg_request
    assert via_top_decision is via_pkg_decision
