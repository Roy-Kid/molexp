"""Event-log helpers for approval requests + decisions (Phase 6).

Thread :class:`ApprovalRequest` / :class:`ApprovalDecision` events into
the existing ``HarnessEvent`` stream via the Phase-1 :class:`EventLog`
Protocol. The helpers are context-free (they take ``EventLog`` directly
rather than ``HarnessRunContext``) so non-Stage callers — the Phase-7+
orchestrator that drives the request → decide loop — can use them
without spinning up a full run context.

Event types come from the Phase-1 ``EventType`` Literal which already
includes ``"approval_requested"`` / ``"approval_granted"`` /
``"approval_rejected"`` — no widening needed.
"""

from __future__ import annotations

from molexp.harness.schemas.approval import ApprovalDecision, ApprovalRequest
from molexp.harness.schemas.event import HarnessEvent
from molexp.harness.store.event_log import EventLog

__all__ = ["record_approval_decision", "record_approval_request"]


def record_approval_request(
    event_log: EventLog,
    run_id: str,
    request: ApprovalRequest,
    actor: str = "harness",
) -> HarnessEvent:
    """Append an ``approval_requested`` event capturing the request.

    The request id lives in ``payload["request_id"]`` rather than
    ``artifact_ids`` because :class:`ApprovalRequest.id` is not an
    artifact_store id — putting it in ``artifact_ids`` would let
    downstream consumers attempt ``artifact_store.get_ref(request.id)``
    and raise :class:`ArtifactNotFoundError`.
    """
    return event_log.append(
        run_id=run_id,
        type="approval_requested",
        actor=actor,
        payload={
            "request_id": request.id,
            "intent": request.intent,
            "reason": request.reason,
            "triggered_by_policy": request.triggered_by_policy,
            "metadata": request.metadata,
        },
        artifact_ids=[],
    )


def record_approval_decision(
    event_log: EventLog,
    run_id: str,
    request: ApprovalRequest,
    decision: ApprovalDecision,
    actor: str | None = None,
) -> HarnessEvent:
    """Append ``approval_granted`` or ``approval_rejected`` per ``decision.granted``.

    ``actor`` defaults to ``decision.decided_by``; pass an explicit value
    to override (e.g. the orchestrator wants to record the harness as the
    actor rather than the human who clicked). The request id lives in
    ``payload["request_id"]`` (see :func:`record_approval_request` for the
    artifact_ids hygiene rationale).
    """
    return event_log.append(
        run_id=run_id,
        type="approval_granted" if decision.granted else "approval_rejected",
        actor=actor if actor is not None else decision.decided_by,
        payload={
            "request_id": request.id,
            "intent": request.intent,
            "decided_by": decision.decided_by,
            "reason": decision.reason,
            "decided_at": decision.decided_at.isoformat(),
        },
        artifact_ids=[],
    )
