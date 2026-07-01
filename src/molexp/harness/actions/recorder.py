"""``ProposalActionRecorder`` — traceable action events for guarded execution.

integration.md §8.3, invariant #7. A static-method helper that threads a
granted proposal's *execution* attempt into the append-only ``HarnessEvent``
stream, mirroring :class:`~molexp.harness.policy.event_log.ApprovalEventRecorder`
(which records the *approval* half). Every action event carries the proposal id
and the :class:`~molexp.harness.schemas.change_proposal.HighRiskOp` in its
payload so an auditor can chain ``approval_granted`` → the action outcome.

It **reuses** the existing ``EventType`` values ``tool_called`` /
``tool_completed`` / ``tool_failed`` — no widening of the ``EventType`` Literal.
The proposal id rides ``payload["proposal_id"]`` rather than ``artifact_ids``
because it is not an artifact-store id (same hygiene rationale as
``ApprovalEventRecorder``); ``record_completed`` puts the outcome's
``result_artifact_ids`` in ``artifact_ids``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from molexp.harness.schemas import HarnessEvent
    from molexp.harness.schemas.change_proposal import ChangeProposal, ProposalOutcome
    from molexp.harness.store.event_log import EventLog

__all__ = ["ProposalActionRecorder"]

#: The actor recorded for every guarded-execution action event.
_ACTOR = "proposal-executor"


class ProposalActionRecorder:
    """Append ``tool_called`` / ``tool_completed`` / ``tool_failed`` for a proposal."""

    @staticmethod
    def record_started(event_log: EventLog, run_id: str, proposal: ChangeProposal) -> HarnessEvent:
        """Append a ``tool_called`` event marking the start of dispatch."""
        return event_log.append(
            run_id=run_id,
            type="tool_called",
            actor=_ACTOR,
            payload={
                "proposal_id": proposal.id,
                "high_risk_op": proposal.proposed_change.op,
            },
            artifact_ids=[],
        )

    @staticmethod
    def record_completed(
        event_log: EventLog, run_id: str, proposal: ChangeProposal, outcome: ProposalOutcome
    ) -> HarnessEvent:
        """Append a ``tool_completed`` event carrying the executed outcome."""
        return event_log.append(
            run_id=run_id,
            type="tool_completed",
            actor=_ACTOR,
            payload={
                "proposal_id": proposal.id,
                "high_risk_op": proposal.proposed_change.op,
                "status": outcome.status,
            },
            artifact_ids=list(outcome.result_artifact_ids),
        )

    @staticmethod
    def record_failed(
        event_log: EventLog, run_id: str, proposal: ChangeProposal, reason: str
    ) -> HarnessEvent:
        """Append a ``tool_failed`` event capturing the failure reason."""
        return event_log.append(
            run_id=run_id,
            type="tool_failed",
            actor=_ACTOR,
            payload={
                "proposal_id": proposal.id,
                "high_risk_op": proposal.proposed_change.op,
                "reason": reason,
            },
            artifact_ids=[],
        )
