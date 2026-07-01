"""High-risk-mutation classifier + the dry, audited ``ChangeProposal`` gate.

integration.md §6.2, §8.3. This is the coordination guard — a §8.1 high-risk
mutation MUST become a :class:`~molexp.harness.schemas.change_proposal.ChangeProposal`
and pass an approval gate before any executor could run.

The gate is deliberately **not** the :class:`~molexp.harness.stages.approval_gate.ApprovalGate`
Stage: that Stage *raises* to abort a pipeline on rejection, but §8.3 requires a
rejected proposal to be **recorded, not aborted**. So :func:`gate_change_proposal`
records the decision and returns the proposal's outcome, reusing the *same* audit
chokepoint (:class:`~molexp.harness.policy.event_log.ApprovalEventRecorder`) — not a
parallel gate. It is **dry**: no executor runs; ``execution_result`` records the
*decision*, not an execution (executors are P2.1).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, get_args

from molexp.harness.core.run_context import HarnessRunContext
from molexp.harness.policy.event_log import ApprovalEventRecorder
from molexp.harness.schemas import ApprovalRequest
from molexp.harness.schemas.change_proposal import (
    CHANGE_PROPOSAL_KIND,
    ApprovalLevel,
    ChangeProposal,
    HighRiskOp,
    ProposalOutcome,
    Reversibility,
)
from molexp.harness.stages.approval_gate import Approver, auto_grant_approver

if TYPE_CHECKING:
    from molexp.harness.actions import ProposalExecutor

__all__ = [
    "HIGH_RISK_OPS",
    "approval_level_for",
    "gate_change_proposal",
    "is_high_risk",
    "requires_change_proposal",
]

HIGH_RISK_OPS: frozenset[str] = frozenset(get_args(HighRiskOp))
"""The §8.1 op strings that require a ChangeProposal (derived from ``HighRiskOp``)."""


def is_high_risk(op: str) -> bool:
    """Whether *op* is a §8.1 high-risk mutation."""
    return op in HIGH_RISK_OPS


def requires_change_proposal(op: str) -> bool:
    """Whether *op* must go through a :class:`ChangeProposal` + gate before executing."""
    return is_high_risk(op)


def approval_level_for(op: HighRiskOp, reversibility: Reversibility) -> ApprovalLevel:
    """The minimum approval level for a high-risk *op*.

    Every high-risk op needs at least ``user``; an ``irreversible`` change
    escalates to ``elevated`` (§8.2: "irreversible forces at least user").
    """
    del op  # every high-risk op is ≥ user; the escalation is reversibility-driven
    return "elevated" if reversibility == "irreversible" else "user"


def _to_approval_request(proposal: ChangeProposal, *, now: datetime) -> ApprovalRequest:
    """Map a proposal onto an :class:`ApprovalRequest` the audit gate decides on.

    Uses the existing ``intent="overwrite"`` (the closest of the fixed
    ``ApprovalIntent`` vocabulary for "mutating existing canonical state"); the
    specific :data:`HighRiskOp` + proposal id ride ``metadata``.
    """
    return ApprovalRequest(
        id=f"apr-{proposal.id}",
        intent="overwrite",
        reason=proposal.intent,
        triggered_by_policy="change_proposal",
        metadata={
            "proposal_id": proposal.id,
            "high_risk_op": proposal.proposed_change.op,
            "approval_level": proposal.approval_level,
        },
        created_at=now,
    )


async def gate_change_proposal(
    ctx: HarnessRunContext,
    proposal: ChangeProposal,
    *,
    approve: Approver | None = None,
    now: datetime | None = None,
    executor: ProposalExecutor | None = None,
) -> ChangeProposal:
    """Store, gate, and (optionally) execute *proposal*.

    Persists the proposal as a ``change_proposal`` artifact (so a rejected
    proposal is preserved, §8.3), records ``approval_requested`` then
    ``approval_granted`` / ``approval_rejected`` via the shared recorder, and
    returns the proposal with its :class:`ProposalOutcome` in
    ``execution_result``.

    Two modes (integration.md §6.1):

    - **Advisory (default, ``executor is None``)** — dry: the outcome is
      ``granted`` or ``rejected``; nothing runs. This is the historical behavior.
    - **Guarded execution (``executor`` provided)** — on a *granted* decision the
      mutation is dispatched through *executor* after ``approval_granted`` is
      recorded, and the executor's ``ProposalOutcome`` (``executed`` / ``failed``)
      becomes ``execution_result``. A *rejected* decision stays dry regardless.

    **Single gate.** This ApprovalGate *is* the approval for the mutation: the
    slice-01/02 curation handlers mutate in-process and never re-run
    :func:`~molexp.harness.policy.side_effect_gate.enforce_side_effect_approvals`,
    so an approved mutation is gated exactly once — here. The dispatch is not
    wrapped in try/except: a runtime op failure is caught *inside* the executor
    and surfaced as ``status="failed"`` (recorded), while a structural error
    (unhandled op) propagates past this gate unchanged.

    Args:
        ctx: The harness run context (artifact store + event log + run id).
        proposal: The change proposal to gate.
        approve: The :class:`Approver` resolving the request (default
            :func:`auto_grant_approver`).
        now: Timestamp stamped on the approval request (default aware-UTC now).
        executor: When provided, the :class:`ProposalExecutor` a *granted*
            proposal is dispatched through (guarded execution). ``None`` keeps
            the dry advisory behavior.

    Returns:
        A copy of *proposal* with ``execution_result`` filled.
    """
    approve = approve if approve is not None else auto_grant_approver
    now = now if now is not None else datetime.now(tz=UTC)

    proposal_ref = ctx.artifact_store.put_json(
        kind=CHANGE_PROPOSAL_KIND,
        obj=proposal.model_dump(mode="json"),
        created_by="change-proposal-gate",
        parent_ids=list(proposal.evidence),
    )

    request = _to_approval_request(proposal, now=now)
    ApprovalEventRecorder.record_request(ctx.event_log, ctx.run_id, request)
    decision = await approve(request)
    ApprovalEventRecorder.record_decision(ctx.event_log, ctx.run_id, request, decision)

    if decision.granted and executor is not None:
        # Guarded execution: the gate granted, so dispatch the mutation. The
        # executor records tool_completed / tool_failed and returns an
        # executed/failed outcome; an unhandled op propagates past the gate.
        outcome = await executor.dispatch(ctx, proposal)
    else:
        # Advisory (dry): reject, or granted-with-no-executor. Nothing runs.
        outcome = ProposalOutcome(
            status="granted" if decision.granted else "rejected",
            decided_by=decision.decided_by,
            decided_at=decision.decided_at,
            reason=decision.reason,
            result_artifact_ids=[proposal_ref.id],
        )
    return proposal.model_copy(update={"execution_result": outcome})
