"""``ApprovalGate`` — pipeline gate that resolves approvals at run time.

Takes the :class:`ApprovalRequest`\\ s to gate on plus an ``approve``
callback (the *approver*) that turns each request into an
:class:`ApprovalDecision` **when the gate runs** — so ``decided_at`` is the
real decision moment, not pipeline-declaration time. The default approver
auto-grants every request (the harness pipeline is non-interactive by
design); an interactive or policy-driven approver plugs in through the same
callback without changing the gate.

Audit contract: for every request the gate records ``approval_requested``
first, then obtains the decision, then records ``approval_granted`` /
``approval_rejected`` — all **before** deciding whether to pass or fail, so
an audit consumer can always answer "who was asked, who decided what?" even
when the stage aborts. If any decision is ``granted=False`` the stage then
raises :class:`StageExecutionError` listing the rejected intents. If every
decision is granted the stage persists a summary artifact (kind
``analysis_result``) and returns its ref.

``subject_artifact_ids`` (optional) lets callers attribute the summary
to the artifacts being gated (e.g. the bound workflow + test spec):
those ids become ``parent_ids`` on the persisted summary so
``trace_backward`` can answer "what was approved to produce X?".
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import ClassVar

from molexp.harness.core.run_context import HarnessRunContext
from molexp.harness.core.stage import Stage
from molexp.harness.errors import StageExecutionError
from molexp.harness.policy.event_log import ApprovalEventRecorder
from molexp.harness.schemas import ApprovalDecision, ApprovalRequest, ArtifactRef

__all__ = ["ApprovalGate", "Approver", "auto_grant_approver"]

Approver = Callable[[ApprovalRequest], Awaitable[ApprovalDecision]]
"""Async callback resolving one :class:`ApprovalRequest` into a decision."""


async def auto_grant_approver(request: ApprovalRequest) -> ApprovalDecision:
    """Grant ``request`` unconditionally, stamped at the actual decision time.

    The default for non-interactive pipelines; the grant is still recorded
    on the event log like any other decision, so the audit trail states
    explicitly that an auto-approver decided.
    """
    return ApprovalDecision(
        request_id=request.id,
        granted=True,
        decided_by="auto-approver",
        decided_at=datetime.now(tz=UTC),
        reason="auto-grant (non-interactive pipeline)",
    )


class ApprovalGate(Stage):
    """Gate pipeline progression on approvals resolved at run time."""

    name: ClassVar[str] = "approval_gate"

    def __init__(
        self,
        requests: list[ApprovalRequest],
        *,
        approve: Approver | None = None,
        subject_artifact_ids: list[str] | None = None,
        name: str | None = None,
    ) -> None:
        self._requests = list(requests)
        self._approve = approve if approve is not None else auto_grant_approver
        self._subject_artifact_ids = list(subject_artifact_ids or [])
        # A Mode keys its completion ledger on ``stage.name``; when a single
        # mode wires more than one gate (e.g. PlanMode's experiment-spec
        # checkpoint plus the terminal final-report gate) each needs a
        # distinct name. We shadow the ``name`` ClassVar on this instance only
        # (class-level ``ApprovalGate.name`` stays "approval_gate"); the
        # ``object.__setattr__`` keeps that explicit and type-checker-clean.
        if name is not None:
            object.__setattr__(self, "name", name)

    async def run(self, ctx: HarnessRunContext) -> ArtifactRef:
        # Record request THEN decision for each ask, before the gate
        # decides pass/fail — so the audit trail captures the full ledger
        # even if the gate then aborts on a rejection.
        decisions: list[tuple[ApprovalRequest, ApprovalDecision]] = []
        for request in self._requests:
            ApprovalEventRecorder.record_request(ctx.event_log, ctx.run_id, request)
            decision = await self._approve(request)
            if decision.request_id != request.id:
                raise StageExecutionError(
                    f"ApprovalGate: approver answered request {decision.request_id!r} "
                    f"for request {request.id!r} — refusing the mismatched decision"
                )
            ApprovalEventRecorder.record_decision(ctx.event_log, ctx.run_id, request, decision)
            decisions.append((request, decision))

        rejected = [(req, dec) for req, dec in decisions if not dec.granted]
        if rejected:
            intents = [req.intent for req, _ in rejected]
            raise StageExecutionError(
                f"ApprovalGate: {len(rejected)} approval(s) rejected; intents={intents}"
            )

        summary = {
            "approved_intents": [req.intent for req, _ in decisions],
            "decided_by": list({dec.decided_by for _, dec in decisions}),
            "decision_count": len(decisions),
        }
        return ctx.artifact_store.put_json(
            kind="analysis_result",
            obj=summary,
            created_by="ApprovalGate",
            parent_ids=list(self._subject_artifact_ids),
        )
