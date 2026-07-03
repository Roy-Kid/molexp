"""``ApprovalGate`` — pipeline gate that resolves approvals at run time.

Takes the :class:`ApprovalRequest`\\ s to gate on plus an optional ``approve``
callback (the *approver*) that turns each request into an
:class:`ApprovalDecision` **when the gate runs** — so ``decided_at`` is the
real decision moment, not pipeline-declaration time.

Resolution order, per request (the store-first contract):

1. a **stored grant** in ``ctx.approval_store`` passes immediately (the
   decision is re-logged so the audit shows the re-entry resolved from the
   stored grant, ``decided_by`` preserved);
2. otherwise an **injected approver** is asked live, and its decision is
   persisted to the store and the event log;
3. otherwise the request is recorded **pending** and the gate raises
   :class:`~molexp.harness.errors.ApprovalPendingError` — the pipeline
   suspends; a decision (UI inbox / TTY re-run / ``--yes``) lets a
   ledger-resumed re-entry pass at step 1.

There is **no auto-grant default**: ``approve=None`` means steps 1-3, never
:func:`auto_grant_approver`. That function survives strictly as an explicit
injection for tests and deliberately unattended runs.

Audit contract: for every request the gate records ``approval_requested``
first, then obtains the decision, then records ``approval_granted`` /
``approval_rejected`` — all **before** deciding whether to pass or fail, so
an audit consumer can always answer "who was asked, who decided what?" even
when the stage aborts. A pending suspension records only the request (there
is no decision yet). If any decision is ``granted=False`` the stage then
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
from molexp.harness.errors import ApprovalPendingError, StageExecutionError
from molexp.harness.policy.event_log import ApprovalEventRecorder
from molexp.harness.schemas import ApprovalDecision, ApprovalRequest, PlanArtifactRef

__all__ = ["ApprovalGate", "Approver", "auto_grant_approver"]

Approver = Callable[[ApprovalRequest], Awaitable[ApprovalDecision]]
"""Async callback resolving one :class:`ApprovalRequest` into a decision."""


async def auto_grant_approver(request: ApprovalRequest) -> ApprovalDecision:
    """Grant ``request`` unconditionally, stamped at the actual decision time.

    **Explicit injection only** — never a default. A pipeline that should run
    unattended (tests, deliberate batch runs) opts in by passing this
    approver; every other ``approve=None`` gate suspends pending instead
    (see the module docstring). The grant is still recorded on the event log
    like any other decision, so the audit trail states explicitly that an
    auto-approver decided.
    """
    return ApprovalDecision(
        request_id=request.id,
        granted=True,
        decided_by="auto-approver",
        decided_at=datetime.now(tz=UTC),
        reason="auto-grant (explicitly injected auto-approver)",
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
        result_kind: str = "analysis_result",
    ) -> None:
        self._requests = list(requests)
        # ``None`` is meaningful (store-first → live approver → suspend);
        # it is NEVER substituted with an auto-granting default.
        self._approve = approve
        self._subject_artifact_ids = list(subject_artifact_ids or [])
        # The kind of the summary artifact this gate persists on grant. Defaults
        # to ``analysis_result``; an EARLY gate (e.g. PlanMode's spec-approval
        # checkpoint) overrides it so it does not collide with the terminal
        # gate's ``analysis_result`` (which keys the UI's "Review" rail step).
        self._result_kind = result_kind
        # A Mode keys its completion ledger on ``stage.name``; when a single
        # mode wires more than one gate (e.g. PlanMode's experiment-spec
        # checkpoint plus the terminal final-report gate) each needs a
        # distinct name. We shadow the ``name`` ClassVar on this instance only
        # (class-level ``ApprovalGate.name`` stays "approval_gate"); the
        # ``object.__setattr__`` keeps that explicit and type-checker-clean.
        if name is not None:
            object.__setattr__(self, "name", name)

    async def run(self, ctx: HarnessRunContext) -> PlanArtifactRef:
        # Record request THEN decision for each ask, before the gate
        # decides pass/fail — so the audit trail captures the full ledger
        # even if the gate then aborts on a rejection or suspends pending.
        decisions: list[tuple[ApprovalRequest, ApprovalDecision]] = []
        pending: list[ApprovalRequest] = []
        for request in self._requests:
            ApprovalEventRecorder.record_request(ctx.event_log, ctx.run_id, request)

            # 1. Store-first: a persisted grant is durable consent.
            stored = (
                ctx.approval_store.granted_decision_for(request.id)
                if ctx.approval_store is not None
                else None
            )
            if stored is not None:
                ApprovalEventRecorder.record_decision(ctx.event_log, ctx.run_id, request, stored)
                decisions.append((request, stored))
                continue

            # 2. Live approver (interactive TTY, --yes, or explicit auto-grant).
            if self._approve is not None:
                decision = await self._approve(request)
                if decision.request_id != request.id:
                    raise StageExecutionError(
                        f"ApprovalGate: approver answered request {decision.request_id!r} "
                        f"for request {request.id!r} — refusing the mismatched decision"
                    )
                if ctx.approval_store is not None:
                    ctx.approval_store.record_pending(ctx.run_id, request)
                    ctx.approval_store.record_decision(decision)
                ApprovalEventRecorder.record_decision(ctx.event_log, ctx.run_id, request, decision)
                decisions.append((request, decision))
                continue

            # 3. Nobody to ask: record pending, suspend after the loop.
            if ctx.approval_store is not None:
                ctx.approval_store.record_pending(ctx.run_id, request)
            pending.append(request)

        if pending:
            raise ApprovalPendingError(pending, ctx.run_id)

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
            kind=self._result_kind,
            obj=summary,
            created_by="ApprovalGate",
            parent_ids=list(self._subject_artifact_ids),
        )
