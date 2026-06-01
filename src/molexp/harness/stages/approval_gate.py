"""``ApprovalGate`` — pipeline gate on a list of resolved ApprovalDecisions.

Takes pre-resolved ``(ApprovalRequest, ApprovalDecision)`` pairs. The actual
*getting* of decisions (interactive UX or auto-approver) is the
orchestrator's responsibility — Phase 9 ships only the gate logic.

Every decision — granted or rejected — is recorded onto the event log
via :func:`record_approval_decision` **before** the gate decides whether
to pass or fail, so an audit consumer can always answer "who decided
what?" even when the stage aborts. If any decision is ``granted=False``
the stage then raises :class:`StageExecutionError` listing the rejected
intents. If every decision is ``granted=True`` the stage persists a
summary artifact (kind ``analysis_result``) and returns its ref.

``subject_artifact_ids`` (optional) lets callers attribute the summary
to the artifacts being gated (e.g. the bound workflow + test spec):
those ids become ``parent_ids`` on the persisted summary so
``trace_backward`` can answer "what was approved to produce X?".
"""

from __future__ import annotations

from typing import ClassVar

from molexp.harness.core.run_context import HarnessRunContext
from molexp.harness.core.stage import Stage
from molexp.harness.errors import StageExecutionError
from molexp.harness.policy.event_log import record_approval_decision
from molexp.harness.schemas import ApprovalDecision, ApprovalRequest, ArtifactRef

__all__ = ["ApprovalGate"]


class ApprovalGate(Stage):
    """Gate pipeline progression on pre-resolved approvals."""

    name: ClassVar[str] = "approval_gate"

    def __init__(
        self,
        decisions: list[tuple[ApprovalRequest, ApprovalDecision]],
        *,
        subject_artifact_ids: list[str] | None = None,
    ) -> None:
        self._decisions = list(decisions)
        self._subject_artifact_ids = list(subject_artifact_ids or [])

    async def run(self, ctx: HarnessRunContext) -> ArtifactRef:
        # Record every decision FIRST so the audit trail captures the
        # full ledger even if the gate then aborts. Skipping this on
        # rejection (the old behavior) lost the audit row for the very
        # rejection that aborted the run — the worst time to lose it.
        for req, dec in self._decisions:
            record_approval_decision(ctx.event_log, ctx.run_id, req, dec)

        rejected = [(req, dec) for req, dec in self._decisions if not dec.granted]
        if rejected:
            intents = [req.intent for req, _ in rejected]
            raise StageExecutionError(
                f"ApprovalGate: {len(rejected)} approval(s) rejected; intents={intents}"
            )

        summary = {
            "approved_intents": [req.intent for req, _ in self._decisions],
            "decided_by": list({dec.decided_by for _, dec in self._decisions}),
            "decision_count": len(self._decisions),
        }
        return ctx.artifact_store.put_json(
            kind="analysis_result",
            obj=summary,
            created_by="ApprovalGate",
            parent_ids=list(self._subject_artifact_ids),
        )
