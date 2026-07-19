"""``StepAuditLoop`` — expert audit dual of :class:`RepairLoop`.

Subject ready → build :class:`ReviewPack` → policy branch:

* ``auto`` — synthetic approve + audit events; never suspends.
* ``soft`` — persist pack only; never suspends.
* ``hard`` — compose :class:`ApprovalGate` store-first; pending raises
  :class:`ApprovalPendingError`; reject/revise feed ``feedback_kind`` and
  optionally re-run ``regenerate`` within ``attempts``.

Optional LLM form enrichment goes **only** through ``ctx.agent_gateway.call``
(structured ``complete_structured`` path) — never hand-rolled JSON parse.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, ClassVar

from mollog import get_logger

from molexp.harness.core.stage import Stage
from molexp.harness.errors import ApprovalPendingError, StageExecutionError
from molexp.harness.policy.event_log import ApprovalEventRecorder
from molexp.harness.schemas import (
    AgentCallSpec,
    ApprovalRequest,
    FormDocument,
    PlanArtifactRef,
    ReviewDecision,
    ReviewPack,
    StepAuditPolicy,
    approval_decision_to_review,
    review_decision_to_approval,
)
from molexp.harness.stages.approval_gate import ApprovalGate, Approver

if TYPE_CHECKING:
    from molexp.harness.core.run_context import HarnessRunContext

__all__ = ["StepAuditLoop"]

_LOG = get_logger(__name__)

PackBuilder = Callable[["HarnessRunContext"], ReviewPack]


class StepAuditLoop(Stage):
    """Composite stage: ReviewPack + policy-gated human/auto decision."""

    name: ClassVar[str] = "step_audit_loop"

    def __init__(
        self,
        *,
        name: str,
        subject_kind: str,
        pack_builder: PackBuilder,
        policy: StepAuditPolicy,
        request: ApprovalRequest,
        approve: Approver | None = None,
        regenerate: Stage | None = None,
        feedback_kind: str = "step_audit_feedback",
        attempts: int = 3,
        enrich_agent: str | None = None,
        result_kind: str = "analysis_result",
        subject_artifact_ids: list[str] | None = None,
    ) -> None:
        if attempts < 1:
            raise ValueError("StepAuditLoop attempts must be >= 1")
        if policy not in ("auto", "soft", "hard"):
            raise ValueError(f"unknown StepAuditPolicy: {policy!r}")
        object.__setattr__(self, "name", name)
        self._subject_kind = subject_kind
        self._pack_builder = pack_builder
        self._policy: StepAuditPolicy = policy
        self._request = request
        self._approve = approve
        self._regenerate = regenerate
        self._feedback_kind = feedback_kind
        self._attempts = attempts
        self._enrich_agent = enrich_agent
        self._result_kind = result_kind
        self._subject_artifact_ids = list(subject_artifact_ids or [])

    async def run(self, ctx: HarnessRunContext) -> PlanArtifactRef:
        last: StageExecutionError | None = None
        for attempt in range(1, self._attempts + 1):
            subject = ctx.artifact_store.latest_by_kind(self._subject_kind)
            if subject is None:
                raise StageExecutionError(
                    f"StepAuditLoop[{self.name}]: missing subject kind {self._subject_kind!r}"
                )

            pack = self._pack_builder(ctx)
            pack = await self._maybe_enrich(ctx, pack, subject)
            pack_ref = ctx.artifact_store.put_json(
                kind="review_pack",
                obj=pack.model_dump(mode="json"),
                created_by=f"stage:{self.name}",
                parent_ids=[subject.id, *self._subject_artifact_ids],
            )

            if self._policy == "auto":
                return self._auto_approve(ctx, pack, pack_ref)

            if self._policy == "soft":
                return pack_ref

            revise = self._consume_pending_revise(ctx)
            if revise is not None:
                self._record_feedback_text(
                    ctx,
                    f"revise by {revise.decided_by}: {revise.reason or ''}\n"
                    f"field_values={json.dumps(revise.field_values, sort_keys=True, default=str)}",
                )
                if self._regenerate is not None:
                    await self._regenerate.run(ctx)
                _LOG.warning(
                    f"[step_audit:{self.name}] attempt {attempt}/{self._attempts} "
                    "applied revise; re-auditing"
                )
                continue

            try:
                return await self._run_hard_gate(ctx, pack, pack_ref, subject)
            except ApprovalPendingError:
                raise
            except StageExecutionError as exc:
                last = exc
                self._record_feedback_text(ctx, str(exc))
                if self._regenerate is not None and attempt < self._attempts:
                    _LOG.warning(
                        f"[step_audit:{self.name}] attempt {attempt}/{self._attempts} "
                        f"rejected; regenerating — {exc}"
                    )
                    await self._regenerate.run(ctx)
                    continue
                raise

        assert last is not None
        raise last

    async def _maybe_enrich(
        self,
        ctx: HarnessRunContext,
        pack: ReviewPack,
        subject: PlanArtifactRef,
    ) -> ReviewPack:
        if self._enrich_agent is None or ctx.agent_gateway is None:
            return pack
        try:
            call = AgentCallSpec(
                agent_name=self._enrich_agent,
                input_artifact_ids=[subject.id],
                output_schema=FormDocument.model_json_schema(),
            )
            result = await ctx.agent_gateway.call(call)
            form_raw = ctx.artifact_store.get(result.output_artifact.id)
            form = FormDocument.model_validate_json(form_raw)
            return pack.model_copy(update={"form": form})
        except Exception as exc:
            _LOG.warning(
                f"[step_audit:{self.name}] form enrich skipped ({type(exc).__name__}: {exc})"
            )
            return pack

    def _auto_approve(
        self,
        ctx: HarnessRunContext,
        pack: ReviewPack,
        pack_ref: PlanArtifactRef,
    ) -> PlanArtifactRef:
        decision = ReviewDecision(
            pack_id=pack.pack_id,
            action="approve",
            decided_by="step-audit-auto",
            decided_at=datetime.now(tz=UTC),
            reason="auto policy",
        )
        ApprovalEventRecorder.record_request(ctx.event_log, ctx.run_id, self._request)
        approval = review_decision_to_approval(decision, request_id=self._request.id)
        ApprovalEventRecorder.record_decision(ctx.event_log, ctx.run_id, self._request, approval)
        ctx.artifact_store.put_json(
            kind="review_decision",
            obj=decision.model_dump(mode="json"),
            created_by=f"stage:{self.name}",
            parent_ids=[pack_ref.id],
        )
        summary = {
            "approved_intents": [self._request.intent],
            "decided_by": [decision.decided_by],
            "decision_count": 1,
            "policy": "auto",
            "pack_id": pack.pack_id,
        }
        return ctx.artifact_store.put_json(
            kind=self._result_kind,
            obj=summary,
            created_by=f"stage:{self.name}",
            parent_ids=[pack_ref.id, *self._subject_artifact_ids],
        )

    async def _run_hard_gate(
        self,
        ctx: HarnessRunContext,
        pack: ReviewPack,
        pack_ref: PlanArtifactRef,
        subject: PlanArtifactRef,
    ) -> PlanArtifactRef:
        # Nested gate keeps a distinct name so a Mode ledger cannot confuse it
        # with this StepAuditLoop instance (see approval-gate-instance-name note).
        gate = ApprovalGate(
            requests=[self._request],
            approve=self._approve,
            subject_artifact_ids=[pack_ref.id, subject.id, *self._subject_artifact_ids],
            name=f"{self.name}__gate",
            result_kind=self._result_kind,
        )
        summary = await gate.run(ctx)
        stored = (
            ctx.approval_store.granted_decision_for(self._request.id)
            if ctx.approval_store is not None
            else None
        )
        if stored is not None:
            review = approval_decision_to_review(stored, pack_id=pack.pack_id)
            ctx.artifact_store.put_json(
                kind="review_decision",
                obj=review.model_dump(mode="json"),
                created_by=f"stage:{self.name}",
                parent_ids=[pack_ref.id],
            )
        return summary

    def _consume_pending_revise(self, ctx: HarnessRunContext) -> ReviewDecision | None:
        """If services/UI left a revise :class:`ReviewDecision`, consume it once."""
        ref = ctx.artifact_store.latest_by_kind("review_decision")
        if ref is None:
            return None
        try:
            decision = ReviewDecision.model_validate_json(ctx.artifact_store.get(ref.id))
        except Exception:
            return None
        if decision.action != "revise":
            return None
        # Tombstone so the next iteration does not re-consume the same revise.
        ctx.artifact_store.put_json(
            kind="review_decision",
            obj=ReviewDecision(
                pack_id=decision.pack_id,
                action="reject",
                decided_by="step-audit-loop",
                decided_at=datetime.now(tz=UTC),
                reason="revise-consumed-internal",
                field_values=decision.field_values,
                edits=decision.edits,
            ).model_dump(mode="json"),
            created_by=f"stage:{self.name}",
            parent_ids=[ref.id],
        )
        return decision

    def _record_feedback_text(self, ctx: HarnessRunContext, text: str) -> None:
        ctx.artifact_store.put_text(
            kind=self._feedback_kind,
            text=text,
            created_by=self.name,
            parent_ids=[],
        )
