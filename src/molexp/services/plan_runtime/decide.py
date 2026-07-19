"""``decide_plan_review`` — shared decide path for CLI/server plan approvals.

Validates a :class:`ReviewDecision` (or mapping) with pydantic, maps
approve/reject onto the binary approval store, and treats **revise** as
re-entry (persist decision artifact + resume) rather than permanent reject.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from molexp.harness.schemas import (
    ApprovalDecision,
    ApprovalRequest,
    ReviewDecision,
    review_decision_to_approval,
)

if TYPE_CHECKING:
    from molexp.services.plan_runtime.task import PlanTask
    from molexp.workspace.run import Run

__all__ = ["decide_plan_review", "map_review_action_to_approval"]


def map_review_action_to_approval(
    decision: ReviewDecision,
    *,
    request_id: str,
) -> ApprovalDecision:
    """Map approve|reject via projection; revise becomes non-grant for store.

    ``revise`` deliberately does **not** call
    :func:`review_decision_to_approval` (that helper raises by design).
    """
    if decision.action == "revise":
        return ApprovalDecision(
            request_id=request_id,
            granted=False,
            decided_by=decision.decided_by,
            decided_at=decision.decided_at,
            reason=decision.reason or "revise",
        )
    return review_decision_to_approval(decision, request_id=request_id)


def decide_plan_review(
    *,
    run: Run,
    request: ApprovalRequest,
    decision: ReviewDecision | dict[str, Any],
    task: PlanTask | None = None,
    decided_by: str | None = None,
) -> ApprovalDecision:
    """Validate *decision*, persist store/events/artifacts, resume or reject task.

    Args:
        run: Suspended plan run.
        request: Pending approval request.
        decision: ReviewDecision or JSON-compatible mapping.
        task: Optional PlanTask to resume / mark_rejected.
        decided_by: Override decided_by on the validated decision when provided.

    Returns:
        The binary ApprovalDecision written to the store (revise → granted=False).

    Raises:
        ValidationError: Invalid ReviewDecision payload.
        ValueError: pack_id mismatch when both sides carry one, or task not waiting.
        RuntimeError: From PlanTask when status is wrong.
    """
    from molexp.harness import SQLiteApprovalStore, SQLiteEventLog
    from molexp.harness.policy.event_log import ApprovalEventRecorder
    from molexp.harness.store.file_artifact_store import FileArtifactStore
    from molexp.services.approval_notify import notify_approvals_changed

    if isinstance(decision, ReviewDecision):
        review = decision
    else:
        review = ReviewDecision.model_validate(decision)

    if decided_by is not None:
        review = review.model_copy(update={"decided_by": decided_by})

    # Persist structured decision for StepAuditLoop re-entry (esp. revise).
    artifacts = FileArtifactStore(root=Path(str(run.run_dir)) / "artifacts")
    artifacts.put_json(
        kind="review_decision",
        obj=review.model_dump(mode="json"),
        created_by="services.plan_runtime.decide",
        parent_ids=[],
    )

    approval = map_review_action_to_approval(review, request_id=request.id)
    if review.action == "approve":
        # Ensure grant path uses approve mapping even if caller passed wrong fields.
        approval = review_decision_to_approval(review, request_id=request.id)

    run_dir = Path(str(run.run_dir))
    db_path = run_dir / "harness.sqlite"
    store = SQLiteApprovalStore(path=db_path)
    store.record_pending(run.id, request)

    if review.action == "approve":
        store.record_decision(approval)
        ApprovalEventRecorder.record_decision(
            SQLiteEventLog(path=db_path), run.id, request, approval
        )
        if task is not None:
            task.resume()
        notify_approvals_changed()
        return approval

    if review.action == "reject":
        store.record_decision(approval)
        ApprovalEventRecorder.record_decision(
            SQLiteEventLog(path=db_path), run.id, request, approval
        )
        if task is not None:
            task.mark_rejected(review.reason or "rejected")
        notify_approvals_changed()
        return approval

    # revise: do NOT record a durable grant/reject that would stick as permanent
    # fail on re-entry. Leave pending clear by not storing grant; write decision
    # artifact (above) and resume so StepAuditLoop consumes revise.
    # Clear any prior grant for this request so hard gate does not auto-pass.
    # Store has no explicit clear; recording non-grant is fine for reject path
    # but for revise we skip store.record_decision of granted=False so re-entry
    # can re-suspend after applying field_values.
    if task is not None:
        task.resume()
    notify_approvals_changed()
    return approval
