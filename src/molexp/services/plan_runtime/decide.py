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
    import asyncio
    from collections.abc import Coroutine

    from molexp.services.plan_runtime.task import PlanTask
    from molexp.workspace.run import Run

__all__ = ["apply_plan_field_values", "decide_plan_review", "map_review_action_to_approval"]


def apply_plan_field_values(run: Run, field_values: dict[str, Any]) -> None:
    """Apply review form field_values onto the run's on-disk task board.

    Currently honours ``keep_tasks`` (list of task ids to retain). Other keys
    are ignored here and remain available on the ``review_decision`` artifact.
    """
    from molexp.harness.plan import board_path, read_board, remove_task, write_board

    keep = field_values.get("keep_tasks")
    if not isinstance(keep, list) or not keep:
        return
    keep_ids = {str(x) for x in keep}
    path = board_path(Path(str(run.run_dir)))
    board = read_board(path)
    for task in list(board.tasks):
        if task.id not in keep_ids:
            try:
                board = remove_task(board, task.id)
            except Exception:
                continue
    write_board(path, board)


def _apply_plan_field_values(run: Run, field_values: dict[str, Any]) -> None:
    apply_plan_field_values(run, field_values)


#: Strong refs to fire-and-forget resume tasks so the running loop cannot GC
#: them mid-flight (RUF006); each task removes itself on completion.
_RESUME_TASKS: set[asyncio.Task[object]] = set()


def _schedule_resume(result: object) -> None:
    """Fire-and-forget a resume coroutine returned by ``propose_plan_patch``.

    ``propose_plan_patch`` returns the ``resume_main`` coroutine; this decide
    path is sync, so schedule it on the running loop (the async server route) or
    run it to completion when called outside one. A non-awaitable (a
    monkeypatched test double) is a no-op.
    """
    import asyncio
    import inspect
    from typing import cast

    if not inspect.isawaitable(result):
        return
    coro = cast("Coroutine[object, object, object]", result)
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(coro)
    else:
        task = loop.create_task(coro)
        _RESUME_TASKS.add(task)
        task.add_done_callback(_RESUME_TASKS.discard)


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

    if request.scope == "intervention_request":
        # Phase-2 resume: route into the named codegen subagent (or the plan-patch
        # fallback), never the phase-1 approval-store grant path. propose_plan_patch
        # is looked up via the module so a test monkeypatch resolves at call time.
        from molexp.services.plan_runtime import resume_scope

        if review.field_values.get("fallback") == "plan_patch":
            patch: dict[str, Any] = {
                "field_values": dict(review.field_values),
                "reason": review.reason,
                "edits": review.edits,
            }
            _schedule_resume(resume_scope.propose_plan_patch(run=run, patch=patch))
            notify_approvals_changed()
            return approval
        if request.target_agent_id is None:
            raise ValueError(
                f"intervention_request {request.id!r} has no target_agent_id and no "
                "plan_patch fallback — a task intervention with no named subagent "
                "target is a caller defect (no silent fallback)"
            )
        payload: dict[str, Any] = {
            "field_values": dict(review.field_values),
            "reason": review.reason,
            "edits": review.edits,
        }
        if task is not None:
            task.resume_intervention(target_agent_id=request.target_agent_id, payload=payload)
        notify_approvals_changed()
        return approval

    if review.action == "approve":
        # Ensure grant path uses approve mapping even if caller passed wrong fields.
        approval = review_decision_to_approval(review, request_id=request.id)
        # Persist operator field choices (keep_tasks, notes, priority) as an
        # artifact so resume / freeze can honor them — not only on revise.
        if review.field_values:
            _apply_plan_field_values(run, review.field_values)

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
    if review.field_values:
        _apply_plan_field_values(run, review.field_values)
    if task is not None:
        task.resume()
    notify_approvals_changed()
    return approval
