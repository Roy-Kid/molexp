"""Approvals inbox — the ONE pending-approvals surface over both gate families.

``GET /api/approvals`` aggregates the pending :class:`ApprovalRequest`\\ s of
every suspended (``waiting_approval``) plan task **and** curate task into one
list; ``POST /api/approvals/{task_kind}/{task_id}/decisions`` persists an
operator decision into that run's approval store (``decided_by="ui-operator"``),
records the decision event, and resumes the task — the stage ledger skips
completed stages and the gate passes on the stored grant. ``GET
/api/approvals/events`` is the SSE ping stream (a ``changed`` event per suspend
or decision; the UI refetches the list).

The inbox is read-through: pending state lives on the suspended tasks (which
carry the requests their gate raised) and decisions live in each run's
``harness.sqlite`` approval store — this module owns no state of its own.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, model_validator

from molexp.server.dependencies import get_workspace

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from molexp.harness.schemas import ApprovalRequest
    from molexp.workspace import Workspace

__all__ = ["router"]

router = APIRouter(prefix="/approvals", tags=["approvals"])

# Canonical SSE headers (same idiom as routes/molq.py `_SSE_HEADERS`).
_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}

TaskKind = Literal["plan", "curate"]
ReviewActionWire = Literal["approve", "reject", "revise"]


class PendingApprovalItem(BaseModel):
    """One pending request awaiting an operator decision."""

    taskKind: TaskKind
    taskId: str
    runId: str
    projectId: str
    experimentId: str
    requestId: str
    intent: str
    reason: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    requestedAt: str
    #: Text of the gated content (spec fields / generated source) so the
    #: operator reviews the actual artifact, not just the reason line.
    preview: str = ""
    #: Latest review_pack artifact id when StepAuditLoop has written one.
    packId: str | None = None
    #: Structured form for UI ReviewSurface (FormDocument JSON shape).
    formDocument: dict[str, Any] | None = None
    #: Resume-routing discriminator (``approval_gate`` | ``intervention_request``);
    #: the UI shows a phase-2 intervention differently from a phase-1 gate.
    scope: str = "approval_gate"
    #: Named phase-2 codegen subagent for an ``intervention_request`` (else ``None``).
    targetAgentId: str | None = None


class PendingApprovalsResponse(BaseModel):
    """The inbox: every pending request across both task kinds."""

    items: list[PendingApprovalItem]
    total: int


class ApprovalDecisionRequest(BaseModel):
    """Operator decision — ReviewDecision-shaped wire body.

    Preferred field is ``action`` (approve|reject|revise). ``granted`` remains
    as a **deprecated** boolean alias for approve/reject only (migration for
    older UI clients that only knew grant/deny).
    """

    requestId: str
    action: ReviewActionWire | None = None
    fieldValues: dict[str, Any] = Field(default_factory=dict)
    reason: str | None = None
    edits: dict[str, Any] | None = None
    granted: bool | None = None

    @model_validator(mode="after")
    def _normalize_action(self) -> ApprovalDecisionRequest:
        if self.action is None and self.granted is None:
            raise ValueError("either action or granted is required")
        if self.action is None and self.granted is not None:
            object.__setattr__(self, "action", "approve" if self.granted else "reject")
            return self
        if self.action is not None and self.granted is not None:
            if self.action == "revise":
                raise ValueError("action=revise cannot be combined with granted")
            mapped = "approve" if self.granted else "reject"
            if self.action != mapped:
                raise ValueError(f"action={self.action!r} conflicts with granted={self.granted!r}")
        return self


class ApprovalDecisionResponse(BaseModel):
    """Post-decision task summary."""

    taskKind: TaskKind
    taskId: str
    status: str


def _experiment_ids(task: Any) -> tuple[str, str]:  # noqa: ANN401 — plan/curate task duck-type
    experiment = task.experiment
    return experiment.project.id, experiment.id


def _preview_for(kind: TaskKind, task: Any, intent: str) -> str:  # noqa: ANN401
    """Best-effort gated-content preview — never blocks the inbox listing."""
    if kind != "plan":
        return ""
    from molexp.services.plan_runtime.preview import render_approval_preview

    try:
        return render_approval_preview(task.run, intent)
    except Exception:  # a broken preview must not hide the pending decision
        return ""


def _pack_fields(
    kind: TaskKind,
    task: Any,  # noqa: ANN401 — plan/curate task duck-type
) -> tuple[str | None, dict[str, Any] | None]:
    """Best-effort packId + formDocument from the run's review_pack artifact."""
    if kind != "plan":
        return None, None
    try:
        from molexp.services.plan_runtime.preview import build_review_pack

        pack = build_review_pack(task.run, "final_report")
        form = pack.form.model_dump(mode="json")
        return pack.pack_id, form
    except Exception:
        return None, None


def _items_for(kind: TaskKind, tasks: list[Any]) -> list[PendingApprovalItem]:
    items: list[PendingApprovalItem] = []
    for task in tasks:
        project_id, experiment_id = _experiment_ids(task)
        pack_id, form_doc = _pack_fields(kind, task)
        for request in task.pending_requests:
            items.append(
                PendingApprovalItem(
                    taskKind=kind,
                    taskId=task.task_id,
                    runId=task.run_id,
                    projectId=project_id,
                    experimentId=experiment_id,
                    requestId=request.id,
                    intent=request.intent,
                    reason=request.reason,
                    metadata=dict(request.metadata),
                    requestedAt=request.created_at.isoformat(),
                    preview=_preview_for(kind, task, request.intent),
                    packId=pack_id,
                    formDocument=form_doc,
                    scope=request.scope,
                    targetAgentId=request.target_agent_id,
                )
            )
    return items


@router.get("", response_model=PendingApprovalsResponse)
async def list_pending_approvals(
    workspace: Workspace = Depends(get_workspace),
) -> PendingApprovalsResponse:
    """List every pending approval across suspended plan + curate tasks."""
    from molexp.server.deps.curate_runtime import get_curate_runtime
    from molexp.server.deps.plan_runtime import get_plan_runtime

    root = str(workspace.root)
    items = _items_for("plan", get_plan_runtime().pending_approvals(root))
    items += _items_for("curate", get_curate_runtime().pending_approvals(root))
    return PendingApprovalsResponse(items=items, total=len(items))


def _find_task(kind: TaskKind, task_id: str, workspace_root: str) -> Any:  # noqa: ANN401
    from molexp.server.deps.curate_runtime import get_curate_runtime
    from molexp.server.deps.plan_runtime import get_plan_runtime

    registry = get_plan_runtime() if kind == "plan" else get_curate_runtime()
    task = registry.get(workspace_root, task_id)
    if task is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"unknown {kind} task {task_id!r}")
    return task


def _pending_request(task: Any, request_id: str) -> ApprovalRequest:  # noqa: ANN401
    for request in task.pending_requests:
        if request.id == request_id:
            return request
    raise HTTPException(
        status.HTTP_404_NOT_FOUND,
        f"task {task.task_id!r} has no pending request {request_id!r}",
    )


@router.post("/{task_kind}/{task_id}/decisions", response_model=ApprovalDecisionResponse)
async def decide_approval(
    task_kind: TaskKind,
    task_id: str,
    request: ApprovalDecisionRequest,
    workspace: Workspace = Depends(get_workspace),
) -> ApprovalDecisionResponse:
    """Record a ReviewDecision-shaped answer and resume/reject the task.

    Plan tasks delegate to :func:`molexp.services.plan_runtime.decide_plan_review`.
    Curate tasks keep the binary store path (no ReviewPack yet).
    """
    from molexp.harness.schemas import ApprovalDecision, ReviewDecision
    from molexp.services.approval_notify import notify_approvals_changed

    task = _find_task(task_kind, task_id, str(workspace.root))
    if task.status != "waiting_approval":
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"{task_kind} task {task_id!r} is {task.status!r}, not waiting_approval",
        )
    pending = _pending_request(task, request.requestId)
    assert request.action is not None  # normalized by model_validator

    if task_kind == "plan":
        from molexp.services.plan_runtime.decide import decide_plan_review
        from molexp.services.plan_runtime.preview import build_review_pack

        try:
            pack = build_review_pack(task.run, pending.intent)
            pack_id = pack.pack_id
        except Exception:
            pack_id = f"pack-{pending.id}"
        review = ReviewDecision(
            pack_id=pack_id,
            action=request.action,
            decided_by="ui-operator",
            decided_at=datetime.now(tz=UTC),
            reason=request.reason,
            field_values=dict(request.fieldValues),
            edits=request.edits,
        )
        decide_plan_review(run=task.run, request=pending, decision=review, task=task)
        return ApprovalDecisionResponse(taskKind=task_kind, taskId=task_id, status=task.status)

    # Curate: binary grant/reject only (revise maps to reject for now).
    from molexp.harness import SQLiteApprovalStore, SQLiteEventLog
    from molexp.harness.policy.event_log import ApprovalEventRecorder

    granted = request.action == "approve"
    decision = ApprovalDecision(
        request_id=pending.id,
        granted=granted,
        decided_by="ui-operator",
        decided_at=datetime.now(tz=UTC),
        reason=request.reason,
    )
    run_dir = task.run.run_dir
    db_path = run_dir / "harness.sqlite"
    store = SQLiteApprovalStore(path=db_path)
    store.record_pending(task.run_id, pending)
    store.record_decision(decision)
    ApprovalEventRecorder.record_decision(
        SQLiteEventLog(path=db_path), task.run_id, pending, decision
    )
    if granted:
        task.resume()
    else:
        task.mark_rejected(request.reason or "rejected by operator")
    notify_approvals_changed()
    return ApprovalDecisionResponse(taskKind=task_kind, taskId=task_id, status=task.status)


@router.get("/events")
async def stream_approval_events() -> StreamingResponse:
    """SSE: one ``changed`` event per suspend/decision — the UI refetch signal."""
    from molexp.services.approval_notify import subscribe_approvals_changed

    async def _generate() -> AsyncIterator[str]:
        yield "event: changed\ndata: connected\n\n"
        async for _ in subscribe_approvals_changed():
            yield "event: changed\ndata: changed\n\n"

    return StreamingResponse(_generate(), media_type="text/event-stream", headers=_SSE_HEADERS)
