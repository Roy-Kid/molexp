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
from pydantic import BaseModel, Field

from molexp.server.dependencies import get_workspace

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from molexp.harness.schemas import ApprovalDecision, ApprovalRequest
    from molexp.workspace import Workspace

__all__ = ["router"]

router = APIRouter(prefix="/approvals", tags=["approvals"])

# Canonical SSE headers (same idiom as routes/molq.py `_SSE_HEADERS`).
_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}

TaskKind = Literal["plan", "curate"]


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


class PendingApprovalsResponse(BaseModel):
    """The inbox: every pending request across both task kinds."""

    items: list[PendingApprovalItem]
    total: int


class ApprovalDecisionRequest(BaseModel):
    """Operator decision on one pending request."""

    requestId: str
    granted: bool
    reason: str | None = None


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


def _items_for(kind: TaskKind, tasks: list[Any]) -> list[PendingApprovalItem]:
    items: list[PendingApprovalItem] = []
    for task in tasks:
        project_id, experiment_id = _experiment_ids(task)
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


def _record_decision(
    run_dir: Path, run_id: str, request: ApprovalRequest, decision: ApprovalDecision
) -> None:
    """Persist the decision (store) and record it (event log) for *run_id*."""
    from molexp.harness import SQLiteApprovalStore, SQLiteEventLog
    from molexp.harness.policy.event_log import ApprovalEventRecorder

    db_path = run_dir / "harness.sqlite"
    approval_store = SQLiteApprovalStore(path=db_path)
    approval_store.record_pending(run_id, request)
    approval_store.record_decision(decision)
    ApprovalEventRecorder.record_decision(SQLiteEventLog(path=db_path), run_id, request, decision)


@router.post("/{task_kind}/{task_id}/decisions", response_model=ApprovalDecisionResponse)
async def decide_approval(
    task_kind: TaskKind,
    task_id: str,
    request: ApprovalDecisionRequest,
    workspace: Workspace = Depends(get_workspace),
) -> ApprovalDecisionResponse:
    """Grant or reject one pending request, then resume (or fail) the task."""
    from molexp.harness.schemas import ApprovalDecision
    from molexp.services.approval_notify import notify_approvals_changed

    task = _find_task(task_kind, task_id, str(workspace.root))
    if task.status != "waiting_approval":
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"{task_kind} task {task_id!r} is {task.status!r}, not waiting_approval",
        )
    pending = _pending_request(task, request.requestId)

    decision = ApprovalDecision(
        request_id=pending.id,
        granted=request.granted,
        decided_by="ui-operator",
        decided_at=datetime.now(tz=UTC),
        reason=request.reason,
    )
    run_dir = task.run.run_dir
    _record_decision(run_dir, task.run_id, pending, decision)

    if request.granted:
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
