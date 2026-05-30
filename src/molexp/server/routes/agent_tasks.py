"""User-facing Agent Task routes.

This module is a compatibility layer over the current AgentSession runtime.
It gives the product/UI a stable ``AgentTask`` surface while the lower-level
session store, event persistence, and review model are migrated separately.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from ..dependencies import get_workspace
from ..schemas import (
    AgentEvent,
    AgentSessionResponse,
    AgentTaskListResponse,
    AgentTaskResponse,
    ApprovalRespondRequest,
    GoalCreateRequest,
    MessageResponse,
    PlanDecisionRequest,
    UserMessageCreateRequest,
)
from . import agent as agent_routes
from .agent_task_store import (
    PersistedAgentTask,
    list_agent_task_metadata,
    read_agent_task_metadata,
    write_agent_task_metadata,
)
from .review_store import (
    ensure_plan_review,
    list_review_metadata,
    plan_review_id,
    read_review_metadata,
    resolve_review,
)

router = APIRouter(prefix="/agent-tasks", tags=["agent-tasks"])


def _title_from_goal(goal: str) -> str:
    compact = " ".join(goal.split())
    if not compact:
        return "Untitled agent task"
    if len(compact) <= 72:
        return compact
    return f"{compact[:69].rstrip()}..."


def _task_from_session(
    session: AgentSessionResponse,
    *,
    task_id: str | None = None,
    persisted: PersistedAgentTask | None = None,
) -> AgentTaskResponse:
    updated_at = session.stats.completedAt or session.stats.startedAt or session.createdAt
    return AgentTaskResponse(
        taskId=(persisted.task_id if persisted is not None else task_id or session.sessionId),
        title=(
            persisted.title
            if persisted is not None and persisted.title
            else _title_from_goal(session.goalDescription)
        ),
        goal=session.goalDescription,
        status=session.status,
        createdAt=persisted.created_at if persisted is not None else session.createdAt,
        updatedAt=updated_at,
        sessionId=session.sessionId,
        events=session.events,
        stats=session.stats,
        planMode=session.planMode,
        skillId=session.skillId,
    )


def _task_from_metadata(task: PersistedAgentTask) -> AgentTaskResponse:
    return AgentTaskResponse(
        taskId=task.task_id,
        title=task.title,
        goal=task.goal,
        status=task.status,
        createdAt=task.created_at,
        updatedAt=task.updated_at,
        sessionId=task.session_id,
        planMode=task.plan_mode,
        skillId=task.skill_id,
    )


def _workspace_root(workspace) -> str | None:  # noqa: ANN001
    root = getattr(workspace, "root", None)
    return str(root) if root is not None else None


def _persist_task_response(workspace, task: AgentTaskResponse) -> None:  # noqa: ANN001
    root = _workspace_root(workspace)
    if root is None:
        return
    write_agent_task_metadata(
        root,
        PersistedAgentTask(
            task_id=task.taskId,
            session_id=task.sessionId,
            title=task.title,
            goal=task.goal,
            status=task.status,
            created_at=task.createdAt,
            updated_at=task.updatedAt,
            plan_mode=task.planMode,
            skill_id=task.skillId,
        ),
    )


def _sync_review_items_for_task(workspace, task: AgentTaskResponse) -> AgentTaskResponse:  # noqa: ANN001
    root = _workspace_root(workspace)
    if root is None:
        return task
    has_pending_review = False
    for event in task.events:
        if event.type != "PlanCreatedEvent":
            continue
        request_id = event.payload.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            continue
        plan_markdown = event.payload.get("plan_markdown")
        workflow_preview = event.payload.get("workflow_preview")
        review = ensure_plan_review(
            root,
            task_id=task.taskId,
            session_id=task.sessionId,
            task_title=task.title,
            request_id=request_id,
            plan_markdown=plan_markdown if isinstance(plan_markdown, str) else "",
            workflow_preview=workflow_preview if isinstance(workflow_preview, dict) else {},
            created_at=event.ts,
        )
        has_pending_review = has_pending_review or review.status == "pending"
    if has_pending_review and task.status == "running":
        task.status = "waiting_for_review"
    return task


def _resolve_plan_review_for_decision(
    workspace,  # noqa: ANN001
    task_id: str,
    request: PlanDecisionRequest,
) -> None:
    root = _workspace_root(workspace)
    if root is None:
        return
    review = read_review_metadata(root, plan_review_id(task_id, request.request_id))
    if review is None or review.status != "pending":
        return
    resolve_review(
        root,
        review,
        status="approved" if request.approved else "rejected",
        comment=request.feedback,
    )


def _persisted_for_session(workspace, session_id: str) -> PersistedAgentTask | None:  # noqa: ANN001
    root = _workspace_root(workspace)
    if root is None:
        return None
    # Today task_id == session_id.  The scan keeps the wrapper ready for a
    # future task id that differs from the runtime session id.
    direct = read_agent_task_metadata(root, session_id)
    if direct is not None:
        return direct
    for task in list_agent_task_metadata(root):
        if task.session_id == session_id:
            return task
    return None


def _session_id_for_task(workspace, task_id: str) -> str:  # noqa: ANN001
    root = _workspace_root(workspace)
    if root is None:
        return task_id
    task = read_agent_task_metadata(root, task_id)
    return task.session_id if task is not None else task_id


def _has_pending_review(workspace, task_id: str) -> bool:  # noqa: ANN001
    root = _workspace_root(workspace)
    if root is None:
        return False
    return any(
        review.task_id == task_id and review.status == "pending"
        for review in list_review_metadata(root)
    )


@router.post("", response_model=AgentTaskResponse)
async def create_agent_task(
    request: GoalCreateRequest,
    workspace=Depends(get_workspace),  # noqa: ANN001
) -> AgentTaskResponse:
    """Create a user-facing agent task.

    Today this starts exactly one runtime session, but task identity is already
    separate from the runtime session id.
    """
    session = await agent_routes.create_session(request, workspace=workspace)
    task = _task_from_session(session, task_id=f"task-{uuid.uuid4().hex[:12]}")
    task = _sync_review_items_for_task(workspace, task)
    _persist_task_response(workspace, task)
    return task


@router.get("", response_model=AgentTaskListResponse)
def list_agent_tasks(workspace=Depends(get_workspace)) -> AgentTaskListResponse:  # noqa: ANN001
    """List active and historical agent tasks."""
    sessions = agent_routes.list_sessions(workspace=workspace)
    tasks: list[AgentTaskResponse] = []
    seen_task_ids: set[str] = set()
    for session in sessions.sessions:
        task = _task_from_session(
            session,
            persisted=_persisted_for_session(workspace, session.sessionId),
        )
        task = _sync_review_items_for_task(workspace, task)
        if _has_pending_review(workspace, task.taskId) and task.status == "running":
            task.status = "waiting_for_review"
        _persist_task_response(workspace, task)
        tasks.append(task)
        seen_task_ids.add(task.taskId)
    root = _workspace_root(workspace)
    if root is not None:
        for persisted in list_agent_task_metadata(root):
            if persisted.task_id in seen_task_ids:
                continue
            task = _task_from_metadata(persisted)
            if _has_pending_review(workspace, task.taskId) and task.status == "running":
                task.status = "waiting_for_review"
            tasks.append(task)
    tasks.sort(key=lambda task: task.updatedAt or task.createdAt, reverse=True)
    return AgentTaskListResponse(tasks=tasks, total=len(tasks))


@router.get("/{task_id}", response_model=AgentTaskResponse)
def get_agent_task(
    task_id: str,
    workspace=Depends(get_workspace),  # noqa: ANN001
) -> AgentTaskResponse:
    """Get a single agent task by task id."""
    session_id = _session_id_for_task(workspace, task_id)
    try:
        session = agent_routes.get_session(session_id, workspace=workspace)
    except HTTPException:
        root = _workspace_root(workspace)
        if root is not None:
            persisted = read_agent_task_metadata(root, task_id)
            if persisted is not None:
                return _task_from_metadata(persisted)
        raise
    task = _task_from_session(
        session,
        persisted=_persisted_for_session(workspace, session.sessionId),
    )
    task = _sync_review_items_for_task(workspace, task)
    if _has_pending_review(workspace, task.taskId) and task.status == "running":
        task.status = "waiting_for_review"
    _persist_task_response(workspace, task)
    return task


@router.get(
    "/{task_id}/events",
    responses={
        200: {
            "model": AgentEvent,
            "description": (
                "Server-Sent Events stream; each `data:` frame is one AgentEvent "
                "(discriminated on `kind`), terminated by a `done` control frame."
            ),
        }
    },
)
async def stream_agent_task_events(
    task_id: str,
    workspace=Depends(get_workspace),  # noqa: ANN001
) -> StreamingResponse:
    """Stream task activity events.

    Delegates to the existing session event stream until task events are
    persisted independently.
    """
    return await agent_routes.stream_events(
        _session_id_for_task(workspace, task_id), workspace=workspace
    )


@router.post("/{task_id}/approve")
async def respond_agent_task_approval(
    task_id: str,
    request: ApprovalRespondRequest,
    workspace=Depends(get_workspace),  # noqa: ANN001
) -> dict:
    """Respond to a runtime approval request for this task."""
    return await agent_routes.respond_approval(
        _session_id_for_task(workspace, task_id), request, workspace=workspace
    )


@router.post("/{task_id}/plan-decision", response_model=MessageResponse)
async def respond_agent_task_plan(
    task_id: str,
    request: PlanDecisionRequest,
    workspace=Depends(get_workspace),  # noqa: ANN001
) -> MessageResponse:
    """Approve or reject the current task plan."""
    session_id = _session_id_for_task(workspace, task_id)
    response = await agent_routes.respond_plan(session_id, request, workspace=workspace)
    _resolve_plan_review_for_decision(workspace, task_id, request)
    get_agent_task(task_id, workspace)
    return response


@router.post("/{task_id}/messages", response_model=MessageResponse)
async def post_agent_task_message(
    task_id: str,
    request: UserMessageCreateRequest,
    workspace=Depends(get_workspace),  # noqa: ANN001
) -> MessageResponse:
    """Send a user message to a running agent task."""
    session_id = _session_id_for_task(workspace, task_id)
    response = await agent_routes.post_user_message(session_id, request, workspace=workspace)
    get_agent_task(task_id, workspace)
    return response
