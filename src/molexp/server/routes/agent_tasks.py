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
    GoalCreateRequest,
    MessageResponse,
    UserMessageCreateRequest,
)
from . import agent as agent_routes
from .agent_task_store import (
    PersistedAgentTask,
    list_agent_task_metadata,
    read_agent_task_metadata,
    write_agent_task_metadata,
)

router = APIRouter(prefix="/agent-tasks", tags=["agent-tasks"])

# Maximum length (in characters) of an auto-derived task title; longer
# goals are truncated with a "..." suffix within this budget.
_TITLE_MAX_CHARS = 72
_TITLE_ELLIPSIS = "..."


def _title_from_goal(goal: str) -> str:
    compact = " ".join(goal.split())
    if not compact:
        return "Untitled agent task"
    if len(compact) <= _TITLE_MAX_CHARS:
        return compact
    clipped = compact[: _TITLE_MAX_CHARS - len(_TITLE_ELLIPSIS)].rstrip()
    return f"{clipped}{_TITLE_ELLIPSIS}"


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
        _persist_task_response(workspace, task)
        tasks.append(task)
        seen_task_ids.add(task.taskId)
    root = _workspace_root(workspace)
    if root is not None:
        for persisted in list_agent_task_metadata(root):
            if persisted.task_id in seen_task_ids:
                continue
            tasks.append(_task_from_metadata(persisted))
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
