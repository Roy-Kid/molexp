"""User-facing Agent Task routes.

This module is a compatibility layer over the current AgentSession runtime.
It gives the product/UI a stable ``AgentTask`` surface while the lower-level
session store, event persistence, and review model are migrated separately.
"""

from __future__ import annotations

import uuid
from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from molexp.services.agent_task_store import (
    PersistedAgentTask,
    append_agent_task_events,
    delete_agent_task,
    list_agent_task_metadata,
    read_agent_task_events,
    read_agent_task_metadata,
    write_agent_task_metadata,
)

from ..dependencies import get_workspace
from ..schemas import (
    AgentEvent,
    AgentSessionResponse,
    AgentTaskListResponse,
    AgentTaskResponse,
    GoalCreateRequest,
    MessageResponse,
    SessionEventResponse,
    UserMessageCreateRequest,
)
from . import agent as agent_routes

if TYPE_CHECKING:
    from molexp.workspace import Workspace

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
        goal=persisted.goal if persisted is not None else session.goalDescription,
        status=session.status,
        createdAt=persisted.created_at if persisted is not None else session.createdAt,
        updatedAt=updated_at,
        sessionId=session.sessionId,
        events=session.events,
        stats=session.stats,
        planMode=session.planMode,
        activeMode=persisted.active_mode if persisted is not None else "chat",
        activeTurnId=persisted.active_turn_id if persisted is not None else None,
        activePlanTaskId=(persisted.active_plan_task_id if persisted is not None else None),
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
        activeMode=task.active_mode,
        activeTurnId=task.active_turn_id,
        activePlanTaskId=task.active_plan_task_id,
        skillId=task.skill_id,
    )


def _workspace_root(workspace) -> str | None:  # noqa: ANN001
    root = getattr(workspace, "root", None)
    return str(root) if root is not None else None


def _persist_task_response(
    workspace: Workspace,
    task: AgentTaskResponse,
    *,
    project_id: str | None = None,
    experiment_id: str | None = None,
    run_id: str | None = None,
    persisted: PersistedAgentTask | None = None,
) -> None:
    """Write the task's on-disk metadata.

    The mount scope either arrives explicitly (task creation) or rides
    *persisted* (read paths refreshing live-session state) — a refresh must
    never wipe the stored scope.
    """
    if persisted is not None:
        project_id = project_id or persisted.project_id
        experiment_id = experiment_id or persisted.experiment_id
        run_id = run_id or persisted.run_id
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
            active_mode=task.activeMode,
            active_turn_id=task.activeTurnId,
            active_plan_task_id=(persisted.active_plan_task_id if persisted is not None else None),
            pending_plan_draft=(persisted.pending_plan_draft if persisted is not None else None),
            skill_id=task.skillId,
            project_id=project_id,
            experiment_id=experiment_id,
            run_id=run_id,
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


def _turn_id() -> str:
    return f"turn-{uuid.uuid4().hex[:12]}"


def _wire_events(raw_events: list[dict]) -> list[SessionEventResponse]:
    return [
        SessionEventResponse(
            type=str(event.get("type", "")),
            ts=str(event.get("ts", "")),
            payload=event.get("payload") or {},
        )
        for event in raw_events
        if isinstance(event, dict)
    ]


def _merge_persisted_events(
    task: AgentTaskResponse,
    workspace: Workspace,
) -> AgentTaskResponse:
    root = _workspace_root(workspace)
    if root is None:
        return task
    persisted_events = _wire_events(read_agent_task_events(root, task.taskId))
    if not persisted_events:
        return task
    seen = {(event.type, event.ts, repr(event.payload)) for event in persisted_events}
    merged = [*persisted_events]
    merged.extend(
        event for event in task.events if (event.type, event.ts, repr(event.payload)) not in seen
    )
    merged.sort(key=lambda event: event.ts)
    return task.model_copy(update={"events": merged})


def _launch_plan_turn(
    *,
    workspace: Workspace,
    task: PersistedAgentTask,
    draft: str,
    turn_id: str,
) -> None:
    """Start one nine-stage plan execution inside an existing AgentTask."""
    if not task.project_id or not task.experiment_id:
        raise ValueError("plan turn requires project and experiment context")
    from molexp.server.routes.plan_tasks import PlanTaskCreateRequest, start_plan_task

    started = start_plan_task(
        project_id=task.project_id,
        experiment_id=task.experiment_id,
        request=PlanTaskCreateRequest(draft=draft, execute=False),
        workspace=workspace,
        record_task_id=task.task_id,
        turn_id=turn_id,
        supersedes_run_id=task.run_id,
    )
    root = _workspace_root(workspace)
    if root is not None:
        refreshed = read_agent_task_metadata(root, task.task_id) or task
        write_agent_task_metadata(
            root,
            replace(
                refreshed,
                status="running",
                plan_mode=True,
                active_mode="plan",
                active_turn_id=turn_id,
                active_plan_task_id=started.taskId,
                run_id=started.runId,
                pending_plan_draft=None,
                updated_at=datetime.now(UTC).isoformat(),
            ),
        )


def _resolve_experiment_context(workspace: Workspace, answer: str) -> tuple[str, str] | None:
    """Resolve `project / experiment`, ids, or a unique experiment name."""
    normalized = answer.strip().casefold()
    if not normalized:
        return None
    project_hint = ""
    experiment_hint = normalized
    for separator in ("/", ":"):
        if separator in normalized:
            project_hint, experiment_hint = (
                part.strip() for part in normalized.split(separator, 1)
            )
            break
    matches: list[tuple[str, str]] = []
    for project in workspace.list_projects():
        if project_hint and project_hint not in {project.id.casefold(), project.name.casefold()}:
            continue
        for experiment in project.list_experiments():
            if experiment_hint in {experiment.id.casefold(), experiment.name.casefold()}:
                matches.append((project.id, experiment.id))
    return matches[0] if len(matches) == 1 else None


@router.post("", response_model=AgentTaskResponse)
async def create_agent_task(
    request: GoalCreateRequest,
    workspace=Depends(get_workspace),  # noqa: ANN001
) -> AgentTaskResponse:
    """Create a user-facing agent task.

    The task is the stable conversation container. Each turn is dispatched to
    either the interactive agent or the nine-stage Planning Agent.
    """
    requested_mode = "plan" if request.plan_mode else request.mode
    task_id = f"task-{uuid.uuid4().hex[:12]}"
    turn_id = _turn_id()
    if requested_mode == "chat":
        session = await agent_routes.create_session(request, workspace=workspace)
        task = _task_from_session(session, task_id=task_id).model_copy(
            update={"activeMode": "chat", "activeTurnId": turn_id}
        )
    else:
        now = datetime.now(UTC).isoformat()
        needs_context = not request.project_id or not request.experiment_id
        task = AgentTaskResponse(
            taskId=task_id,
            title=_title_from_goal(request.description),
            goal=request.description,
            status="awaiting_user" if needs_context else "running",
            createdAt=now,
            updatedAt=now,
            sessionId=task_id,
            planMode=True,
            activeMode="plan",
            activeTurnId=turn_id,
            skillId=request.skill_id,
        )
    # The mount scope (vision-loop-11) persists with the task so a re-attach
    # rebuilds the same context block verbatim.
    _persist_task_response(
        workspace,
        task,
        project_id=request.project_id,
        experiment_id=request.experiment_id,
        run_id=request.run_id,
    )
    if requested_mode == "plan":
        root = _workspace_root(workspace)
        persisted = read_agent_task_metadata(root, task_id) if root is not None else None
        if persisted is not None and (not persisted.project_id or not persisted.experiment_id):
            append_agent_task_events(
                root,
                task_id,
                [
                    {
                        "type": "clarification_required",
                        "ts": datetime.now(UTC).isoformat(),
                        "payload": {
                            "request_id": f"context-{turn_id}",
                            "questions": (
                                "Which project and experiment should this plan belong to? "
                                "Reply with `project / experiment` or an experiment id."
                            ),
                            "context_kind": "experiment",
                            "turn_id": turn_id,
                            "mode": "plan",
                        },
                    }
                ],
            )
            return _merge_persisted_events(task, workspace)
        if persisted is not None:
            _launch_plan_turn(
                workspace=workspace,
                task=persisted,
                draft=request.description,
                turn_id=turn_id,
            )
            return get_agent_task(task_id, workspace)
    return task


@router.get("", response_model=AgentTaskListResponse)
def list_agent_tasks(workspace=Depends(get_workspace)) -> AgentTaskListResponse:  # noqa: ANN001
    """List active and historical agent tasks."""
    sessions = agent_routes.list_sessions(workspace=workspace)
    tasks: list[AgentTaskResponse] = []
    seen_task_ids: set[str] = set()
    for session in sessions.sessions:
        persisted = _persisted_for_session(workspace, session.sessionId)
        if persisted is not None and persisted.active_mode == "plan":
            task = _task_from_metadata(persisted).model_copy(update={"events": session.events})
        else:
            task = _task_from_session(session, persisted=persisted)
            _persist_task_response(workspace, task, persisted=persisted)
        task = _merge_persisted_events(task, workspace)
        tasks.append(task)
        seen_task_ids.add(task.taskId)
    root = _workspace_root(workspace)
    if root is not None:
        for persisted in list_agent_task_metadata(root):
            if persisted.task_id in seen_task_ids:
                continue
            tasks.append(_merge_persisted_events(_task_from_metadata(persisted), workspace))
    tasks.sort(key=lambda task: task.updatedAt or task.createdAt, reverse=True)
    return AgentTaskListResponse(tasks=tasks, total=len(tasks))


@router.get("/{task_id}", response_model=AgentTaskResponse)
def get_agent_task(
    task_id: str,
    workspace=Depends(get_workspace),  # noqa: ANN001
) -> AgentTaskResponse:
    """Get a single agent task by task id."""
    root = _workspace_root(workspace)
    persisted_direct = read_agent_task_metadata(root, task_id) if root is not None else None
    if persisted_direct is not None and persisted_direct.active_mode == "plan":
        task = _task_from_metadata(persisted_direct)
        try:
            chat_session = agent_routes.get_session(
                persisted_direct.session_id,
                workspace=workspace,
            )
        except HTTPException:
            pass
        else:
            task = task.model_copy(
                update={"events": chat_session.events, "stats": chat_session.stats}
            )
        return _merge_persisted_events(task, workspace)
    session_id = _session_id_for_task(workspace, task_id)
    try:
        session = agent_routes.get_session(session_id, workspace=workspace)
    except HTTPException:
        root = _workspace_root(workspace)
        if root is not None:
            persisted = read_agent_task_metadata(root, task_id)
            if persisted is not None:
                task = _task_from_metadata(persisted)
                # A persisted task (e.g. a PlanMode run) carries a synthesized
                # transcript so the session view shows the whole flow.
                return _merge_persisted_events(task, workspace)
        raise
    persisted = _persisted_for_session(workspace, session.sessionId)
    task = _task_from_session(session, persisted=persisted)
    _persist_task_response(workspace, task, persisted=persisted)
    return _merge_persisted_events(task, workspace)


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
    root = _workspace_root(workspace)
    persisted = read_agent_task_metadata(root, task_id) if root is not None else None
    if persisted is not None and persisted.active_mode == "plan":
        import asyncio
        import json

        async def _generate():  # noqa: ANN202
            sent = 0
            while True:
                current_events = read_agent_task_events(root, task_id)
                for event in current_events[sent:]:
                    yield f"data: {json.dumps(event)}\n\n"
                sent = len(current_events)
                current = read_agent_task_metadata(root, task_id)
                if current is None or current.status not in {
                    "running",
                    "waiting_approval",
                    "awaiting_user",
                }:
                    yield 'data: {"type":"done"}\n\n'
                    return
                await asyncio.sleep(0.5)

        return StreamingResponse(_generate(), media_type="text/event-stream")
    return await agent_routes.stream_events(
        _session_id_for_task(workspace, task_id), workspace=workspace
    )


@router.post("/{task_id}/messages", response_model=MessageResponse)
async def post_agent_task_message(
    task_id: str,
    request: UserMessageCreateRequest,
    workspace=Depends(get_workspace),  # noqa: ANN001
) -> MessageResponse:
    """Send a follow-up user message on an existing agent task.

    Continues the *same* runtime session (does not create a new task). A turn
    already in flight is rejected with 409 by the session layer.
    """
    root = _workspace_root(workspace)
    persisted = read_agent_task_metadata(root, task_id) if root is not None else None
    if persisted is None:
        raise HTTPException(status_code=404, detail=f"agent task {task_id!r} not found")

    # A pending context question belongs to the plan turn regardless of the
    # mode currently shown in the composer.
    if persisted.status == "awaiting_user" and persisted.active_mode == "plan":
        resolved = _resolve_experiment_context(workspace, request.content)
        if resolved is None:
            append_agent_task_events(
                root,
                task_id,
                [
                    {
                        "type": "clarification_required",
                        "ts": datetime.now(UTC).isoformat(),
                        "payload": {
                            "request_id": request.request_id
                            or f"context-{persisted.active_turn_id}",
                            "questions": (
                                "I could not uniquely resolve that experiment. "
                                "Reply with `project / experiment` or an experiment id."
                            ),
                            "context_kind": "experiment",
                            "turn_id": persisted.active_turn_id,
                            "mode": "plan",
                        },
                    }
                ],
            )
            return MessageResponse(message="context still required")
        project_id, experiment_id = resolved
        persisted = replace(
            persisted,
            project_id=project_id,
            experiment_id=experiment_id,
            status="running",
            updated_at=datetime.now(UTC).isoformat(),
        )
        write_agent_task_metadata(root, persisted)
        _launch_plan_turn(
            workspace=workspace,
            task=persisted,
            draft=persisted.pending_plan_draft or persisted.goal,
            turn_id=persisted.active_turn_id or _turn_id(),
        )
        return MessageResponse(message="context accepted; plan started")

    requested_mode = "plan" if request.mode == "plan" else "chat"
    if requested_mode == "plan":
        if persisted.status in {"running", "waiting_approval"}:
            raise HTTPException(status_code=409, detail="a turn is already in flight for this task")
        turn_id = _turn_id()
        revision_draft = request.content
        if persisted.run_id:
            revision_draft = (
                f"Original request:\n{persisted.goal}\n\n"
                f"Revise the previous plan ({persisted.run_id}) using this feedback:\n"
                f"{request.content}"
            )
        if not persisted.project_id or not persisted.experiment_id:
            write_agent_task_metadata(
                root,
                replace(
                    persisted,
                    status="awaiting_user",
                    plan_mode=True,
                    active_mode="plan",
                    active_turn_id=turn_id,
                    active_plan_task_id=None,
                    pending_plan_draft=revision_draft,
                    updated_at=datetime.now(UTC).isoformat(),
                ),
            )
            append_agent_task_events(
                root,
                task_id,
                [
                    {
                        "type": "loop_started",
                        "ts": datetime.now(UTC).isoformat(),
                        "payload": {
                            "user_input": request.content,
                            "turn_id": turn_id,
                            "mode": "plan",
                        },
                    },
                    {
                        "type": "clarification_required",
                        "ts": datetime.now(UTC).isoformat(),
                        "payload": {
                            "request_id": f"context-{turn_id}",
                            "questions": (
                                "Which project and experiment should this plan belong to? "
                                "Reply with `project / experiment` or an experiment id."
                            ),
                            "context_kind": "experiment",
                            "turn_id": turn_id,
                            "mode": "plan",
                        },
                    },
                ],
            )
            return MessageResponse(message="experiment context required")
        _launch_plan_turn(
            workspace=workspace,
            task=persisted,
            draft=revision_draft,
            turn_id=turn_id,
        )
        return MessageResponse(message="plan turn started")

    # Chat turns reuse the interactive runtime when one exists. A plan-first
    # task lazily creates that runtime here, while retaining the same task id.
    session_id = persisted.session_id
    from molexp.server.dependencies import get_agent_runtime

    runtime = get_agent_runtime().get(root or "", session_id)
    if runtime is None:
        chat_request = GoalCreateRequest(
            description=request.content,
            projectId=persisted.project_id,
            experimentId=persisted.experiment_id,
            runId=persisted.run_id,
            mode="chat",
        )
        session = await agent_routes.create_session(chat_request, workspace=workspace)
        session_id = session.sessionId
    else:
        await agent_routes.post_user_message(session_id, request, workspace=workspace)
    write_agent_task_metadata(
        root,
        replace(
            persisted,
            session_id=session_id,
            status="running",
            plan_mode=False,
            active_mode="chat",
            active_turn_id=_turn_id(),
            active_plan_task_id=None,
            pending_plan_draft=None,
            updated_at=datetime.now(UTC).isoformat(),
        ),
    )
    return MessageResponse(message="accepted")


@router.post("/{task_id}/cancel", response_model=MessageResponse)
async def cancel_agent_task(
    task_id: str,
    workspace=Depends(get_workspace),  # noqa: ANN001
) -> MessageResponse:
    """Stop the in-flight turn for this task (idempotent when already idle)."""
    root = _workspace_root(workspace)
    persisted = read_agent_task_metadata(root, task_id) if root is not None else None
    if persisted is not None and persisted.active_mode == "plan" and persisted.active_plan_task_id:
        from molexp.server.deps.plan_runtime import get_plan_runtime

        plan_task = get_plan_runtime().get(root or "", persisted.active_plan_task_id)
        if plan_task is not None:
            plan_task.cancel()
            await plan_task.await_finished()
        response = MessageResponse(message="cancelled")
    else:
        session_id = _session_id_for_task(workspace, task_id)
        response = await agent_routes.cancel_session(session_id, workspace=workspace)
    if root is not None:
        persisted = read_agent_task_metadata(root, task_id)
        if persisted is not None:
            write_agent_task_metadata(
                root,
                PersistedAgentTask(
                    task_id=persisted.task_id,
                    session_id=persisted.session_id,
                    title=persisted.title,
                    goal=persisted.goal,
                    status="cancelled",
                    created_at=persisted.created_at,
                    plan_mode=persisted.plan_mode,
                    active_mode=persisted.active_mode,
                    active_turn_id=persisted.active_turn_id,
                    active_plan_task_id=persisted.active_plan_task_id,
                    pending_plan_draft=persisted.pending_plan_draft,
                    skill_id=persisted.skill_id,
                    project_id=persisted.project_id,
                    experiment_id=persisted.experiment_id,
                    run_id=persisted.run_id,
                ),
            )
    return response


@router.delete("/{task_id}", response_model=MessageResponse)
async def delete_agent_task_route(
    task_id: str,
    workspace=Depends(get_workspace),  # noqa: ANN001
) -> MessageResponse:
    """Cancel any live turn, drop the runtime, and remove task metadata."""
    from molexp.server.dependencies import get_agent_runtime

    root = _workspace_root(workspace) or ""
    session_id = _session_id_for_task(workspace, task_id)
    had_runtime = get_agent_runtime().get(root, session_id) is not None
    had_meta = bool(root) and read_agent_task_metadata(root, task_id) is not None
    if not had_runtime and not had_meta:
        raise HTTPException(status_code=404, detail=f"agent task {task_id!r} not found")
    persisted = read_agent_task_metadata(root, task_id) if root else None
    if persisted is not None and persisted.active_plan_task_id:
        from molexp.server.deps.plan_runtime import get_plan_runtime

        plan_task = get_plan_runtime().get(root, persisted.active_plan_task_id)
        if plan_task is not None:
            plan_task.cancel()
            await plan_task.await_finished()
    await agent_routes.delete_session(session_id, workspace=workspace)
    if root:
        delete_agent_task(root, task_id)
    return MessageResponse(message="deleted")
