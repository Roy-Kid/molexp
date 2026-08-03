"""User-facing Agent Task routes.

This module is a compatibility layer over the current AgentSession runtime.
It gives the product/UI a stable ``AgentTask`` surface while the lower-level
session store, event persistence, and review model are migrated separately.
"""

from __future__ import annotations

import uuid
from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from molexp.services.agent_task_store import (
    PersistedAgentTask,
    append_agent_task_events,
    delete_agent_task,
    list_agent_task_metadata,
    merge_agent_task_events,
    read_agent_task_events,
    read_agent_task_metadata,
    write_agent_task_metadata,
)

from ..dependencies import get_workspace
from ..schemas import (
    AgentEvent,
    AgentSessionResponse,
    AgentSystemPromptResponse,
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
        projectId=persisted.project_id if persisted is not None else None,
        experimentId=persisted.experiment_id if persisted is not None else None,
        runId=persisted.run_id if persisted is not None else None,
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
        projectId=task.project_id,
        experimentId=task.experiment_id,
        runId=task.run_id,
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
    """Write the task's on-disk metadata + merge any live events.

    The mount scope either arrives explicitly (task creation) or rides
    *persisted* (read paths refreshing live-session state) — a refresh must
    never wipe the stored scope. Events are merged (not replaced) so multi-turn
    chat history accumulates under ``agent/_tasks/<id>/events.json`` and
    survives ``molexp serve`` restarts.
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
    # Stamp the live runtime so turn-complete flush uses the product task id.
    from molexp.server.dependencies import get_agent_runtime

    runtime = get_agent_runtime().get(root, task.sessionId)
    if runtime is not None:
        runtime.task_id = task.taskId
        runtime.workspace_root = root
    if task.events:
        merge_agent_task_events(
            root,
            task.taskId,
            [
                {
                    "type": event.type,
                    "ts": event.ts,
                    "payload": event.payload if isinstance(event.payload, dict) else {},
                }
                for event in task.events
            ],
        )


def _record_task_error(
    root: str,
    task_id: str,
    *,
    message: str,
    stage: str,
    persisted: PersistedAgentTask | None = None,
) -> None:
    """Append a visible error + failed completion to the task transcript."""
    now = datetime.now(UTC).isoformat()
    append_agent_task_events(
        root,
        task_id,
        [
            {
                "type": "error",
                "ts": now,
                "payload": {
                    "message": message,
                    "stage": stage,
                    "detail": message,
                },
            },
            {
                "type": "loop_completed",
                "ts": now,
                "payload": {
                    "text": message,
                    "failed": True,
                    "stage": stage,
                },
            },
        ],
    )
    meta = persisted or read_agent_task_metadata(root, task_id)
    if meta is not None:
        write_agent_task_metadata(
            root,
            replace(meta, status="failed", updated_at=now),
        )


# Disk statuses that claim an exclusive turn is underway.
_IN_FLIGHT_STATUSES = frozenset({"running", "waiting_approval"})


def _is_plan_task(persisted: PersistedAgentTask) -> bool:
    """Plan Mode is ``active_mode == "plan"`` — single source of truth on disk."""
    return persisted.active_mode == "plan"


def _plan_turn_live(root: str | None, plan_task_id: str | None) -> bool:
    """True when the plan-task registry still holds ``plan_task_id``."""
    if not root or not plan_task_id:
        return False
    try:
        from molexp.server.deps.plan_runtime import get_plan_runtime

        return get_plan_runtime().get(root, plan_task_id) is not None
    except Exception:
        return False


def _chat_runtime(root: str | None, session_id: str):  # noqa: ANN202
    if not root:
        return None
    from molexp.server.dependencies import get_agent_runtime

    return get_agent_runtime().get(root, session_id)


def _turn_is_live(workspace: Workspace, persisted: PersistedAgentTask) -> bool:
    """Whether a real in-process turn owns this task (not merely disk status)."""
    root = _workspace_root(workspace)
    if _is_plan_task(persisted):
        return _plan_turn_live(root, persisted.active_plan_task_id)
    live = _chat_runtime(root, persisted.session_id)
    return live is not None and live.status() == "running"


def _has_server_restart_error(root: str, task_id: str) -> bool:
    events = read_agent_task_events(root, task_id)
    return any(
        isinstance(e, dict)
        and e.get("type") == "error"
        and (e.get("payload") or {}).get("stage") == "server_restart"
        for e in events
    )


def _demote_stale_in_flight(
    workspace: Workspace,
    persisted: PersistedAgentTask,
    *,
    message: str,
) -> PersistedAgentTask:
    """Write failed + a transcript error; return refreshed metadata."""
    root = _workspace_root(workspace)
    if root is None:
        return replace(persisted, status="failed")
    if not _has_server_restart_error(root, persisted.task_id):
        _record_task_error(
            root,
            persisted.task_id,
            message=message,
            stage="server_restart",
            persisted=persisted,
        )
    else:
        write_agent_task_metadata(
            root,
            replace(
                persisted,
                status="failed",
                updated_at=datetime.now(UTC).isoformat(),
            ),
        )
    return read_agent_task_metadata(root, persisted.task_id) or replace(persisted, status="failed")


def _reap_stale_in_flight(
    workspace: Workspace,
    persisted: PersistedAgentTask,
    *,
    aggressive: bool = False,
) -> PersistedAgentTask:
    """Demote disk in-flight status when no live turn exists.

    Plan Mode runs in the PlanTask registry, **not** the chat session registry.
    Demoting every disk ``running`` plan that lacks a chat runtime would mark
    plans failed the moment the operator supplies project/experiment (status
    flips to running before ``active_plan_task_id`` is written).

    * ``aggressive=False`` (GET hydrate): keep plan ``running`` without
      ``active_plan_task_id`` (brief launch race). Still demote when a stored
      plan-task id is gone, or when ``waiting_approval`` has no live plan.
    * ``aggressive=True`` (POST message / cancel prep): also demote plan
      ``running`` with no plan-task id so a zombie cannot 409 forever after
      frontend refresh or a dead process.
    """
    if persisted.status not in _IN_FLIGHT_STATUSES:
        return persisted

    root = _workspace_root(workspace)
    if _is_plan_task(persisted):
        if _plan_turn_live(root, persisted.active_plan_task_id):
            return persisted
        # No live plan. Demote durable approval waits always; demote running
        # when the plan id is known-dead or the caller needs a free turn.
        should_demote = (
            persisted.status == "waiting_approval"
            or bool(persisted.active_plan_task_id)
            or aggressive
        )
        if not should_demote:
            return persisted
        return _demote_stale_in_flight(
            workspace,
            persisted,
            message=(
                "This plan turn is no longer running (server restart or cancelled "
                "worker). Open the experiment run for partial artifacts, or send a "
                "new message to re-plan."
            ),
        )

    # Chat: demote only when disk says running and the session runtime is gone.
    live = _chat_runtime(root, persisted.session_id)
    if live is not None:
        return persisted
    if persisted.status != "running":
        return persisted
    return _demote_stale_in_flight(
        workspace,
        persisted,
        message=(
            "Server restarted while this chat turn was running. "
            "Reply to continue, or start a new task."
        ),
    )


def _hydrate_disk_task(
    workspace: Workspace,
    persisted: PersistedAgentTask,
) -> AgentTaskResponse:
    """Load a task from disk; demote stale in-flight when no live runtime."""
    reaped = _reap_stale_in_flight(workspace, persisted, aggressive=False)
    task = _task_from_metadata(reaped)
    return _merge_persisted_events(task, workspace)


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
        now = datetime.now(UTC).isoformat()
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
                updated_at=now,
            ),
        )
        # Immediate transcript breadcrumb so the UI is not stuck on a lone
        # user loop_started while molmcp ground / first LLM call is still pending.
        append_agent_task_events(
            root,
            task.task_id,
            [
                {
                    "type": "stage_started",
                    "ts": now,
                    "payload": {
                        "stage": "plan",
                        "message": (
                            f"Plan started under `{task.project_id}/{task.experiment_id}` "
                            f"(run {started.runId}). Grounding capabilities, then drafting "
                            "the task board…"
                        ),
                        "project_id": task.project_id,
                        "experiment_id": task.experiment_id,
                        "run_id": started.runId,
                        "plan_task_id": started.taskId,
                        "turn_id": turn_id,
                    },
                }
            ],
        )


def _identity_keys(name: str, entity_id: str) -> set[str]:
    """Casefold + slug identity keys for matching free-text context answers."""
    from molexp.ids import slugify

    keys = {name.strip().casefold(), entity_id.strip().casefold()}
    for raw in (name, entity_id):
        slug = slugify(raw)
        if slug:
            keys.add(slug.casefold())
    return {k for k in keys if k}


def _parse_context_answer(answer: str) -> tuple[str, str]:
    """Split ``project / experiment`` (or ``:``) into raw name parts.

    Returns ``(project_part, experiment_part)``. When no separator is present,
    ``project_part`` is empty and the whole answer is the experiment hint.
    """
    text = answer.strip()
    if not text:
        return "", ""
    for separator in ("/", ":"):
        if separator in text:
            left, right = (part.strip() for part in text.split(separator, 1))
            return left, right
    return "", text


def _experiment_catalog(workspace: Workspace) -> list[dict[str, str]]:
    """Structured project/experiment options for the in-bubble scope form."""
    rows: list[dict[str, str]] = []
    for project in workspace.list_projects():
        experiments = project.list_experiments()
        if not experiments:
            rows.append(
                {
                    "project_id": project.id,
                    "experiment_id": "",
                    "label": f"{project.id} (no experiments yet)",
                }
            )
            continue
        for experiment in experiments:
            rows.append(
                {
                    "project_id": project.id,
                    "experiment_id": experiment.id,
                    "label": f"{project.id} / {experiment.id}",
                }
            )
    return rows[:40]


def _list_context_catalog(workspace: Workspace) -> str:
    """Short human listing of project / experiment pairs (legacy text fallback)."""
    rows = _experiment_catalog(workspace)
    if not rows:
        return "No projects yet — create new names in the form."
    lines = [f"- `{r['label']}`" for r in rows[:12]]
    extra = len(rows) - min(len(rows), 12)
    body = "\n".join(lines)
    if extra > 0:
        body += f"\n- … and {extra} more"
    return f"Available:\n{body}"


def _experiment_clarification_payload(
    workspace: Workspace,
    *,
    request_id: str,
    turn_id: str | None,
    questions: str | None = None,
) -> dict[str, object]:
    """Payload for UI bubble form (catalog + short prompt)."""
    catalog = _experiment_catalog(workspace)
    intro = questions or (
        "Choose where this plan should live. Pick an existing scope or create new names."
    )
    return {
        "request_id": request_id,
        "questions": intro,
        "context_kind": "experiment",
        "allow_create": True,
        "catalog": catalog,
        "turn_id": turn_id,
        "mode": "plan",
    }


def _resolve_experiment_context(workspace: Workspace, answer: str) -> tuple[str, str] | None:
    """Resolve `project / experiment`, ids, or a unique experiment name.

    Matching is case-insensitive and slug-aware (``PE_Foo`` ≡ ``pe-foo``).
    Returns ``None`` when zero or multiple matches (caller should ensure/create
    or emit a clearer clarification).
    """
    project_raw, experiment_raw = _parse_context_answer(answer)
    if not experiment_raw and not project_raw:
        return None
    project_keys = _identity_keys(project_raw, project_raw) if project_raw else set()
    experiment_keys = _identity_keys(experiment_raw, experiment_raw)

    matches: list[tuple[str, str]] = []
    for project in workspace.list_projects():
        p_keys = _identity_keys(project.name, project.id)
        if project_keys and p_keys.isdisjoint(project_keys):
            continue
        for experiment in project.list_experiments():
            e_keys = _identity_keys(experiment.name, experiment.id)
            if e_keys.isdisjoint(experiment_keys):
                continue
            matches.append((project.id, experiment.id))
    if len(matches) == 1:
        return matches[0]
    return None


def _ensure_experiment_context(
    workspace: Workspace, answer: str
) -> tuple[tuple[str, str] | None, str | None]:
    """Resolve an existing scope, or create ``project / experiment`` when named.

    Plan Mode needs a durable home. When the user replies with two path
    segments that do not yet exist (common for goals like “创建一个项目…”),
    create them idempotently via :meth:`Workspace.add_project` /
    :meth:`Project.add_experiment`.

    Returns:
        ``((project_id, experiment_id), None)`` on success, or
        ``(None, clarification_text)`` when the answer is still unusable.
    """
    project_raw, experiment_raw = _parse_context_answer(answer)
    if not experiment_raw and not project_raw:
        return None, (
            "Which project and experiment should this plan belong to?\n"
            "Reply with `project / experiment` (I can create them if missing) "
            "or an experiment id.\n\n" + _list_context_catalog(workspace)
        )

    resolved = _resolve_experiment_context(workspace, answer)
    if resolved is not None:
        return resolved, None

    # Ambiguous experiment-only name (no project segment).
    if not project_raw:
        matches: list[tuple[str, str]] = []
        experiment_keys = _identity_keys(experiment_raw, experiment_raw)
        for project in workspace.list_projects():
            for experiment in project.list_experiments():
                if _identity_keys(experiment.name, experiment.id) & experiment_keys:
                    matches.append((project.id, experiment.id))
        if len(matches) > 1:
            opts = ", ".join(f"`{p} / {e}`" for p, e in matches[:8])
            return None, (
                f"`{experiment_raw}` matches more than one experiment ({opts}). "
                "Reply with the full `project / experiment` path."
            )
        return None, (
            f"I could not find experiment `{experiment_raw}`.\n"
            "Reply with `project / experiment` — new names are created if missing.\n\n"
            + _list_context_catalog(workspace)
        )

    # Two segments: create-or-get project + experiment (idempotent).
    try:
        project = workspace.add_project(project_raw)
        experiment = project.add_experiment(experiment_raw)
    except Exception as exc:
        return None, (
            f"Could not open or create `{project_raw} / {experiment_raw}`: "
            f"{type(exc).__name__}: {exc}"
        )
    return (project.id, experiment.id), None


@router.post("", response_model=AgentTaskResponse)
async def create_agent_task(
    request: GoalCreateRequest,
    workspace=Depends(get_workspace),  # noqa: ANN001
) -> AgentTaskResponse:
    """Create a user-facing agent task.

    The task is the stable conversation container. Each turn is dispatched to
    either the interactive agent or the nine-stage Planning Agent.
    """
    requested_mode: Literal["chat", "plan"] = request.mode
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
            now = datetime.now(UTC).isoformat()
            # loop_started first so the goal turn opens cleanly; clarification
            # closes it. (Clarification-only logs left the next scope reply's
            # loop_started absorbed into a finished bubble — no live spinner.)
            append_agent_task_events(
                root,
                task_id,
                [
                    {
                        "type": "loop_started",
                        "ts": now,
                        "payload": {
                            "user_input": request.description,
                            "turn_id": turn_id,
                            "mode": "plan",
                        },
                    },
                    {
                        "type": "clarification_required",
                        "ts": now,
                        "payload": _experiment_clarification_payload(
                            workspace,
                            request_id=f"context-{turn_id}",
                            turn_id=turn_id,
                        ),
                    },
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
            tasks.append(_hydrate_disk_task(workspace, persisted))
    tasks.sort(key=lambda task: task.updatedAt or task.createdAt, reverse=True)
    return AgentTaskListResponse(tasks=tasks, total=len(tasks))


def _resolve_task_meta(workspace, task_or_session_id: str) -> PersistedAgentTask | None:  # noqa: ANN001
    """Resolve persisted task metadata by task id or runtime session id."""
    root = _workspace_root(workspace)
    if root is None:
        return None
    direct = read_agent_task_metadata(root, task_or_session_id)
    if direct is not None:
        return direct
    return _persisted_for_session(workspace, task_or_session_id)


def _compose_system_prompt_response(
    workspace,  # noqa: ANN001
    *,
    task_meta: PersistedAgentTask | None,
    plan_mode: bool,
) -> AgentSystemPromptResponse:
    """Build the inspector breakdown of the effective InteractiveLoop system prompt.

    Mirrors production composition order (ops preamble → optional mount
    context → plan addendum). Does not require a live LLM runtime.
    """
    from molexp.agent.ops.preamble import CHAT_OPS_PREAMBLE, DEFAULT_OPS_PREAMBLE
    from molexp.services.agent_context import mount_session_scope

    # Chat Mode default preamble (scratch-only); plan adds its own addendum.
    base = (DEFAULT_OPS_PREAMBLE or CHAT_OPS_PREAMBLE).strip()
    workspace_instructions = ""
    if task_meta is not None and (
        task_meta.project_id or task_meta.experiment_id or task_meta.run_id
    ):
        try:
            block, _ = mount_session_scope(
                workspace,
                project_id=task_meta.project_id or None,
                experiment_id=task_meta.experiment_id or None,
                run_id=task_meta.run_id or None,
            )
            workspace_instructions = (block or "").strip()
        except (ValueError, LookupError):
            # Scope ids on the task may be stale after a reorg — show base only.
            workspace_instructions = ""

    skill_instructions = ""
    session_override: str | None = None
    plan_addendum = ""
    if plan_mode:
        plan_addendum = (
            "You are in PLAN MODE (peer of Chat Mode).\n"
            "Use the plan tool surface (task board) — not chat scratch land. "
            "Multi-step reviewable workflow graphs are the goal; "
            "side-effecting workspace writes follow the plan pipeline, not ad-hoc run_land."
        )

    parts = [base]
    if workspace_instructions:
        parts.append(workspace_instructions)
    if skill_instructions:
        parts.append(skill_instructions)
    if session_override:
        parts.append(session_override)
    if plan_addendum:
        parts.append(plan_addendum)
    effective = "\n\n".join(parts)

    return AgentSystemPromptResponse(
        base=base,
        workspaceInstructions=workspace_instructions,
        skillInstructions=skill_instructions,
        sessionOverride=session_override,
        planMode=plan_mode,
        effective=effective,
    )


@router.get("/{task_id}/system-prompt", response_model=AgentSystemPromptResponse)
def get_agent_task_system_prompt(
    task_id: str,
    workspace=Depends(get_workspace),  # noqa: ANN001
) -> AgentSystemPromptResponse:
    """Return the composed system prompt for an agent task (inspector).

    Accepts either a task id or a runtime session id. Live surface replacement
    for the retired ``GET /api/agent/sessions/{id}/system-prompt`` (which
    503s via the legacy agent catch-all).
    """
    task_meta = _resolve_task_meta(workspace, task_id)
    if task_meta is None:
        # Live chat session without a persisted task row — still answer if runtime is live.
        root = _workspace_root(workspace)
        runtime = None
        if root is not None:
            from molexp.server.dependencies import get_agent_runtime

            runtime = get_agent_runtime().get(root, task_id)
            if runtime is None:
                session_id = _session_id_for_task(workspace, task_id)
                if session_id != task_id:
                    runtime = get_agent_runtime().get(root, session_id)
        if runtime is None:
            raise HTTPException(status_code=404, detail=f"agent task {task_id!r} not found")
        plan_mode = False
    else:
        plan_mode = task_meta.active_mode == "plan"
    return _compose_system_prompt_response(workspace, task_meta=task_meta, plan_mode=plan_mode)


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
            # No live runtime after restart — disk events only.
            return _hydrate_disk_task(workspace, persisted_direct)
        else:
            task = task.model_copy(
                update={"events": chat_session.events, "stats": chat_session.stats}
            )
            _persist_task_response(workspace, task, persisted=persisted_direct)
        return _merge_persisted_events(task, workspace)
    session_id = _session_id_for_task(workspace, task_id)
    try:
        session = agent_routes.get_session(session_id, workspace=workspace)
    except HTTPException:
        root = _workspace_root(workspace)
        if root is not None:
            persisted = read_agent_task_metadata(root, task_id)
            if persisted is not None:
                # Disk-only recovery after serve restart (chat + plan).
                return _hydrate_disk_task(workspace, persisted)
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
        import json

        async def _generate():  # noqa: ANN202
            from molexp.server.shutdown import is_shutting_down, wait_or_shutdown

            sent = 0
            while not is_shutting_down():
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
                if await wait_or_shutdown(0.5):
                    yield 'data: {"type":"done"}\n\n'
                    return

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
    that is genuinely live is rejected with 409; disk-only zombie
    ``running`` / ``waiting_approval`` rows are reaped first so a frontend
    refresh or server restart cannot trap the task forever.
    """
    root = _workspace_root(workspace)
    persisted = read_agent_task_metadata(root, task_id) if root is not None else None
    if persisted is None:
        raise HTTPException(status_code=404, detail=f"agent task {task_id!r} not found")

    # Reap dead in-flight status before any mode branch (including scope bind).
    persisted = _reap_stale_in_flight(workspace, persisted, aggressive=True)

    # Context binding for Plan Mode — awaiting_user, or a failed attempt that
    # never got project/experiment (silent launch failure / bad demote).
    needs_scope = persisted.active_mode == "plan" and (
        persisted.status == "awaiting_user"
        or (
            persisted.status in {"failed", "cancelled"}
            and (not persisted.project_id or not persisted.experiment_id)
        )
    )
    if needs_scope:
        resolved, clarify = _ensure_experiment_context(workspace, request.content)
        if resolved is None:
            now = datetime.now(UTC).isoformat()
            append_agent_task_events(
                root,
                task_id,
                [
                    {
                        "type": "loop_started",
                        "ts": now,
                        "payload": {
                            "user_input": request.content,
                            "turn_id": persisted.active_turn_id,
                            "mode": "plan",
                        },
                    },
                    {
                        "type": "clarification_required",
                        "ts": now,
                        "payload": _experiment_clarification_payload(
                            workspace,
                            request_id=request.request_id or f"context-{persisted.active_turn_id}",
                            turn_id=persisted.active_turn_id,
                            questions=clarify
                            or "Could not use that scope — pick or create project / experiment.",
                        ),
                    },
                ],
            )
            # Stay awaiting — never flip to failed just because resolve missed.
            write_agent_task_metadata(
                root,
                replace(
                    persisted,
                    status="awaiting_user",
                    plan_mode=True,
                    active_mode="plan",
                    updated_at=now,
                ),
            )
            return MessageResponse(message="context still required")
        project_id, experiment_id = resolved
        turn_id = persisted.active_turn_id or _turn_id()
        now = datetime.now(UTC).isoformat()
        # Record the user's scope reply as its own turn boundary so the
        # transcript shows You → then plan activity (not a silent accept).
        append_agent_task_events(
            root,
            task_id,
            [
                {
                    "type": "loop_started",
                    "ts": now,
                    "payload": {
                        "user_input": request.content,
                        "turn_id": turn_id,
                        "mode": "plan",
                    },
                }
            ],
        )
        persisted = replace(
            persisted,
            project_id=project_id,
            experiment_id=experiment_id,
            status="running",
            plan_mode=True,
            active_mode="plan",
            updated_at=now,
        )
        write_agent_task_metadata(root, persisted)
        try:
            _launch_plan_turn(
                workspace=workspace,
                task=persisted,
                draft=persisted.pending_plan_draft or persisted.goal,
                turn_id=turn_id,
            )
        except HTTPException as exc:
            detail = exc.detail
            message = detail if isinstance(detail, str) else str(detail)
            _record_task_error(
                root,
                task_id,
                message=message or "Failed to start plan.",
                stage="start_plan",
                persisted=persisted,
            )
            raise
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            _record_task_error(
                root,
                task_id,
                message=message,
                stage="start_plan",
                persisted=persisted,
            )
            raise HTTPException(status_code=500, detail=message) from exc
        return MessageResponse(message="context accepted; plan started")

    requested_mode = "plan" if request.mode == "plan" else "chat"
    if requested_mode == "plan":
        # 409 only when a real process still owns the turn (disk status alone
        # is not enough — see _reap_stale_in_flight above).
        if persisted.status in _IN_FLIGHT_STATUSES and _turn_is_live(workspace, persisted):
            raise HTTPException(
                status_code=409,
                detail="a turn is already in flight for this task",
            )
        if persisted.status in _IN_FLIGHT_STATUSES and not _turn_is_live(workspace, persisted):
            # Defensive second reap (status race between check and launch).
            persisted = _reap_stale_in_flight(workspace, persisted, aggressive=True)
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
                        "payload": _experiment_clarification_payload(
                            workspace,
                            request_id=f"context-{turn_id}",
                            turn_id=turn_id,
                        ),
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
    """Stop the in-flight turn for this task (idempotent when already idle).

    Always succeeds when task metadata exists on disk — including zombie
    ``running`` / ``waiting_approval`` rows after a server restart (no live
    plan or chat runtime). Previously the chat cancel path 404'd when the
    session registry was empty, leaving the UI without a Stop recovery.
    """
    root = _workspace_root(workspace)
    persisted = read_agent_task_metadata(root, task_id) if root is not None else None
    if persisted is None and root is not None:
        # Fall through: maybe only a chat runtime exists under this id.
        pass

    if persisted is not None and _is_plan_task(persisted) and persisted.active_plan_task_id:
        from molexp.server.deps.plan_runtime import get_plan_runtime

        plan_task = get_plan_runtime().get(root or "", persisted.active_plan_task_id)
        if plan_task is not None:
            plan_task.cancel()
            await plan_task.await_finished()
    else:
        session_id = (
            persisted.session_id
            if persisted is not None
            else _session_id_for_task(workspace, task_id)
        )
        live = _chat_runtime(root, session_id)
        if live is not None:
            await agent_routes.cancel_session(session_id, workspace=workspace)
        elif persisted is None:
            raise HTTPException(status_code=404, detail=f"agent task {task_id!r} not found")

    if root is not None:
        persisted = read_agent_task_metadata(root, task_id)
        if persisted is not None:
            write_agent_task_metadata(
                root,
                replace(
                    persisted,
                    status="cancelled",
                    active_plan_task_id=None,
                    updated_at=datetime.now(UTC).isoformat(),
                ),
            )
    return MessageResponse(message="cancelled")


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
