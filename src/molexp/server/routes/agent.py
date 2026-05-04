"""Agent routes for MolExp API — backed by ``molexp.agent.AgentService``.

The route layer is a thin translator between FastAPI schemas and the
harness public surface. The session registry and asyncio task
ownership live entirely on :class:`AgentService`; these handlers do
not own ``_sessions`` globals.

Skills come from :mod:`molexp.agent.state.skills`, provider credentials
and config from :mod:`molexp.plugins.model_pydanticai`, and the system
prompt from :mod:`molexp.agent.context.prompt`.

Handlers take ``workspace`` as their only injected dependency and
derive the workspace-scoped :class:`AgentService` via
:func:`_service_for`. That way external callers (e.g. the
``/agent-tasks`` route layer) can invoke a handler with a real
workspace object instead of going through FastAPI's dependency chain.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from molexp.agent._serialize import to_jsonable as _to_jsonable
from molexp.agent import (
    AgentMode,
    AgentService,
    Goal,
    SessionStatus,
)
from molexp.agent.orchestration import (
    SessionCompleted,
    UserMessageReceived,
)
from molexp.agent.tools import ApprovalDecision

from ..dependencies import get_workspace
from ..schemas import (
    AgentSessionListResponse,
    AgentSessionResponse,
    AgentSystemPromptResponse,
    ApprovalRespondRequest,
    GoalCreateRequest,
    MessageResponse,
    PlanDecisionRequest,
    SessionEventResponse,
    SessionStatsResponse,
    SkillLaunchRequest,
    UserMessageCreateRequest,
)

router = APIRouter(prefix="/agent", tags=["agent"])


_service_cache: dict[str, AgentService] = {}
_service_cache_lock = Lock()


def _service_for(workspace) -> AgentService:
    root = getattr(workspace, "root", None)
    key = str(root) if root is not None else "<no-root>"
    with _service_cache_lock:
        existing = _service_cache.get(key)
        if existing is None:
            existing = AgentService.from_workspace(
                workspace_path=root or Path("."),
                workspace=workspace,
                model=_resolve_model_client(root),
            )
            _service_cache[key] = existing
    return existing


def _resolve_model_client(root: Path | None):
    """Build a :class:`ModelClient` from the workspace's provider config.

    Returns ``None`` when no API key is configured — sessions then run
    in metadata-only mode (no background turn-loop task). When a
    client is built, the plugin is wired with a ``model_io.jsonl``
    sink so each request/response pair lands on disk (Decision M1).
    """

    if root is None:
        return None
    import molexp.plugins.model_pydanticai  # noqa: F401 — registers the factory
    from molexp.agent import AgentService, create_model_client
    from molexp.agent.state.sessions import SessionStore
    from molexp.plugins.model_pydanticai.store import ProviderStore

    config = ProviderStore(root).load()
    if not config.api_key:
        return None
    sessions_root = root / AgentService.AGENT_DIRNAME / "sessions"
    store = SessionStore(sessions_root)
    return create_model_client(config, model_io_sink=store.append_model_io)


def get_agent_service(workspace=Depends(get_workspace)) -> AgentService:
    return _service_for(workspace)


def reset_agent_service_cache() -> None:
    """Drop every cached :class:`AgentService` (used by tests)."""

    with _service_cache_lock:
        _service_cache.clear()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _serialize_event(event: Any) -> SessionEventResponse:
    """Convert a harness event dataclass into the wire response shape.

    The event ``type`` is the dataclass class name; everything else
    flattens into ``payload`` (timestamps rendered ISO 8601). UI
    consumers use ``type`` to dispatch to the matching renderer.
    """

    cls_name = type(event).__name__
    payload: dict[str, Any] = {}
    ts: datetime | None = None
    if dataclasses.is_dataclass(event):
        for f in dataclasses.fields(event):
            value = getattr(event, f.name)
            if f.name == "ts" and isinstance(value, datetime):
                ts = value
                continue
            payload[f.name] = _to_jsonable(value)
    else:
        payload = {"value": str(event)}
    return SessionEventResponse(
        type=cls_name,
        ts=(ts or datetime.now(timezone.utc)).isoformat(),
        payload=payload,
    )




def _goal_from_request(request: GoalCreateRequest, skill_instructions: str) -> Goal:
    """Translate the wire ``GoalCreateRequest`` into a harness ``Goal``.

    ``plan_mode`` (legacy boolean) maps to :class:`AgentMode.PLAN`.
    ``skill_instructions`` flows into the harness via
    ``instructions_override`` when no explicit override is set.
    """

    return Goal(
        description=request.description,
        constraints=dict(request.constraints),
        success_criteria=list(request.success_criteria),
        mode=AgentMode.PLAN if request.plan_mode else AgentMode.CHAT,
        instructions_override=request.instructions_override or (skill_instructions or None),
        skill_id=request.skill_id,
    )


def _session_response(
    *,
    session_id: str,
    goal: Goal,
    status: SessionStatus | str,
    created_at: str | None = None,
    completed_at: str | None = None,
) -> AgentSessionResponse:
    status_str = status.value if isinstance(status, SessionStatus) else status
    stats = SessionStatsResponse(
        startedAt=created_at,
        completedAt=completed_at,
    )
    return AgentSessionResponse(
        sessionId=session_id,
        status=status_str,
        goalDescription=goal.description,
        createdAt=created_at or _now_iso(),
        events=[],
        stats=stats,
        planMode=goal.mode is AgentMode.PLAN,
        skillId=goal.skill_id,
    )


def _resolve_skill_instructions(workspace, skill_id: str | None) -> str:
    if not skill_id:
        return ""
    root = getattr(workspace, "root", None)
    if root is None:
        return ""
    from molexp.agent.state.skills import SkillStore

    skill = SkillStore(root).get(skill_id)
    return skill.instructions if skill is not None else ""


def _require_credentials(workspace) -> None:
    """Pre-flight: refuse to start a session when no API key is reachable."""

    root = getattr(workspace, "root", None)
    if root is None:
        return
    from molexp.plugins.model_pydanticai import ProviderStore, check_credentials

    config = ProviderStore(root).load()
    status = check_credentials(config)
    if status.ready:
        return
    raise HTTPException(
        status_code=400,
        detail={
            "code": "agent_not_configured",
            "message": status.reason,
            "provider": status.provider,
            "model": status.model,
            "envVar": status.env_var,
        },
    )


@router.post("/sessions", response_model=AgentSessionResponse)
async def create_session(
    request: GoalCreateRequest,
    workspace=Depends(get_workspace),
) -> AgentSessionResponse:
    """Start a new agent session via :class:`AgentService`."""

    skill_instructions = _resolve_skill_instructions(workspace, request.skill_id)
    goal = _goal_from_request(request, skill_instructions)

    _require_credentials(workspace)

    service = _service_for(workspace)
    session = service.start_session(goal)
    return _session_response(
        session_id=session.session_id,
        goal=goal,
        status=session.status,
    )


@router.get("/sessions", response_model=AgentSessionListResponse)
def list_sessions(workspace=Depends(get_workspace)) -> AgentSessionListResponse:
    """List every agent session known to the workspace store."""

    service = _service_for(workspace)
    rows = []
    for meta in service.list_sessions():
        completed_at = (
            meta.updated_at.isoformat()
            if meta.status in {SessionStatus.COMPLETED, SessionStatus.FAILED}
            else None
        )
        rows.append(
            _session_response(
                session_id=meta.session_id,
                goal=meta.goal,
                status=meta.status,
                created_at=meta.created_at.isoformat(),
                completed_at=completed_at,
            )
        )
    return AgentSessionListResponse(sessions=rows, total=len(rows))


@router.get("/sessions/{session_id}", response_model=AgentSessionResponse)
def get_session(
    session_id: str,
    workspace=Depends(get_workspace),
) -> AgentSessionResponse:
    service = _service_for(workspace)
    live = service.get_session(session_id)
    if live is not None:
        return _session_response(
            session_id=live.session_id,
            goal=live.goal,
            status=live.status,
        )
    meta = service.state.sessions.read_metadata(session_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return _session_response(
        session_id=meta.session_id,
        goal=meta.goal,
        status=meta.status,
        created_at=meta.created_at.isoformat(),
    )


@router.get("/sessions/{session_id}/events")
async def stream_events(
    session_id: str,
    workspace=Depends(get_workspace),
) -> StreamingResponse:
    """Stream agent session events via Server-Sent Events."""

    service = _service_for(workspace)
    session = service.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    iterator = session.stream_events()

    async def generate() -> AsyncGenerator[str, None]:
        async for event in iterator:
            wire = _serialize_event(event)
            yield f"data: {wire.model_dump_json()}\n\n"
            if isinstance(event, SessionCompleted):
                break
        yield 'data: {"type": "done"}\n\n'

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/sessions/{session_id}/approve")
async def respond_approval(
    session_id: str,
    request: ApprovalRespondRequest,
    workspace=Depends(get_workspace),
) -> dict:
    service = _service_for(workspace)
    session = service.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    applied = await session.respond_approval(
        ApprovalDecision(request_id=request.request_id, approved=request.approved)
    )
    return {
        "request_id": request.request_id,
        "approved": request.approved,
        "applied": applied,
    }


@router.post("/sessions/{session_id}/plan-decision", response_model=MessageResponse)
async def respond_plan(
    session_id: str,
    request: PlanDecisionRequest,
    workspace=Depends(get_workspace),
) -> MessageResponse:
    """Resolve a pending plan handoff.

    On approval the session flips out of PLAN mode and the runner
    proceeds; on rejection the runner injects a synthetic user message
    with the feedback (see ``render_reject_feedback``).
    """

    service = _service_for(workspace)
    session = service.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    ok = await session.respond_plan(
        request_id=request.request_id,
        approved=request.approved,
        edited_plan=request.edited_plan,
        edited_workflow_ir=request.edited_workflow_ir,
        feedback=request.feedback,
    )
    if not ok:
        raise HTTPException(
            status_code=409,
            detail="No matching pending plan; the request id may be stale.",
        )
    return MessageResponse(message="approved" if request.approved else "rejected")


@router.post("/sessions/{session_id}/messages", response_model=MessageResponse)
async def post_user_message(
    session_id: str,
    request: UserMessageCreateRequest,
    workspace=Depends(get_workspace),
) -> MessageResponse:
    """Deliver a chat message to a running session.

    Either resolves a pending :class:`UserMessageRequested` (when
    ``request_id`` matches) or queues an unsolicited follow-up onto the
    session inbox.
    """

    service = _service_for(workspace)
    session = service.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    await session.send_user_message(request.content, request.request_id)
    # Echo to the bus so the UI can render the inbound message inline.
    await session.bus.publish(
        UserMessageReceived(
            content=request.content,
            request_id=request.request_id,
        )
    )
    return MessageResponse(message="queued")


@router.post("/skills/{skill_id}/launch", response_model=AgentSessionResponse)
async def launch_skill(
    skill_id: str,
    request: SkillLaunchRequest,
    workspace=Depends(get_workspace),
) -> AgentSessionResponse:
    """Materialize a saved skill into a Goal and start a new session."""

    root = getattr(workspace, "root", None)
    if root is None:
        raise HTTPException(status_code=500, detail="Workspace has no root path")
    from molexp.agent.state.skills import SkillStore

    skill = SkillStore(root).get(skill_id)
    if skill is None:
        raise HTTPException(status_code=404, detail=f"Skill '{skill_id}' not found")
    try:
        rendered = skill.materialize(request.parameters)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    plan_mode = (
        request.plan_mode if request.plan_mode is not None else skill.default_plan_mode
    )
    goal = Goal(
        description=rendered["description"],
        constraints={"items": rendered["constraints"]} if rendered["constraints"] else {},
        success_criteria=list(rendered["success_criteria"]),
        mode=AgentMode.PLAN if plan_mode else AgentMode.CHAT,
        instructions_override=skill.instructions or None,
        skill_id=skill.id,
    )
    service = _service_for(workspace)
    session = service.start_session(goal)
    return _session_response(
        session_id=session.session_id,
        goal=goal,
        status=session.status,
    )


@router.get(
    "/sessions/{session_id}/system-prompt",
    response_model=AgentSystemPromptResponse,
)
def get_session_system_prompt(
    session_id: str,
    workspace=Depends(get_workspace),
) -> AgentSystemPromptResponse:
    """Return the layered system prompt for a session."""

    from molexp.agent.context.prompt import BASE_SYSTEM_PROMPT, compose_system_prompt

    workspace_instructions = ""
    root = getattr(workspace, "root", None)
    if root is not None:
        from molexp.plugins.model_pydanticai.store import ProviderStore

        workspace_instructions = ProviderStore(root).load().instructions

    service = _service_for(workspace)
    live = service.get_session(session_id)
    if live is not None:
        goal = live.goal
    else:
        meta = service.state.sessions.read_metadata(session_id)
        if meta is None:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        goal = meta.goal

    plan_mode = goal.mode is AgentMode.PLAN
    effective = compose_system_prompt(
        base=BASE_SYSTEM_PROMPT,
        workspace_instructions=workspace_instructions,
        skill_instructions="",
        session_override=goal.instructions_override,
        plan_mode=plan_mode,
    )
    return AgentSystemPromptResponse(
        base=BASE_SYSTEM_PROMPT,
        workspaceInstructions=workspace_instructions,
        skillInstructions="",
        sessionOverride=goal.instructions_override,
        planMode=plan_mode,
        effective=effective,
    )
