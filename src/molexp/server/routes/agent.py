"""Agent routes for MolExp API."""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from molexp.plugins import Capability, registry
from molexp.plugins.agent_pydanticai._pydantic_ai.system_prompt import (
    BASE_SYSTEM_PROMPT,
    compose_system_prompt,
)
from molexp.plugins.agent_pydanticai.provider import ProviderStore, check_credentials
from molexp.plugins.agent_pydanticai.sessions_store import (
    PersistedSessionSummary,
    list_persisted_sessions,
)
from molexp.plugins.agent_pydanticai.skills import SkillStore
from molexp.plugins.agent_pydanticai.types import Goal, SessionStats

from ..dependencies import get_workspace
from ..schemas import (
    AgentSessionListResponse,
    AgentSessionResponse,
    AgentSystemPromptResponse,
    ApprovalRespondRequest,
    GoalCreateRequest,
    MessageResponse,
    SessionEventResponse,
    SessionStatsResponse,
    SkillLaunchRequest,
    UserMessageCreateRequest,
)

router = APIRouter(prefix="/agent", tags=["agent"])

# In-memory session store — NOT shared across workers.
# Deploy with a single uvicorn worker (--workers 1) or replace with
# an external store (Redis, database) before scaling horizontally.
_sessions: dict[str, AgentSessionResponse] = {}
_live_sessions: dict[str, Any] = {}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stats_to_response(stats: SessionStats) -> SessionStatsResponse:
    return SessionStatsResponse(
        inputTokens=stats.input_tokens,
        outputTokens=stats.output_tokens,
        cacheReadTokens=stats.cache_read_tokens,
        cacheWriteTokens=stats.cache_write_tokens,
        totalTokens=stats.total_tokens,
        requests=stats.requests,
        toolCalls=stats.tool_calls,
        events=stats.events,
        startedAt=stats.started_at.isoformat() if stats.started_at else None,
        completedAt=stats.completed_at.isoformat() if stats.completed_at else None,
        durationSeconds=stats.duration_seconds(),
    )


def _serialize_event(event: Any) -> SessionEventResponse:
    """Convert a molexp SessionEvent dataclass to a wire response.

    The event ``type`` is the dataclass class name; everything else is
    flattened into ``payload`` (timestamps are rendered ISO 8601).
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
            if isinstance(value, datetime):
                payload[f.name] = value.isoformat()
            else:
                payload[f.name] = value
    else:
        payload = {"value": str(event)}
    return SessionEventResponse(
        type=cls_name,
        ts=(ts or datetime.now(timezone.utc)).isoformat(),
        payload=payload,
    )


def _drain_live_events(response: AgentSessionResponse, live: Any) -> None:
    drainer = getattr(live, "drain_pending_events", None)
    if drainer is None:
        return
    try:
        new_events = drainer()
    except Exception:
        return
    for ev in new_events:
        response.events.append(_serialize_event(ev))


def _refresh_response(response: AgentSessionResponse) -> AgentSessionResponse:
    """Update mutable fields (status, stats, buffered events) from the live session."""
    live = _live_sessions.get(response.sessionId)
    if live is None:
        return response
    response.status = getattr(live, "status", response.status)
    stats = getattr(live, "stats", None)
    if stats is not None:
        response.stats = _stats_to_response(stats)
    _drain_live_events(response, live)
    return response


def _require_credentials(workspace) -> None:
    """Pre-flight: refuse to start a session when no API key is reachable.

    Raises 400 with ``code: "agent_not_configured"`` so the UI can route
    the user straight to the Provider settings tab instead of letting the
    agent kick off and then crash inside the LLM client a beat later.
    """
    root = getattr(workspace, "root", None)
    if root is None:
        return
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


def _resolve_skill_instructions(workspace, skill_id: str | None) -> str:
    """Look up a skill's instructions by id; return ``""`` when missing.

    Failures (workspace without root, missing skill, malformed store) all
    degrade to an empty string — a session must not refuse to start
    because the skill row vanished between two clicks.
    """
    if not skill_id:
        return ""
    root = getattr(workspace, "root", None)
    if root is None:
        return ""
    try:
        skill = SkillStore(root).get(skill_id)
    except Exception:
        return ""
    return skill.instructions if skill is not None else ""


def _session_to_response(session, goal: Goal, *, created_at: str) -> AgentSessionResponse:
    return AgentSessionResponse(
        sessionId=session.session_id,
        status=session.status,
        goalDescription=goal.description,
        createdAt=created_at,
        events=[],
        stats=_stats_to_response(session.stats),
        planMode=goal.plan_mode,
        skillId=goal.skill_id,
    )


@router.post("/sessions", response_model=AgentSessionResponse)
async def create_session(
    request: GoalCreateRequest,
    workspace=Depends(get_workspace),
) -> AgentSessionResponse:
    """Start a new agent session."""
    skill_instructions = _resolve_skill_instructions(workspace, request.skill_id)
    goal = Goal(
        description=request.description,
        constraints=request.constraints,
        success_criteria=request.success_criteria,
        plan_mode=request.plan_mode,
        instructions_override=request.instructions_override,
        skill_id=request.skill_id,
        skill_instructions=skill_instructions,
    )

    if not registry.is_available(Capability.AGENT):
        raise HTTPException(
            501, "Agent capability not available. Install: pip install molexp[agent]"
        )

    _require_credentials(workspace)

    try:
        AgentService = registry.get(Capability.AGENT)
        service = AgentService.from_workspace(str(workspace.root))
        session = await service.start(goal)
        response = _session_to_response(session, goal, created_at=_now_iso())
    except NotImplementedError:
        raise HTTPException(501, "Agent runtime not yet implemented")

    _sessions[response.sessionId] = response
    _live_sessions[response.sessionId] = session
    return response


def _persisted_to_response(summary: PersistedSessionSummary) -> AgentSessionResponse:
    """Render an on-disk session summary as a wire response.

    Stats are zero-filled because per-attempt usage isn't persisted in
    metadata.json — only what the disk knows: id, status, goal,
    timestamps. The UI shows these as historical rows.
    """
    return AgentSessionResponse(
        sessionId=summary.session_id,
        status=summary.status,
        goalDescription=summary.goal_description,
        createdAt=summary.created_at or _now_iso(),
        events=[],
        stats=SessionStatsResponse(
            startedAt=summary.created_at,
            completedAt=summary.completed_at,
        ),
        planMode=summary.plan_mode,
        skillId=summary.skill_id,
    )


@router.get("/sessions", response_model=AgentSessionListResponse)
def list_sessions(workspace=Depends(get_workspace)) -> AgentSessionListResponse:
    """List all agent sessions — in-memory active + on-disk historical.

    Active sessions take precedence over their on-disk metadata so the
    list reflects live status/stats. Historical sessions surviving a
    restart appear with whatever final status was flushed at termination.
    """
    sessions = [_refresh_response(s) for s in _sessions.values()]
    in_memory_ids = {s.sessionId for s in sessions}
    root = getattr(workspace, "root", None)
    if root is not None:
        for summary in list_persisted_sessions(root):
            if summary.session_id in in_memory_ids:
                continue
            sessions.append(_persisted_to_response(summary))
    return AgentSessionListResponse(sessions=sessions, total=len(sessions))


@router.get("/sessions/{session_id}", response_model=AgentSessionResponse)
def get_session(
    session_id: str,
    workspace=Depends(get_workspace),
) -> AgentSessionResponse:
    """Get a specific agent session — falls back to disk for historicals."""
    session = _sessions.get(session_id)
    if session:
        return _refresh_response(session)
    root = getattr(workspace, "root", None)
    if root is not None:
        from molexp.plugins.agent_pydanticai.sessions_store import get_persisted_session

        summary = get_persisted_session(root, session_id)
        if summary is not None:
            return _persisted_to_response(summary)
    raise HTTPException(status_code=404, detail=f"Session {session_id} not found")


@router.get("/sessions/{session_id}/events")
async def stream_events(session_id: str) -> StreamingResponse:
    """Stream agent session events via Server-Sent Events."""
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    async def generate() -> AsyncGenerator[str, None]:
        for event in session.events:
            data = json.dumps(event.model_dump())
            yield f"data: {data}\n\n"
        if session.status != "running":
            yield 'data: {"type": "done"}\n\n'
            return
        yield 'data: {"type": "waiting"}\n\n'

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/sessions/{session_id}/approve")
def respond_approval(
    session_id: str,
    request: ApprovalRespondRequest,
) -> dict:
    """Respond to a human-in-the-loop approval request."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return {"request_id": request.request_id, "approved": request.approved, "applied": True}


@router.post("/sessions/{session_id}/messages", response_model=MessageResponse)
async def post_user_message(
    session_id: str,
    request: UserMessageCreateRequest,
) -> MessageResponse:
    """Deliver a chat message from the user to a running agent session.

    Either resolves a pending ``UserMessageRequestEvent`` (when
    ``request_id`` is supplied and matches) or queues an unsolicited
    follow-up that re-prompts the agent with the user's content.
    """
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    live = _live_sessions.get(session_id)
    if live is None or not hasattr(live, "respond_user_message"):
        raise HTTPException(
            status_code=409,
            detail="Session is not interactive",
        )
    await live.respond_user_message(request.content, request.request_id)
    return MessageResponse(message="queued")


@router.post("/skills/{skill_id}/launch", response_model=AgentSessionResponse)
async def launch_skill(
    skill_id: str,
    request: SkillLaunchRequest,
    workspace=Depends(get_workspace),
) -> AgentSessionResponse:
    """Materialize a saved skill into a Goal and start a new session.

    The skill's ``instructions`` are threaded through to the runtime
    automatically; ``plan_mode`` defaults to the skill's
    ``default_plan_mode`` and may be overridden via the request body.
    """
    root = getattr(workspace, "root", None)
    if root is None:
        raise HTTPException(status_code=500, detail="Workspace has no root path")
    skill = SkillStore(root).get(skill_id)
    if skill is None:
        raise HTTPException(status_code=404, detail=f"Skill '{skill_id}' not found")
    try:
        rendered = skill.materialize(request.parameters)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not registry.is_available(Capability.AGENT):
        raise HTTPException(
            501, "Agent capability not available. Install: pip install molexp[agent]"
        )

    plan_mode = (
        request.plan_mode if request.plan_mode is not None else skill.default_plan_mode
    )
    AgentService = registry.get(Capability.AGENT)
    service = AgentService.from_workspace(str(workspace.root))
    goal = Goal(
        description=rendered["description"],
        constraints={"items": rendered["constraints"]} if rendered["constraints"] else {},
        success_criteria=rendered["success_criteria"],
        plan_mode=plan_mode,
        skill_id=skill.id,
        skill_instructions=skill.instructions,
    )
    session = await service.start(goal)
    response = _session_to_response(session, goal, created_at=_now_iso())
    _sessions[response.sessionId] = response
    _live_sessions[response.sessionId] = session
    return response


# ── Plan-mode follow-up + per-session prompt inspection ────────────────────


def _summary_text_from_events(response: AgentSessionResponse) -> str:
    """Pull the ``SessionCompletedEvent.summary`` from the buffered events.

    Returns ``""`` when the session has no completion event yet — callers
    handle that case as "no plan available".
    """
    for event in response.events:
        if event.type == "SessionCompletedEvent":
            payload = event.payload
            if isinstance(payload, dict):
                return str(payload.get("summary") or "")
    return ""


@router.post(
    "/sessions/{session_id}/execute-plan",
    response_model=AgentSessionResponse,
)
async def execute_plan(
    session_id: str,
    workspace=Depends(get_workspace),
) -> AgentSessionResponse:
    """Promote a finished plan-mode session into an executing follow-up.

    Inherits the original goal (description, constraints, success
    criteria, ``skill_id``); flips ``plan_mode`` off; injects the prior
    session's final answer (the plan) as ``instructions_override`` so the
    new session can execute it without re-deriving it.
    """
    response = _sessions.get(session_id)
    if response is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    response = _refresh_response(response)
    if not response.planMode:
        raise HTTPException(
            status_code=409,
            detail="Session was not started in plan mode.",
        )
    if response.status not in {"completed", "failed"}:
        raise HTTPException(
            status_code=409,
            detail=f"Plan-mode session is still {response.status}; wait for it to finish.",
        )

    plan_text = _summary_text_from_events(response)
    if not plan_text.strip():
        raise HTTPException(
            status_code=409,
            detail="Plan-mode session produced no plan to execute.",
        )

    live = _live_sessions.get(session_id)
    original_goal: Goal = getattr(live, "goal", None) if live is not None else None
    if original_goal is None:
        # Fallback to whatever the persisted summary keeps. Constraints and
        # success_criteria aren't on the AgentSessionResponse, so we accept
        # an empty seed — the plan itself carries the substance.
        original_goal = Goal(description=response.goalDescription)

    skill_instructions = _resolve_skill_instructions(workspace, response.skillId)
    follow_up = Goal(
        description=original_goal.description,
        constraints=dict(original_goal.constraints or {}),
        success_criteria=list(original_goal.success_criteria or []),
        plan_mode=False,
        instructions_override=(
            "An approved execution plan follows. Execute it step-by-step using "
            "the available tools, then report the outcome.\n\n" + plan_text.strip()
        ),
        skill_id=response.skillId,
        skill_instructions=skill_instructions,
    )

    if not registry.is_available(Capability.AGENT):
        raise HTTPException(
            501, "Agent capability not available. Install: pip install molexp[agent]"
        )
    _require_credentials(workspace)
    AgentService = registry.get(Capability.AGENT)
    service = AgentService.from_workspace(str(workspace.root))
    new_session = await service.start(follow_up)
    new_response = _session_to_response(new_session, follow_up, created_at=_now_iso())
    _sessions[new_response.sessionId] = new_response
    _live_sessions[new_response.sessionId] = new_session
    return new_response


@router.get(
    "/sessions/{session_id}/system-prompt",
    response_model=AgentSystemPromptResponse,
)
def get_session_system_prompt(
    session_id: str,
    workspace=Depends(get_workspace),
) -> AgentSystemPromptResponse:
    """Return the layered system prompt the session was started with.

    Live sessions report what the runtime actually composed (so live
    edits to workspace instructions don't drift the displayed value);
    historical (disk-only) sessions are re-composed from current
    workspace + persisted goal fields, which is a best-effort reflection.
    """
    live = _live_sessions.get(session_id)
    workspace_instructions = ""
    root = getattr(workspace, "root", None)
    if root is not None:
        try:
            workspace_instructions = ProviderStore(root).load().instructions
        except Exception:
            workspace_instructions = ""
    if live is not None:
        goal = getattr(live, "goal", Goal(description=""))
        effective = getattr(live, "system_prompt", "") or compose_system_prompt(
            base=BASE_SYSTEM_PROMPT,
            workspace_instructions=workspace_instructions,
            skill_instructions=goal.skill_instructions,
            session_override=goal.instructions_override,
            plan_mode=goal.plan_mode,
        )
        return AgentSystemPromptResponse(
            base=BASE_SYSTEM_PROMPT,
            workspaceInstructions=workspace_instructions,
            skillInstructions=goal.skill_instructions,
            sessionOverride=goal.instructions_override,
            planMode=goal.plan_mode,
            effective=effective,
        )

    # Disk-only fallback: re-derive from persisted goal fields.
    if root is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    from molexp.plugins.agent_pydanticai.sessions_store import (
        SESSIONS_DIR_NAME,
        METADATA_FILE,
    )

    meta_path = Path(root) / SESSIONS_DIR_NAME / session_id / METADATA_FILE
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    try:
        raw = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=500, detail=f"Could not read session metadata: {exc}"
        ) from exc
    goal_meta = raw.get("goal") or {}
    skill_instructions = str(goal_meta.get("skill_instructions") or "")
    session_override = goal_meta.get("instructions_override")
    plan_mode = bool(goal_meta.get("plan_mode", False))
    effective = compose_system_prompt(
        base=BASE_SYSTEM_PROMPT,
        workspace_instructions=workspace_instructions,
        skill_instructions=skill_instructions,
        session_override=session_override,
        plan_mode=plan_mode,
    )
    return AgentSystemPromptResponse(
        base=BASE_SYSTEM_PROMPT,
        workspaceInstructions=workspace_instructions,
        skillInstructions=skill_instructions,
        sessionOverride=session_override,
        planMode=plan_mode,
        effective=effective,
    )
