"""Agent routes for MolExp API."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from molexp.plugins import Capability, registry
from molexp.plugins.agent_pydanticai.types import Goal

from ..dependencies import get_workspace
from ..schemas import (
    AgentSessionListResponse,
    AgentSessionResponse,
    ApprovalRespondRequest,
    GoalCreateRequest,
)

router = APIRouter(prefix="/agent", tags=["agent"])

# In-memory session store — NOT shared across workers.
# Deploy with a single uvicorn worker (--workers 1) or replace with
# an external store (Redis, database) before scaling horizontally.
_sessions: dict[str, AgentSessionResponse] = {}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@router.post("/sessions", response_model=AgentSessionResponse)
async def create_session(
    request: GoalCreateRequest,
    workspace=Depends(get_workspace),
) -> AgentSessionResponse:
    """Start a new agent session."""
    goal = Goal(
        description=request.description,
        constraints=request.constraints,
        success_criteria=request.success_criteria,
    )

    if not registry.is_available(Capability.AGENT):
        raise HTTPException(501, "Agent capability not available. Install: pip install molexp[agent]")

    try:
        AgentService = registry.get(Capability.AGENT)
        service = AgentService.from_workspace(str(workspace.root))
        session = await service.start(goal)
        response = AgentSessionResponse(
            sessionId=session.session_id,
            status=session.status,
            goalDescription=goal.description,
            createdAt=_now_iso(),
            events=[],
        )
    except NotImplementedError:
        raise HTTPException(501, "Agent runtime not yet implemented")

    _sessions[response.sessionId] = response
    return response


@router.get("/sessions", response_model=AgentSessionListResponse)
def list_sessions() -> AgentSessionListResponse:
    """List all agent sessions."""
    sessions = list(_sessions.values())
    return AgentSessionListResponse(sessions=sessions, total=len(sessions))


@router.get("/sessions/{session_id}", response_model=AgentSessionResponse)
def get_session(session_id: str) -> AgentSessionResponse:
    """Get a specific agent session."""
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return session


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
