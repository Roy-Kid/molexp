"""Agent routes — Slice 1 stub.

The original endpoints depended on ``AgentService`` / ``ModelClient`` /
``Goal`` / ``coding_protocol`` — all removed by spec
``agent-pydanticai-as-core``. Server-side routing is rebuilt around
``AgentRunner`` in a follow-up spec (``server-routes-agent-rectification``).

This stub keeps ``from molexp.server.routes.agent import router`` working
so the rest of the API surface still mounts; every agent-route call
returns 503. The stub helpers below preserve the call-site signatures
that ``agent_tasks.py`` reaches for so static analysis stays clean
until the rebuild lands.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

if TYPE_CHECKING:
    from molexp.workspace.workspace import Workspace

    from ..schemas import (
        AgentSessionListResponse,
        AgentSessionResponse,
        ApprovalRespondRequest,
        GoalCreateRequest,
        MessageResponse,
        PlanDecisionRequest,
        UserMessageCreateRequest,
    )

router = APIRouter(prefix="/api/agent", tags=["agent"])


_GONE_DETAIL = (
    "Agent HTTP routes are temporarily disabled while the layer is rebuilt "
    "around AgentRunner; restoration is tracked by the server-routes-agent-"
    "rectification spec."
)


def _gone() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=_GONE_DETAIL,
    )


@router.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    name="agent_disabled",
)
async def agent_disabled(path: str) -> None:
    raise _gone()


# ── Stub call-site helpers ──────────────────────────────────────────────────
#
# ``agent_tasks.py`` calls these as if the legacy session router were still
# wired up. Each one raises 503 so the runtime behaviour is consistent with
# the catch-all route above; the explicit signatures keep the static type-
# checker happy until the rebuild lands.


async def create_session(
    request: GoalCreateRequest, *, workspace: Workspace
) -> AgentSessionResponse:
    raise _gone()


def list_sessions(*, workspace: Workspace) -> AgentSessionListResponse:
    raise _gone()


def get_session(session_id: str, *, workspace: Workspace) -> AgentSessionResponse:
    raise _gone()


async def stream_events(session_id: str, *, workspace: Workspace) -> StreamingResponse:
    raise _gone()


async def respond_approval(
    session_id: str, request: ApprovalRespondRequest, *, workspace: Workspace
) -> dict[str, object]:
    raise _gone()


async def respond_plan(
    session_id: str, request: PlanDecisionRequest, *, workspace: Workspace
) -> MessageResponse:
    raise _gone()


async def post_user_message(
    session_id: str, request: UserMessageCreateRequest, *, workspace: Workspace
) -> MessageResponse:
    raise _gone()


__all__ = [
    "create_session",
    "get_session",
    "list_sessions",
    "post_user_message",
    "respond_approval",
    "respond_plan",
    "router",
    "stream_events",
]
