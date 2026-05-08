"""Agent admin routes — Slice 1 stub.

The original endpoints managed ``ModelConfig`` / provider credentials —
all removed alongside the ``ModelClient`` boundary. The new
``AgentRunner`` takes a model string directly, so per-provider config
storage moves out of the server layer; restoration is tracked by the
``server-routes-agent-rectification`` follow-up spec.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

router = APIRouter(prefix="/api/agent/admin", tags=["agent-admin"])


_GONE_DETAIL = (
    "Agent admin routes are temporarily disabled; provider config moved "
    "into AgentRunner construction. See server-routes-agent-rectification."
)


@router.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    name="agent_admin_disabled",
)
async def agent_admin_disabled(path: str) -> None:
    raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=_GONE_DETAIL)


__all__ = ["router"]
