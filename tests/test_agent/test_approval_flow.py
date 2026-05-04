"""Phase 2: HITL approval flow through the runner."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from molexp.agent import (
    AgentService,
    FailureKind,
    Goal,
    ToolContext,
    ToolResult,
    ToolSpec,
)
from molexp.agent.orchestration import (
    ToolApprovalRequested,
    ToolCallCompleted,
)
from molexp.agent.testing import FakeModelClient
from molexp.agent.tools import ApprovalDecision


@pytest.fixture
def workspace_path(tmp_path: Path) -> Path:
    return tmp_path / "ws"


@pytest.mark.asyncio
async def test_mutating_tool_emits_approval_event_and_waits(workspace_path: Path) -> None:
    model = FakeModelClient()
    model.queue_tool_call("native:test_mutate", {"v": 1}, call_id="c1")
    model.queue_text("ok")

    service = AgentService.from_workspace(workspace_path, model=model)

    async def stub(args: dict, ctx: ToolContext) -> ToolResult:
        return ToolResult(ok=True, value={"got": args})

    service.registry.register(
        ToolSpec(
            name="native:test_mutate",
            description="",
            input_schema={"type": "object", "properties": {}},
            mutates=True,
        ),
        stub,
    )

    session = service.start_session(Goal(description="approve me"))

    approval_seen: dict = {}
    approval_event = asyncio.Event()

    async def watch_approval() -> None:
        async for ev in session.stream_events():
            if isinstance(ev, ToolApprovalRequested):
                approval_seen["request_id"] = ev.request_id
                approval_seen["tool"] = ev.tool_name
                approval_event.set()
                break

    await asyncio.wait_for(watch_approval(), timeout=2.0)
    assert approval_seen["tool"] == "native:test_mutate"

    completion_event = asyncio.Event()

    async def watch_completion() -> None:
        async for ev in session.stream_events():
            if isinstance(ev, ToolCallCompleted) and ev.tool_name == "native:test_mutate":
                completion_event.set()
                break

    completion_task = asyncio.create_task(watch_completion())
    resolved = await session.respond_approval(
        ApprovalDecision(request_id=approval_seen["request_id"], approved=True)
    )
    assert resolved is True
    await asyncio.wait_for(completion_task, timeout=2.0)
    await service.cancel(session.session_id)


@pytest.mark.asyncio
async def test_denied_approval_returns_typed_failure(workspace_path: Path) -> None:
    model = FakeModelClient()
    model.queue_tool_call("native:test_mutate", {}, call_id="c1")
    model.queue_text("ok understood")

    service = AgentService.from_workspace(workspace_path, model=model)

    async def stub(args: dict, ctx: ToolContext) -> ToolResult:
        return ToolResult(ok=True)

    service.registry.register(
        ToolSpec(
            name="native:test_mutate",
            description="",
            input_schema={"type": "object", "properties": {}},
            mutates=True,
        ),
        stub,
    )

    session = service.start_session(Goal(description="reject me"))

    completion_event = asyncio.Event()
    captured: dict = {}

    async def watch() -> None:
        async for ev in session.stream_events():
            if isinstance(ev, ToolApprovalRequested):
                await session.respond_approval(
                    ApprovalDecision(
                        request_id=ev.request_id,
                        approved=False,
                        reason="nope",
                    )
                )
            if isinstance(ev, ToolCallCompleted):
                captured["ok"] = ev.ok
                captured["tool"] = ev.tool_name
                captured["error"] = ev.error
                completion_event.set()
                break

    await asyncio.wait_for(watch(), timeout=2.0)
    await service.cancel(session.session_id)

    assert captured["ok"] is False
    assert captured["error"] is not None
    assert captured["error"].kind is FailureKind.APPROVAL_DENIED
    assert "nope" in captured["error"].message
