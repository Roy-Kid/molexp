"""Phase 2: native tools register on the AgentService-owned registry."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from molexp.agent import AgentService, Goal, ToolContext
from molexp.agent.orchestration import (
    ToolCallCompleted,
    UserMessageRequested,
)
from molexp.agent.testing import FakeModelClient


@pytest.fixture
def workspace_path(tmp_path: Path) -> Path:
    return tmp_path / "ws"


def test_native_tools_register_on_service_construction(workspace_path: Path) -> None:
    service = AgentService.from_workspace(workspace_path)
    names = {spec.name for spec in service.registry.list()}
    assert "native:list_projects" in names
    assert "native:create_project" in names
    assert "native:submit_run" in names
    assert "native:ask_user" in names
    # Decision O2 — exit_plan_mode is gone.
    assert "native:exit_plan_mode" not in names


def test_each_service_owns_its_own_registry_per_decision_t1(workspace_path: Path) -> None:
    a = AgentService.from_workspace(workspace_path / "a")
    b = AgentService.from_workspace(workspace_path / "b")
    assert a.registry is not b.registry
    assert len(a.registry) == len(b.registry)


def test_can_opt_out_of_native_tool_registration(workspace_path: Path) -> None:
    service = AgentService(
        workspace_path=workspace_path,
        register_native_tools=False,
    )
    assert len(service.registry) == 0


@pytest.mark.asyncio
async def test_ask_user_round_trip_via_chat_gateway(workspace_path: Path) -> None:
    model = FakeModelClient()
    model.queue_tool_call("native:ask_user", {"prompt": "more info?"}, call_id="c1")
    model.queue_text("got it, thanks")

    service = AgentService.from_workspace(workspace_path, model=model)
    session = service.start_session(Goal(description="ask me"))

    request_seen = asyncio.Event()
    captured: dict = {}

    async def watch() -> None:
        async for ev in session.stream_events():
            if isinstance(ev, UserMessageRequested):
                captured["request_id"] = ev.request_id
                captured["prompt"] = ev.prompt
                request_seen.set()
                break

    await asyncio.wait_for(watch(), timeout=2.0)
    assert captured["prompt"] == "more info?"

    completion_seen = asyncio.Event()

    async def wait_completion() -> None:
        async for ev in session.stream_events():
            if isinstance(ev, ToolCallCompleted) and ev.tool_name == "native:ask_user":
                completion_seen.set()
                break

    completion_task = asyncio.create_task(wait_completion())
    await session.send_user_message("here is the info", request_id=captured["request_id"])
    await asyncio.wait_for(completion_task, timeout=2.0)
    await service.cancel(session.session_id)
