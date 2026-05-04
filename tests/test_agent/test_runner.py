"""Phase 1b/1c: AgentRunner end-to-end against FakeModelClient."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from molexp.agent import (
    AgentService,
    Goal,
    Message,
    SessionStatus,
    ToolContext,
    ToolResult,
    ToolSpec,
)
from molexp.agent.orchestration import (
    ModelResponded,
    PlanCreated,
    PlanDecided,
    SessionStarted,
    ToolCallCompleted,
    ToolCallRequested,
)
from molexp.agent.testing import FakeModelClient
from molexp.agent.types import AgentMode


def _spec(name: str, **kwargs) -> ToolSpec:
    base = {
        "name": name,
        "description": "",
        "input_schema": {"type": "object", "properties": {}},
    }
    base.update(kwargs)
    return ToolSpec(**base)


@pytest.fixture
def workspace_path(tmp_path: Path) -> Path:
    return tmp_path / "ws"


@pytest.mark.asyncio
async def test_text_only_turn_persists_messages_and_events(workspace_path: Path) -> None:
    model = FakeModelClient()
    model.queue_text("hello back")
    service = AgentService.from_workspace(
        workspace_path, model=model
    )
    session = service.start_session(Goal(description="say hi"))

    received = []

    async def collect() -> None:
        async for ev in session.stream_events():
            received.append(ev)
            if isinstance(ev, ModelResponded):
                break

    await asyncio.wait_for(collect(), timeout=2.0)
    # Drain the inbox so the runner doesn't sit blocked forever.
    await service.cancel(session.session_id)
    await asyncio.sleep(0)

    assert any(isinstance(e, SessionStarted) for e in received)
    assert any(isinstance(e, ModelResponded) for e in received)

    sessions_root = workspace_path / ".molexp-agent" / "sessions" / session.session_id
    messages_path = sessions_root / "messages.jsonl"
    events_path = sessions_root / "events.jsonl"
    assert messages_path.exists()
    assert events_path.exists()
    msgs = [json.loads(l) for l in messages_path.read_text().splitlines() if l.strip()]
    roles = [m["role"] for m in msgs]
    assert roles == ["user", "assistant"]
    assert msgs[0]["content"] == "say hi"
    assert msgs[1]["content"] == "hello back"


@pytest.mark.asyncio
async def test_tool_call_loop_dispatches_native_tool(workspace_path: Path) -> None:
    model = FakeModelClient()
    model.queue_tool_call("native:echo", {"x": 7}, call_id="c1")
    model.queue_text("done")

    service = AgentService.from_workspace(workspace_path, model=model)

    captured = {}

    async def echo(args: dict, ctx: ToolContext) -> ToolResult:
        captured["args"] = args
        captured["session"] = ctx.session_id
        return ToolResult(ok=True, value={"echoed": args})

    service.registry.register(_spec("native:echo"), echo)

    session = service.start_session(Goal(description="run echo"))

    seen_call = asyncio.Event()
    seen_result = asyncio.Event()

    async def collect() -> None:
        async for ev in session.stream_events():
            if isinstance(ev, ToolCallRequested):
                seen_call.set()
            if isinstance(ev, ToolCallCompleted):
                seen_result.set()
                break

    await asyncio.wait_for(collect(), timeout=2.0)
    await service.cancel(session.session_id)

    assert seen_call.is_set() and seen_result.is_set()
    assert captured["args"] == {"x": 7}

    msgs_path = (
        workspace_path
        / ".molexp-agent"
        / "sessions"
        / session.session_id
        / "messages.jsonl"
    )
    msgs = [json.loads(l) for l in msgs_path.read_text().splitlines() if l.strip()]
    roles = [m["role"] for m in msgs]
    # user goal, assistant tool-call (no text → no assistant), tool result, assistant final text
    assert "tool" in roles
    assert msgs[-1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_plan_mode_emits_and_resolves(workspace_path: Path) -> None:
    model = FakeModelClient()
    plan_text = (
        "Step 1: do the thing\n"
        '```json\n{"workflow_ir": {"task_configs": [{"task_id": "t1"}]}}\n```'
    )
    model.queue_text(plan_text)
    model.queue_text("Plan complete; proceeding.")

    service = AgentService.from_workspace(workspace_path, model=model)
    goal = Goal(description="plan something", mode=AgentMode.PLAN)
    session = service.start_session(goal)

    plan_created: list[PlanCreated] = []

    async def watch_plan() -> None:
        async for ev in session.stream_events():
            if isinstance(ev, PlanCreated):
                plan_created.append(ev)
                break

    await asyncio.wait_for(watch_plan(), timeout=2.0)
    assert plan_created, "PlanCreated should have fired"
    request_id = plan_created[0].request_id

    plan_decided: list[PlanDecided] = []

    async def watch_decided() -> None:
        async for ev in session.stream_events():
            if isinstance(ev, PlanDecided):
                plan_decided.append(ev)
                break

    decided_task = asyncio.create_task(watch_decided())
    await asyncio.sleep(0)
    ok = await session.respond_plan(request_id=request_id, approved=True)
    assert ok is True
    await asyncio.wait_for(decided_task, timeout=2.0)
    assert plan_decided[0].approved is True

    # Wait for the runner to finish the post-approval turn.
    await asyncio.sleep(0.05)
    await service.cancel(session.session_id)

    ckpt_dir = (
        workspace_path
        / ".molexp-agent"
        / "sessions"
        / session.session_id
        / "checkpoints"
    )
    assert ckpt_dir.exists()
    files = list(ckpt_dir.glob("*.json"))
    assert files, "plan-decision checkpoint should have been written"


@pytest.mark.asyncio
async def test_plan_rejection_loops_back_with_synthetic_user_message(
    workspace_path: Path,
) -> None:
    model = FakeModelClient()
    model.queue_text("draft plan v1")
    model.queue_text("revised plan v2")
    model.queue_text("ok done")

    service = AgentService.from_workspace(workspace_path, model=model)
    goal = Goal(description="plan it", mode=AgentMode.PLAN)
    session = service.start_session(goal)

    first_created: list[PlanCreated] = []

    async def wait_first() -> None:
        async for ev in session.stream_events():
            if isinstance(ev, PlanCreated):
                first_created.append(ev)
                break

    await asyncio.wait_for(wait_first(), timeout=2.0)
    assert await session.respond_plan(
        request_id=first_created[0].request_id,
        approved=False,
        feedback="be terser",
    )

    # Runner should emit PlanDecided then loop back; second PlanCreated arrives.
    second_created: list[PlanCreated] = []

    async def wait_second() -> None:
        async for ev in session.stream_events():
            if isinstance(ev, PlanCreated):
                second_created.append(ev)
                break

    await asyncio.wait_for(wait_second(), timeout=2.0)
    assert second_created, "model should be re-prompted with reject feedback"
    await service.cancel(session.session_id)

    msgs_path = (
        workspace_path
        / ".molexp-agent"
        / "sessions"
        / session.session_id
        / "messages.jsonl"
    )
    msgs = [json.loads(l) for l in msgs_path.read_text().splitlines() if l.strip()]
    user_msgs = [m for m in msgs if m["role"] == "user"]
    assert any("Plan rejected." in m["content"] for m in user_msgs)
    assert any("be terser" in m["content"] for m in user_msgs)


@pytest.mark.asyncio
async def test_no_model_skips_runner_keeps_phase_1a_surface(workspace_path: Path) -> None:
    """Phase 1a fixtures pass no model — start_session must stay synchronous."""

    service = AgentService.from_workspace(workspace_path)
    session = service.start_session(Goal(description="hi"))
    assert session.status is SessionStatus.PENDING
    assert session.task is None
