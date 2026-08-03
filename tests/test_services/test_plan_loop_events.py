"""Plan loop event projection into agent-task transcripts."""

from __future__ import annotations

from pathlib import Path

import pytest

from molexp.agent.events import (
    LoopCompletedEvent,
    ThinkingDeltaEvent,
    ToolCallCompletedEvent,
    ToolCallStartedEvent,
)
from molexp.services.agent_task_store import (
    PersistedAgentTask,
    read_agent_task_events,
    write_agent_task_metadata,
)
from molexp.services.plan_runtime.loop_events import make_plan_loop_event_observer


@pytest.mark.asyncio
async def test_observer_projects_thinking_and_tools(tmp_path: Path) -> None:
    task_id = "task-stream-1"
    write_agent_task_metadata(
        tmp_path,
        PersistedAgentTask(
            task_id=task_id,
            session_id=task_id,
            title="t",
            goal="g",
            status="running",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
            plan_mode=True,
            active_mode="plan",
        ),
    )
    observe = make_plan_loop_event_observer(str(tmp_path), task_id, turn_id="turn-1")

    await observe(ThinkingDeltaEvent(text="reason "))
    await observe(ThinkingDeltaEvent(text="more"))
    # Force flush of coalesced buffer via a non-delta event.
    await observe(ToolCallStartedEvent(tool_name="place_task", args_summary="t1"))
    await observe(ToolCallCompletedEvent(tool_name="place_task", result_summary="ok", ok=True))
    # Terminal planning events must not close the agent-task turn early.
    await observe(LoopCompletedEvent(text="board done"))

    events = read_agent_task_events(tmp_path, task_id)
    kinds = [e["type"] for e in events]
    assert "thinking_delta" in kinds
    assert "tool_call_started" in kinds
    assert "tool_call_completed" in kinds
    assert "loop_completed" not in kinds
    thinking = next(e for e in events if e["type"] == "thinking_delta")
    assert "reason" in thinking["payload"]["text"]
    assert thinking["payload"]["turn_id"] == "turn-1"
