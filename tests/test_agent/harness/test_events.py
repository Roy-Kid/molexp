"""Tests for ``molexp.agent.events`` — the ``AgentEvent`` union + sink bridge."""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter

from molexp.agent.events import (
    AgentEvent,
    ApprovalDecidedEvent,
    ApprovalRequestedEvent,
    ArtifactWrittenEvent,
    AsyncIteratorEventSink,
    CompactionPerformedEvent,
    ErrorEvent,
    LoopCompletedEvent,
    LoopStartedEvent,
    PlanEmittedEvent,
    PreflightFailedEvent,
    RepairProposedEvent,
    StageCompletedEvent,
    StageStartedEvent,
    ThinkingDeltaEvent,
    TokenDeltaEvent,
    ToolCallCompletedEvent,
    ToolCallStartedEvent,
)


class TestAgentEvent:
    def test_discriminator_routes_every_member_back_through_json(self) -> None:
        """Each union member round-trips to its concrete class via ``kind``."""
        adapter: TypeAdapter[AgentEvent] = TypeAdapter(AgentEvent)
        samples: list[AgentEvent] = [
            LoopStartedEvent(loop_name="chat", user_input="hi"),
            StageStartedEvent(stage_name="draft"),
            StageCompletedEvent(stage_name="draft"),
            ArtifactWrittenEvent(path="out.txt", description="result"),
            ApprovalRequestedEvent(gate="approve_direction", summary="check"),
            ApprovalDecidedEvent(gate="approve_direction", approved=True),
            PlanEmittedEvent(plan_id="p1", step_count=3),
            PreflightFailedEvent(failed_checks=("acyclic", "io")),
            RepairProposedEvent(failed_invariant="dag", rationale="fix"),
            CompactionPerformedEvent(summary="...", tokens_before=100, entries_summarized=4),
            LoopCompletedEvent(text="done"),
            ErrorEvent(message="boom", error_type="ValueError"),
            ThinkingDeltaEvent(text="reasoning"),
            TokenDeltaEvent(text="hel"),
            ToolCallStartedEvent(tool_name="read_file", args_summary="path=a.py"),
            ToolCallCompletedEvent(tool_name="read_file", result_summary="42 lines", ok=True),
        ]
        for ev in samples:
            loaded = adapter.validate_json(adapter.dump_json(ev))
            assert loaded.kind == ev.kind
            assert type(loaded) is type(ev)


class TestAsyncIteratorEventSink:
    @pytest.mark.asyncio
    async def test_close_sentinel_drains_buffered_events_then_stops(self) -> None:
        """Push N, ``close()``, then ``async for`` yields the N in push order."""
        sink = AsyncIteratorEventSink()
        pushed = [TokenDeltaEvent(text=f"e{i}") for i in range(5)]
        for ev in pushed:
            await sink(ev)
        await sink.close()

        collected: list[AgentEvent] = []
        async for ev in sink:
            collected.append(ev)
        assert len(collected) == 5
        assert [e.text for e in collected if isinstance(e, TokenDeltaEvent)] == [
            e.text for e in pushed
        ]
