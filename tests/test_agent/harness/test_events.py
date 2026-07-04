"""``AgentEvent`` discriminated-union tests (spec ac-001)."""

from __future__ import annotations

import asyncio

import pytest
from pydantic import TypeAdapter

from molexp.agent.events import (
    AgentEvent,
    ApprovalDecidedEvent,
    ApprovalRequestedEvent,
    ArtifactWrittenEvent,
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

ALL_EVENT_CLASSES = (
    LoopStartedEvent,
    StageStartedEvent,
    StageCompletedEvent,
    ArtifactWrittenEvent,
    ApprovalRequestedEvent,
    ApprovalDecidedEvent,
    PlanEmittedEvent,
    PreflightFailedEvent,
    RepairProposedEvent,
    CompactionPerformedEvent,
    LoopCompletedEvent,
    ErrorEvent,
    ThinkingDeltaEvent,
    TokenDeltaEvent,
    ToolCallStartedEvent,
    ToolCallCompletedEvent,
)

EXPECTED_KINDS = {
    "loop_started",
    "stage_started",
    "stage_completed",
    "artifact_written",
    "approval_requested",
    "approval_decided",
    "plan_emitted",
    "preflight_failed",
    "repair_proposed",
    "compaction_performed",
    "loop_completed",
    "error",
    "thinking_delta",
    "token_delta",
    "tool_call_started",
    "tool_call_completed",
}


def test_union_covers_all_sixteen_kinds() -> None:
    kinds = {cls.model_fields["kind"].default for cls in ALL_EVENT_CLASSES}
    assert kinds == EXPECTED_KINDS
    assert len(ALL_EVENT_CLASSES) == 16


def test_discriminated_union_round_trips_through_json() -> None:
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
        dumped = adapter.dump_json(ev)
        loaded = adapter.validate_json(dumped)
        assert loaded.kind == ev.kind
        assert type(loaded) is type(ev)


# ── AsyncIteratorEventSink (queue-backed bridge, ac-006..011) ──────────────


@pytest.mark.asyncio
async def test_async_iterator_event_sink_single_producer_ordered_drain() -> None:
    """Single producer push N → consumer async-for yields N in push order.

    Spec ac-007.
    """
    from molexp.agent.events import AsyncIteratorEventSink

    sink = AsyncIteratorEventSink()
    pushed = [TokenDeltaEvent(text=f"e{i}") for i in range(5)]
    for ev in pushed:
        await sink(ev)
    await sink.close()

    collected: list[AgentEvent] = []
    async for ev in sink:
        collected.append(ev)
    assert [e.text for e in collected if isinstance(e, TokenDeltaEvent)] == [e.text for e in pushed]
    assert len(collected) == 5


@pytest.mark.asyncio
async def test_async_iterator_event_sink_preserves_per_producer_order_under_concurrency() -> None:
    """Two concurrent producers; consumer sees 2*N events; per-producer order monotonic.

    Spec ac-008: cross-producer interleaving is allowed; per-producer order is
    strictly increasing in the producer's monotonic counter.
    """
    from molexp.agent.events import AsyncIteratorEventSink

    sink = AsyncIteratorEventSink()
    n = 100

    async def producer(label: str) -> None:
        for i in range(n):
            await sink(TokenDeltaEvent(text=f"{label}-{i:03d}"))

    async def consume() -> list[AgentEvent]:
        out: list[AgentEvent] = []
        async for ev in sink:
            out.append(ev)
        return out

    consumer_task = asyncio.create_task(consume())
    await asyncio.gather(producer("A"), producer("B"))
    await sink.close()
    collected = await asyncio.wait_for(consumer_task, timeout=2.0)

    assert len(collected) == 2 * n
    for label in ("A", "B"):
        subseq = [
            int(e.text.split("-")[1])
            for e in collected
            if isinstance(e, TokenDeltaEvent) and e.text.startswith(f"{label}-")
        ]
        assert subseq == list(range(n)), f"producer {label} out of order: {subseq[:5]}…"
