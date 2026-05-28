"""``AgentEvent`` discriminated-union tests (spec ac-001)."""

from __future__ import annotations

import asyncio
import inspect
from datetime import datetime

import pytest
from pydantic import TypeAdapter, ValidationError

from molexp.agent.events import (
    AgentEvent,
    ApprovalDecidedEvent,
    ApprovalRequestedEvent,
    ArtifactWrittenEvent,
    CompactionPerformedEvent,
    ErrorEvent,
    ModeCompletedEvent,
    ModeStartedEvent,
    PlanEmittedEvent,
    PreflightFailedEvent,
    RepairProposedEvent,
    StageCompletedEvent,
    StageStartedEvent,
    TokenDeltaEvent,
    ToolCallCompletedEvent,
    ToolCallStartedEvent,
)

ALL_EVENT_CLASSES = (
    ModeStartedEvent,
    StageStartedEvent,
    StageCompletedEvent,
    ArtifactWrittenEvent,
    ApprovalRequestedEvent,
    ApprovalDecidedEvent,
    PlanEmittedEvent,
    PreflightFailedEvent,
    RepairProposedEvent,
    CompactionPerformedEvent,
    ModeCompletedEvent,
    ErrorEvent,
    TokenDeltaEvent,
    ToolCallStartedEvent,
    ToolCallCompletedEvent,
)

EXPECTED_KINDS = {
    "mode_started",
    "stage_started",
    "stage_completed",
    "artifact_written",
    "approval_requested",
    "approval_decided",
    "plan_emitted",
    "preflight_failed",
    "repair_proposed",
    "compaction_performed",
    "mode_completed",
    "error",
    "token_delta",
    "tool_call_started",
    "tool_call_completed",
}


def test_union_covers_all_fifteen_kinds() -> None:
    kinds = {cls.model_fields["kind"].default for cls in ALL_EVENT_CLASSES}
    assert kinds == EXPECTED_KINDS
    assert len(ALL_EVENT_CLASSES) == 15


def test_each_event_carries_a_timestamp() -> None:
    ev = ModeStartedEvent(mode_name="chat", user_input="hi")
    assert isinstance(ev.timestamp, datetime)
    assert ev.timestamp.tzinfo is not None


def test_events_are_frozen() -> None:
    ev = StageStartedEvent(stage_name="draft")
    with pytest.raises(ValidationError):
        ev.stage_name = "other"  # type: ignore[misc]


def test_discriminated_union_round_trips_through_json() -> None:
    adapter: TypeAdapter[AgentEvent] = TypeAdapter(AgentEvent)
    samples: list[AgentEvent] = [
        ModeStartedEvent(mode_name="chat", user_input="hi"),
        StageStartedEvent(stage_name="draft"),
        StageCompletedEvent(stage_name="draft"),
        ArtifactWrittenEvent(path="out.txt", description="result"),
        ApprovalRequestedEvent(gate="approve_direction", summary="check"),
        ApprovalDecidedEvent(gate="approve_direction", approved=True),
        PlanEmittedEvent(plan_id="p1", step_count=3),
        PreflightFailedEvent(failed_checks=("acyclic", "io")),
        RepairProposedEvent(failed_invariant="dag", rationale="fix"),
        CompactionPerformedEvent(summary="...", tokens_before=100, entries_summarized=4),
        ModeCompletedEvent(text="done"),
        ErrorEvent(message="boom", error_type="ValueError"),
        TokenDeltaEvent(text="hel"),
        ToolCallStartedEvent(tool_name="read_file", args_summary="path=a.py"),
        ToolCallCompletedEvent(tool_name="read_file", result_summary="42 lines", ok=True),
    ]
    for ev in samples:
        dumped = adapter.dump_json(ev)
        loaded = adapter.validate_json(dumped)
        assert loaded.kind == ev.kind
        assert type(loaded) is type(ev)


def test_discriminator_selects_concrete_class() -> None:
    adapter: TypeAdapter[AgentEvent] = TypeAdapter(AgentEvent)
    payload = {"kind": "error", "message": "x", "error_type": "RuntimeError"}
    loaded = adapter.validate_python(payload)
    assert isinstance(loaded, ErrorEvent)
    assert loaded.message == "x"


def test_mode_completed_carries_optional_result_payload() -> None:
    ev = ModeCompletedEvent(text="done", result={"mode_state": {"k": 1}})
    assert ev.result == {"mode_state": {"k": 1}}


# ── AsyncIteratorEventSink (queue-backed bridge, ac-006..011) ──────────────


def test_async_iterator_event_sink_module_surface() -> None:
    """`AsyncIteratorEventSink` is module-scope in events.py and in `__all__`.

    Spec ac-006: exposes `__call__` (async), `__aiter__`, `__anext__` (async),
    `close` (async).
    """
    from molexp.agent import events as events_mod

    assert "AsyncIteratorEventSink" in events_mod.__all__
    cls = events_mod.AsyncIteratorEventSink
    assert inspect.isclass(cls)
    assert inspect.iscoroutinefunction(cls.__call__)
    assert hasattr(cls, "__aiter__")
    assert inspect.iscoroutinefunction(cls.__anext__)
    assert inspect.iscoroutinefunction(cls.close)


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


@pytest.mark.asyncio
async def test_async_iterator_event_sink_close_terminates_async_for() -> None:
    """`close()` after K pushes → async-for yields exactly K and returns within 1s.

    Spec ac-009.
    """
    from molexp.agent.events import AsyncIteratorEventSink

    sink = AsyncIteratorEventSink()
    k = 3
    for i in range(k):
        await sink(TokenDeltaEvent(text=f"t{i}"))
    await sink.close()

    async def drain() -> int:
        count = 0
        async for _ in sink:
            count += 1
        return count

    count = await asyncio.wait_for(drain(), timeout=1.0)
    assert count == k


@pytest.mark.asyncio
async def test_async_iterator_event_sink_cancelled_consumer_raises() -> None:
    """Consumer parked in `__anext__` and `cancel()`ed raises CancelledError.

    Spec ac-010: cancellation is observable to the awaiter; not silently
    swallowed by `__anext__`.
    """
    from molexp.agent.events import AsyncIteratorEventSink

    sink = AsyncIteratorEventSink()

    async def drain() -> None:
        async for _ in sink:
            pass  # would never end without close — we cancel below

    task = asyncio.create_task(drain())
    await asyncio.sleep(0.05)  # let the task park in __anext__
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_async_iterator_event_sink_bounded_backpressure() -> None:
    """maxsize=1 → producer's 2nd push blocks until consumer drains.

    Spec ac-011: producer's second `await sink(event)` does not complete
    within 200ms when no consumer drains; the producer task stays PENDING.
    """
    from molexp.agent.events import AsyncIteratorEventSink

    sink = AsyncIteratorEventSink(maxsize=1)
    await sink(TokenDeltaEvent(text="first"))

    async def push_second() -> None:
        await sink(TokenDeltaEvent(text="second"))

    second_task = asyncio.create_task(push_second())
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(asyncio.shield(second_task), timeout=0.2)
    assert not second_task.done()
    second_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await second_task
