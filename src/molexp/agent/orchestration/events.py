"""Session event types + in-memory event bus (spec §6.5).

Per spec §6.5 the harness emits a fixed set of event categories. Each
event is a frozen dataclass; the bus is async-iterable so server SSE
routes can stream straight from it.

The event taxonomy here matches the migration table in §6.5: the old
``ApprovalRequestEvent`` becomes :class:`ToolApprovalRequested`, the
old ``ToolCallEvent`` becomes :class:`ToolCallRequested`, and so on.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncIterator

from molexp.agent.types import (
    AgentFailure,
    ArtifactRef,
    Usage,
    WorkflowPreview,
    utc_now,
)


@dataclass(frozen=True)
class SessionStarted:
    session_id: str
    goal_description: str
    ts: datetime = field(default_factory=utc_now)


@dataclass(frozen=True)
class TurnStarted:
    session_id: str
    turn_id: str
    ts: datetime = field(default_factory=utc_now)


@dataclass(frozen=True)
class ContextBuilt:
    turn_id: str
    used_chars: int
    diagnostics: tuple[str, ...] = ()
    ts: datetime = field(default_factory=utc_now)


@dataclass(frozen=True)
class ModelRequested:
    turn_id: str
    model_name: str
    metadata: dict[str, Any] = field(default_factory=dict)
    ts: datetime = field(default_factory=utc_now)


@dataclass(frozen=True)
class ModelResponded:
    turn_id: str
    finish_reason: str
    usage: Usage
    ts: datetime = field(default_factory=utc_now)


@dataclass(frozen=True)
class ToolCallRequested:
    turn_id: str
    call_id: str
    tool_name: str
    arguments: dict[str, Any]
    ts: datetime = field(default_factory=utc_now)


@dataclass(frozen=True)
class ToolApprovalRequested:
    turn_id: str
    request_id: str
    tool_name: str
    arguments: dict[str, Any]
    ts: datetime = field(default_factory=utc_now)


@dataclass(frozen=True)
class ToolCallCompleted:
    turn_id: str
    call_id: str
    tool_name: str
    ok: bool
    value: Any = None
    error: AgentFailure | None = None
    artifacts: tuple[ArtifactRef, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    ts: datetime = field(default_factory=utc_now)


@dataclass(frozen=True)
class PlanCreated:
    turn_id: str
    request_id: str
    plan_markdown: str
    workflow_preview: WorkflowPreview
    ts: datetime = field(default_factory=utc_now)


@dataclass(frozen=True)
class PlanDecided:
    request_id: str
    approved: bool
    feedback: str = ""
    edited_plan: str | None = None
    edited_workflow_ir: dict[str, Any] | None = None
    ts: datetime = field(default_factory=utc_now)


@dataclass(frozen=True)
class UserMessageRequested:
    request_id: str
    prompt: str
    ts: datetime = field(default_factory=utc_now)


@dataclass(frozen=True)
class UserMessageReceived:
    content: str
    request_id: str | None = None
    ts: datetime = field(default_factory=utc_now)


@dataclass(frozen=True)
class FailureRecorded:
    turn_id: str
    failure: AgentFailure
    ts: datetime = field(default_factory=utc_now)


@dataclass(frozen=True)
class SessionCompleted:
    session_id: str
    summary: str
    ts: datetime = field(default_factory=utc_now)


SessionEvent = (
    SessionStarted
    | TurnStarted
    | ContextBuilt
    | ModelRequested
    | ModelResponded
    | ToolCallRequested
    | ToolApprovalRequested
    | ToolCallCompleted
    | PlanCreated
    | PlanDecided
    | UserMessageRequested
    | UserMessageReceived
    | FailureRecorded
    | SessionCompleted
)


class EventBus:
    """Async fan-out queue for session events.

    A single bus is owned by ``AgentService`` per session; multiple
    consumers (server SSE, JSONL trace sink, in-memory tests) attach
    via :meth:`subscribe` and receive every event in order.
    """

    def __init__(self) -> None:
        self._subscribers: list[asyncio.Queue[SessionEvent | None]] = []
        self._closed = False

    async def publish(self, event: SessionEvent) -> None:
        if self._closed:
            return
        for queue in list(self._subscribers):
            await queue.put(event)

    async def close(self) -> None:
        self._closed = True
        for queue in list(self._subscribers):
            await queue.put(None)

    def subscribe(self) -> AsyncIterator[SessionEvent]:
        queue: asyncio.Queue[SessionEvent | None] = asyncio.Queue()
        self._subscribers.append(queue)
        return _drain(queue, lambda: self._subscribers.remove(queue))


async def _drain(
    queue: "asyncio.Queue[SessionEvent | None]",
    on_done,
) -> AsyncIterator[SessionEvent]:
    try:
        while True:
            event = await queue.get()
            if event is None:
                break
            yield event
    finally:
        try:
            on_done()
        except ValueError:
            # already removed
            pass
