"""Session event types + in-memory event bus.

The harness emits a fixed set of event categories. Each event is a
frozen ``BaseModel`` carrying a ``kind: Literal[...]`` discriminator;
the bus is async-iterable so server SSE routes can stream straight
from it. ``SESSION_EVENT_ADAPTER`` round-trips events through JSON
without manual dispatch.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Annotated, Any, AsyncIterator, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from molexp.agent.types import (
    AgentFailure,
    Usage,
    utc_now,
)
from molexp.workflow import PlanProposal, WorkflowPreviewView

_FROZEN = ConfigDict(frozen=True)


class SessionStarted(BaseModel):
    model_config = _FROZEN
    kind: Literal["session_started"] = "session_started"
    session_id: str
    goal_description: str
    ts: datetime = Field(default_factory=utc_now)


class TurnStarted(BaseModel):
    model_config = _FROZEN
    kind: Literal["turn_started"] = "turn_started"
    session_id: str
    turn_id: str
    ts: datetime = Field(default_factory=utc_now)


class ContextBuilt(BaseModel):
    model_config = _FROZEN
    kind: Literal["context_built"] = "context_built"
    turn_id: str
    used_chars: int
    diagnostics: tuple[str, ...] = ()
    ts: datetime = Field(default_factory=utc_now)


class ModelRequested(BaseModel):
    model_config = _FROZEN
    kind: Literal["model_requested"] = "model_requested"
    turn_id: str
    model_name: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    ts: datetime = Field(default_factory=utc_now)


class ModelResponded(BaseModel):
    model_config = _FROZEN
    kind: Literal["model_responded"] = "model_responded"
    turn_id: str
    finish_reason: str
    usage: Usage
    ts: datetime = Field(default_factory=utc_now)


class ToolCallRequested(BaseModel):
    model_config = _FROZEN
    kind: Literal["tool_call_requested"] = "tool_call_requested"
    turn_id: str
    call_id: str
    tool_name: str
    arguments: dict[str, Any]
    ts: datetime = Field(default_factory=utc_now)


class ToolApprovalRequested(BaseModel):
    model_config = _FROZEN
    kind: Literal["tool_approval_requested"] = "tool_approval_requested"
    turn_id: str
    request_id: str
    tool_name: str
    arguments: dict[str, Any]
    ts: datetime = Field(default_factory=utc_now)


class ToolCallCompleted(BaseModel):
    model_config = _FROZEN
    kind: Literal["tool_call_completed"] = "tool_call_completed"
    turn_id: str
    call_id: str
    tool_name: str
    ok: bool
    value: Any = None
    error: AgentFailure | None = None
    artifacts: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)
    ts: datetime = Field(default_factory=utc_now)


class PlanCreated(BaseModel):
    model_config = _FROZEN
    kind: Literal["plan_created"] = "plan_created"
    turn_id: str
    request_id: str
    plan_markdown: str
    workflow_preview: WorkflowPreviewView
    ts: datetime = Field(default_factory=utc_now)


class PlanDecided(BaseModel):
    model_config = _FROZEN
    kind: Literal["plan_decided"] = "plan_decided"
    request_id: str
    approved: bool
    feedback: str = ""
    edited_plan: str | None = None
    edited_proposal: PlanProposal | None = None
    ts: datetime = Field(default_factory=utc_now)


class UserMessageRequested(BaseModel):
    model_config = _FROZEN
    kind: Literal["user_message_requested"] = "user_message_requested"
    request_id: str
    prompt: str
    ts: datetime = Field(default_factory=utc_now)


class UserMessageReceived(BaseModel):
    model_config = _FROZEN
    kind: Literal["user_message_received"] = "user_message_received"
    content: str
    request_id: str | None = None
    ts: datetime = Field(default_factory=utc_now)


class FailureRecorded(BaseModel):
    model_config = _FROZEN
    kind: Literal["failure_recorded"] = "failure_recorded"
    turn_id: str
    failure: AgentFailure
    ts: datetime = Field(default_factory=utc_now)


class SessionCompleted(BaseModel):
    model_config = _FROZEN
    kind: Literal["session_completed"] = "session_completed"
    session_id: str
    summary: str
    ts: datetime = Field(default_factory=utc_now)


SessionEvent = Annotated[
    Union[
        SessionStarted,
        TurnStarted,
        ContextBuilt,
        ModelRequested,
        ModelResponded,
        ToolCallRequested,
        ToolApprovalRequested,
        ToolCallCompleted,
        PlanCreated,
        PlanDecided,
        UserMessageRequested,
        UserMessageReceived,
        FailureRecorded,
        SessionCompleted,
    ],
    Field(discriminator="kind"),
]
"""Discriminated union of every session event the harness emits."""


SESSION_EVENT_ADAPTER: TypeAdapter[SessionEvent] = TypeAdapter(SessionEvent)
"""Module-level TypeAdapter for round-tripping events through JSON."""


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
