"""The typed orchestration-level ``AgentEvent`` stream.

An :data:`AgentEvent` is a discriminated union (pydantic
``Field(discriminator="kind")``) of frozen-pydantic event models. Each
member carries a ``kind`` :data:`typing.Literal`, a typed payload, and a
``timestamp`` sourced from :func:`molexp.agent.types.utc_now`.

These events describe **orchestration lifecycle** — a mode starting, a
stage opening/closing, an artefact landing, an approval being requested
or decided, a plan being emitted, a preflight failing, a repair being
proposed, a compaction running, a mode finishing, an error.

Four members carry the **emergent loop** of
:class:`~molexp.agent.loops.interactive.InteractiveLoop`: a reasoning-level
:class:`ThinkingDeltaEvent`, a token-level :class:`TokenDeltaEvent`, and the
:class:`ToolCallStartedEvent` / :class:`ToolCallCompletedEvent` pair. They are
the orchestration-level projection of the pydantic-ai agentic loop — the
per-call streaming machinery itself stays inside pydantic-ai, behind the
:meth:`~molexp.agent.router.Router.stream_agentic` surface.

The module imports nothing from ``pydantic_ai`` / ``pydantic_graph`` —
it is pure data.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from molexp.agent.types import utc_now

__all__ = [
    "AgentEvent",
    "AnyAgentEvent",
    "ApprovalDecidedEvent",
    "ApprovalRequestedEvent",
    "ArtifactWrittenEvent",
    "AsyncIteratorEventSink",
    "ClarificationRequiredEvent",
    "CompactionPerformedEvent",
    "ErrorEvent",
    "EventSink",
    "ModeCompletedEvent",
    "ModeStartedEvent",
    "PlanEmittedEvent",
    "PreflightFailedEvent",
    "RepairProposedEvent",
    "StageCompletedEvent",
    "StageStartedEvent",
    "ThinkingDeltaEvent",
    "TokenDeltaEvent",
    "ToolCallCompletedEvent",
    "ToolCallStartedEvent",
]

_FROZEN = ConfigDict(frozen=True, extra="forbid")


class _BaseEvent(BaseModel):
    """Common shape every :data:`AgentEvent` member shares.

    Subclasses pin ``kind`` to a unique :data:`typing.Literal` so the
    discriminated union can route a serialized payload back to its
    concrete class.
    """

    model_config = _FROZEN

    timestamp: datetime = Field(default_factory=utc_now)


class ModeStartedEvent(_BaseEvent):
    """Emitted once when a mode begins driving the harness."""

    kind: Literal["mode_started"] = "mode_started"
    mode_name: str
    user_input: str


class StageStartedEvent(_BaseEvent):
    """Emitted at the start of a logical stage in a mode's body."""

    kind: Literal["stage_started"] = "stage_started"
    stage_name: str


class StageCompletedEvent(_BaseEvent):
    """Emitted at the end of a logical stage in a mode's body."""

    kind: Literal["stage_completed"] = "stage_completed"
    stage_name: str


class ArtifactWrittenEvent(_BaseEvent):
    """Emitted when a mode materializes a file artefact."""

    kind: Literal["artifact_written"] = "artifact_written"
    path: str
    description: str = ""


class ApprovalRequestedEvent(_BaseEvent):
    """Emitted when the harness opens an approval gate for a reviewer."""

    kind: Literal["approval_requested"] = "approval_requested"
    gate: str
    summary: str = ""


class ApprovalDecidedEvent(_BaseEvent):
    """Emitted with the verdict once an approval gate resolves."""

    kind: Literal["approval_decided"] = "approval_decided"
    gate: str
    approved: bool
    reason: str = ""


class PlanEmittedEvent(_BaseEvent):
    """Emitted when a mode produces a plan graph.

    Carries a lightweight reference (``plan_id`` / ``step_count``)
    rather than the whole ``PlanGraph`` so the event stream stays cheap
    to serialize; consumers re-load the graph from disk by ``plan_id``.
    """

    kind: Literal["plan_emitted"] = "plan_emitted"
    plan_id: str
    step_count: int = 0


class PreflightFailedEvent(_BaseEvent):
    """Emitted when a plan's preflight checks fail.

    ``failed_checks`` holds the names of the failing ``PlanCheck``\\ s.
    """

    kind: Literal["preflight_failed"] = "preflight_failed"
    failed_checks: tuple[str, ...]


class RepairProposedEvent(_BaseEvent):
    """Emitted when the repair loop proposes a plan diff."""

    kind: Literal["repair_proposed"] = "repair_proposed"
    failed_invariant: str
    rationale: str = ""


class ClarificationRequiredEvent(_BaseEvent):
    """Emitted when an intake stage cannot proceed without user clarification.

    PlanMode's ``ClarifyIntent`` stage yields this when the intent spec
    carries unresolved ``MissingInfoItem``\\ s with ``blocking=True``;
    a registered :class:`~molexp.agent.repair.RepairPolicy`
    routes the pipeline to the ``needs_clarification`` terminal state.

    Attributes:
        questions: One-line concatenation of the blocking questions the
            user must answer before planning can resume.
    """

    kind: Literal["clarification_required"] = "clarification_required"
    questions: str


class CompactionPerformedEvent(_BaseEvent):
    """Emitted after the harness compacts the session entry tree."""

    kind: Literal["compaction_performed"] = "compaction_performed"
    summary: str
    tokens_before: int
    entries_summarized: int


class ModeCompletedEvent(_BaseEvent):
    """Terminal event — carries the run's final text + optional result.

    ``result`` is the JSON-mode dump of the terminal
    :class:`~molexp.agent.loop.AgentRunResult` (minus ``events`` to
    avoid recursion); the harness reconstructs the typed result from
    the accumulated stream.
    """

    kind: Literal["mode_completed"] = "mode_completed"
    text: str
    result: dict[str, Any] | None = None


class ErrorEvent(_BaseEvent):
    """Emitted when a stage or the whole run raises."""

    kind: Literal["error"] = "error"
    message: str
    error_type: str = ""
    stage_name: str = ""


class ThinkingDeltaEvent(_BaseEvent):
    """Emitted for one reasoning-token increment from the emergent loop.

    The orchestration-level projection of a
    :class:`~molexp.agent.router.ThinkingDeltaChunk`: a reasoning model's
    private chain-of-thought, streamed *before* the answer.
    :class:`~molexp.agent.loops.interactive.InteractiveLoop` yields one per
    reasoning delta so a CLI / SSE consumer can surface "thinking…" in a
    collapsed / dimmed treatment, kept distinct from the answer's
    :class:`TokenDeltaEvent`\\ s. A model that does not reason emits none.
    """

    kind: Literal["thinking_delta"] = "thinking_delta"
    text: str


class TokenDeltaEvent(_BaseEvent):
    """Emitted for one token-level text increment from the emergent loop.

    :class:`~molexp.agent.loops.interactive.InteractiveLoop` yields one
    of these per assistant text delta so a CLI / SSE consumer can render
    the reply as it streams. v1 keeps these in the accumulated
    :attr:`~molexp.agent.loop.AgentRunResult.events` stream unfiltered.
    """

    kind: Literal["token_delta"] = "token_delta"
    text: str


class ToolCallStartedEvent(_BaseEvent):
    """Emitted when the emergent loop dispatches a tool call.

    ``args_summary`` is a short human-readable rendering of the call
    arguments — never the full payload, so the event stream stays cheap.
    """

    kind: Literal["tool_call_started"] = "tool_call_started"
    tool_name: str
    args_summary: str = ""


class ToolCallCompletedEvent(_BaseEvent):
    """Emitted when a dispatched tool call returns.

    ``ok`` is ``False`` when the tool raised / returned a retry prompt;
    ``result_summary`` is a short rendering of the return value.
    """

    kind: Literal["tool_call_completed"] = "tool_call_completed"
    tool_name: str
    result_summary: str = ""
    ok: bool = True


AgentEvent = Annotated[
    ModeStartedEvent
    | StageStartedEvent
    | StageCompletedEvent
    | ArtifactWrittenEvent
    | ApprovalRequestedEvent
    | ApprovalDecidedEvent
    | PlanEmittedEvent
    | PreflightFailedEvent
    | RepairProposedEvent
    | ClarificationRequiredEvent
    | CompactionPerformedEvent
    | ModeCompletedEvent
    | ErrorEvent
    | ThinkingDeltaEvent
    | TokenDeltaEvent
    | ToolCallStartedEvent
    | ToolCallCompletedEvent,
    Field(discriminator="kind"),
]
"""Discriminated union of every orchestration-level harness event."""

AnyAgentEvent = AgentEvent
"""Alias kept for the spec's vocabulary (``AnyAgentEvent``)."""


EventSink = Callable[[AgentEvent], Awaitable[None]]
"""The emission callback shape the harness exposes — one async callable
that receives a single :data:`AgentEvent`."""


class _Sentinel:
    """Module-private singleton marker for :class:`AsyncIteratorEventSink`.

    Not an :data:`AgentEvent` subclass — :meth:`AsyncIteratorEventSink.__anext__`
    discriminates via ``is`` identity, never via type or attribute checks.
    """

    __slots__ = ()


_SENTINEL = _Sentinel()


class AsyncIteratorEventSink:
    """Queue-backed :data:`EventSink` that also exposes :class:`AsyncIterator`.

    Bridges the callable-sink Protocol with an async iterator so future
    ``async def mode.run(...) -> ArtifactRef`` flows can route AgentEvent
    through a side-channel: producers ``await sink(event)``; the runner
    drains via ``async for event in sink:``.

    Complementary to :class:`molexp.agent.runner._SinkCollector` (drain-
    after-yield list collector). The collector relies on the mode yielding
    periodically to trigger a drain; this sink is a live queue where push
    and consume run concurrently.

    Concurrency: safe for multi-task push within one event loop (the
    underlying :class:`asyncio.Queue` guarantee). Not safe across threads
    or processes.

    Lifecycle: callers terminate via :meth:`close` (enqueues a sentinel);
    the consumer's ``async for`` loop then drains buffered events and
    exits. Without :meth:`close`, a consumer parked in :meth:`__anext__`
    can be torn down with ``task.cancel()`` — the queue's get propagates
    :class:`asyncio.CancelledError` without swallowing.
    """

    def __init__(self, *, maxsize: int | None = None) -> None:
        """Build an empty sink.

        Args:
            maxsize: Queue capacity. ``None`` (default) → unbounded.
                A positive ``int`` → bounded; ``__call__`` awaits when the
                queue is full (asyncio.Queue native backpressure).
        """
        # ``asyncio.Queue(maxsize=0)`` is the stdlib's unbounded form.
        self._queue: asyncio.Queue[AgentEvent | _Sentinel] = asyncio.Queue(
            maxsize=maxsize if maxsize is not None else 0
        )

    async def __call__(self, event: AgentEvent) -> None:
        """:data:`EventSink` Protocol: producers push one event."""
        await self._queue.put(event)

    def __aiter__(self) -> AsyncIterator[AgentEvent]:
        return self

    async def __anext__(self) -> AgentEvent:
        item = await self._queue.get()
        if isinstance(item, _Sentinel):
            raise StopAsyncIteration
        return item

    async def close(self) -> None:
        """Enqueue the termination sentinel.

        After buffered events drain, the consumer's ``async for`` loop
        exits. Calling :meth:`close` more than once enqueues additional
        sentinels — harmless because the first one already terminates
        iteration; the extras stay buffered and unread.
        """
        await self._queue.put(_SENTINEL)
