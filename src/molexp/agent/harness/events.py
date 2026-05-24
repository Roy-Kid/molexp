"""Cluster 1 — the typed orchestration-level ``AgentEvent`` stream.

An :data:`AgentEvent` is a discriminated union (pydantic
``Field(discriminator="kind")``) of frozen-pydantic event models. Each
member carries a ``kind`` :data:`typing.Literal`, a typed payload, and a
``timestamp`` sourced from :func:`molexp.agent.types.utc_now`.

These events describe **orchestration lifecycle** — a mode starting, a
stage opening/closing, an artefact landing, an approval being requested
or decided, a plan being emitted, a preflight failing, a repair being
proposed, a compaction running, a mode finishing, an error.

Three members carry the **emergent loop** of
:class:`~molexp.agent.modes.interactive.InteractiveMode`: a token-level
:class:`TokenDeltaEvent` and the :class:`ToolCallStartedEvent` /
:class:`ToolCallCompletedEvent` pair. They are the orchestration-level
projection of the pydantic-ai agentic loop — the per-call streaming
machinery itself stays inside pydantic-ai, behind the
:meth:`~molexp.agent.router.Router.stream_agentic` surface.

The module imports nothing from ``pydantic_ai`` / ``pydantic_graph`` —
it is pure data.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
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
    """Emitted on entry to an ``AgentHarness.stage(name)`` context."""

    kind: Literal["stage_started"] = "stage_started"
    stage_name: str


class StageCompletedEvent(_BaseEvent):
    """Emitted on normal exit from an ``AgentHarness.stage(name)`` context."""

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
    a registered :class:`~molexp.agent.harness.repair.RepairPolicy`
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
    :class:`~molexp.agent.mode.AgentRunResult` (minus ``events`` to
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


class TokenDeltaEvent(_BaseEvent):
    """Emitted for one token-level text increment from the emergent loop.

    :class:`~molexp.agent.modes.interactive.InteractiveMode` yields one
    of these per assistant text delta so a CLI / SSE consumer can render
    the reply as it streams. v1 keeps these in the accumulated
    :attr:`~molexp.agent.mode.AgentRunResult.events` stream unfiltered.
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
