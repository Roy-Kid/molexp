"""Core types for the molexp agent layer."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Literal


@dataclass
class Goal:
    """A user-specified objective for autonomous agent execution.

    The optional fields below carry the configuration that the chat box
    can attach to a goal:

    - ``plan_mode``: when True, the runtime registers only read-only tools
      and asks the agent to emit a structured plan instead of executing.
    - ``instructions_override``: replaces the layered system prompt for
      this single session (workspace + skill addenda are bypassed).
    - ``skill_id`` / ``skill_instructions``: populated when the goal was
      built from a slash command. ``skill_id`` is informational; the
      runtime uses ``skill_instructions`` to extend the system prompt.
    """

    description: str
    constraints: dict[str, Any] = field(default_factory=dict)
    success_criteria: list[str] = field(default_factory=list)
    plan_mode: bool = False
    instructions_override: str | None = None
    skill_id: str | None = None
    skill_instructions: str = ""


@dataclass
class SessionStats:
    """Live counters for a single agent session.

    Token fields mirror pydantic-ai's RunUsage; counts are accumulated
    across every model request the session performs.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    total_tokens: int = 0
    requests: int = 0
    tool_calls: int = 0
    events: int = 0
    started_at: datetime | None = None
    completed_at: datetime | None = None

    def duration_seconds(self) -> float | None:
        if self.started_at is None:
            return None
        end = self.completed_at or datetime.now(timezone.utc)
        return (end - self.started_at).total_seconds()


class ToolContext:
    """Context passed to every agent tool invocation."""

    def __init__(self, workspace: Any, run: Any, session: Any) -> None:
        self.workspace = workspace
        self.run = run
        self.session = session


@dataclass
class PlanCreatedEvent:
    plan_steps: list[str]
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ToolCallEvent:
    tool_name: str
    args: dict[str, Any]
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ToolResultEvent:
    tool_name: str
    result: Any
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class WorkflowStartedEvent:
    run_id: str
    workflow_id: str | None = None
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ObservationEvent:
    content: str
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ReplanEvent:
    reason: str
    new_plan: list[str] = field(default_factory=list)
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ApprovalRequestEvent:
    request_id: str
    tool_name: str
    args: dict[str, Any]
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class SessionCompletedEvent:
    summary: str
    produced_runs: list[Any] = field(default_factory=list)
    artifacts: list[Any] = field(default_factory=list)
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ResultArtifactEvent:
    """Inline artifact emitted by the agent (e.g. a Plotly chart, a table).

    The session detects tool results shaped as
    ``{"kind": "plot"|"table"|"text", ...}`` and converts them into this
    event so the UI can render the artifact inline in the event stream.
    """

    kind: Literal["plot", "table", "text"]
    title: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class UserMessageRequestEvent:
    """Agent → user prompt requesting input mid-session.

    Pairs with a UserMessageEvent reply identified by the same ``request_id``.
    """

    request_id: str
    prompt: str
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class UserMessageEvent:
    """User → agent chat message (reply to a request, or unsolicited follow-up)."""

    content: str
    request_id: str | None = None
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


SessionEvent = (
    PlanCreatedEvent
    | ToolCallEvent
    | ToolResultEvent
    | WorkflowStartedEvent
    | ObservationEvent
    | ReplanEvent
    | ApprovalRequestEvent
    | SessionCompletedEvent
    | ResultArtifactEvent
    | UserMessageRequestEvent
    | UserMessageEvent
)


class AgentSession(ABC):
    """Handle for a running or completed agent session."""

    def __init__(self, session_id: str, goal: Goal) -> None:
        self.session_id = session_id
        self.goal = goal
        self.status: str = "running"
        self.produced_runs: list[Any] = []
        self.artifacts: list[Any] = []
        self.stats: SessionStats = SessionStats()

    @abstractmethod
    def stream_events(self) -> AsyncIterator[SessionEvent]:
        """Stream session events in real time.

        Implementations are typically ``async def`` generators that ``yield`` events.
        Returning ``AsyncIterator`` (rather than declaring ``async def``) keeps the
        abstract type compatible with both async-generator and coroutine-returning-
        iterator implementations.
        """
        ...

    @abstractmethod
    async def respond_approval(self, request_id: str, approved: bool) -> None:
        """Respond to a human-in-the-loop approval request."""
        ...
