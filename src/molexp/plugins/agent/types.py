"""Core types for the molexp agent layer."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator


@dataclass
class Goal:
    """A user-specified objective for autonomous agent execution."""
    description: str
    constraints: dict[str, Any] = field(default_factory=dict)
    success_criteria: list[str] = field(default_factory=list)


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


SessionEvent = (
    PlanCreatedEvent
    | ToolCallEvent
    | ToolResultEvent
    | WorkflowStartedEvent
    | ObservationEvent
    | ReplanEvent
    | ApprovalRequestEvent
    | SessionCompletedEvent
)


class AgentSession(ABC):
    """Handle for a running or completed agent session."""

    def __init__(self, session_id: str, goal: Goal) -> None:
        self.session_id = session_id
        self.goal = goal
        self.status: str = "running"
        self.produced_runs: list[Any] = []
        self.artifacts: list[Any] = []

    @abstractmethod
    async def stream_events(self) -> AsyncIterator[SessionEvent]:
        """Stream session events in real time."""
        ...

    @abstractmethod
    async def respond_approval(self, request_id: str, approved: bool) -> None:
        """Respond to a human-in-the-loop approval request."""
        ...
