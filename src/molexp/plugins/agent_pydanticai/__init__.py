"""AI agent plugin (PydanticAI-based autonomous execution).

Loaded lazily by :class:`~molexp.plugins.PluginRegistry`.
Requires ``pydantic-ai``: ``pip install molexp[agent]``.

Public API::

    from molexp.plugins.agent_pydanticai import AgentService, Goal
    service = AgentService.from_workspace("./lab")
    session = await service.run(Goal(description="..."))
"""

from .policy import ApprovalPolicy
from .runtime import AgentRuntime
from .service import AgentService
from .tools import Tool, agent_tool
from .types import (
    AgentSession,
    ApprovalRequestEvent,
    Goal,
    ObservationEvent,
    PlanCreatedEvent,
    ReplanEvent,
    SessionCompletedEvent,
    SessionEvent,
    ToolCallEvent,
    ToolContext,
    ToolResultEvent,
    WorkflowStartedEvent,
)

__all__ = [
    "AgentService",
    "AgentRuntime",
    "AgentSession",
    "ApprovalPolicy",
    "Goal",
    "Tool",
    "ToolContext",
    "agent_tool",
    "SessionEvent",
    "PlanCreatedEvent",
    "ToolCallEvent",
    "ToolResultEvent",
    "WorkflowStartedEvent",
    "ObservationEvent",
    "ReplanEvent",
    "ApprovalRequestEvent",
    "SessionCompletedEvent",
]


def get_agent_plugin():
    """Entry point for :class:`~molexp.plugins.PluginRegistry`."""
    import pydantic_ai  # noqa: F401 — availability check

    return AgentService
