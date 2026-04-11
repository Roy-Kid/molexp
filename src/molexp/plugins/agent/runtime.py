"""Abstract AgentRuntime interface.

Internal adapter contract — users never see this.
Phase 2 will provide PydanticAIRuntime as the concrete implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .policy import ApprovalPolicy
from .tools import Tool
from .types import AgentSession, Goal


class AgentRuntime(ABC):
    """Abstract interface for agent execution backends."""

    @abstractmethod
    async def start_session(
        self,
        goal: Goal,
        workspace: Any,
        extra_tools: list[Tool],
        approval_policy: ApprovalPolicy,
    ) -> AgentSession: ...

    @abstractmethod
    async def resume_session(self, session_id: str, workspace: Any) -> AgentSession: ...

    @abstractmethod
    async def get_session_history(self, session_id: str) -> Any: ...


class _NotImplementedAgentRuntime(AgentRuntime):
    """Placeholder until Phase 2 PydanticAI implementation."""

    _MSG = (
        "AgentRuntime is not yet implemented. "
        "See Phase 2 of the pydantic-graph integration proposal."
    )

    async def start_session(self, goal: Goal, workspace: Any, extra_tools: list[Tool], approval_policy: ApprovalPolicy) -> AgentSession:
        raise NotImplementedError(self._MSG)

    async def resume_session(self, session_id: str, workspace: Any) -> AgentSession:
        raise NotImplementedError(self._MSG)

    async def get_session_history(self, session_id: str) -> Any:
        raise NotImplementedError(self._MSG)


def create_default_agent_runtime() -> AgentRuntime:
    """Instantiate the appropriate AgentRuntime.

    Returns PydanticAIRuntime (Phase 2 implementation).
    Override the model via molcfg config (key: ``agent.model``).
    """
    try:
        from molcfg import ConfigLoader, DictSource

        from ._pydantic_ai import PydanticAIRuntime

        sources = [DictSource({"agent": {"model": "anthropic:claude-sonnet-4-6"}})]
        config = ConfigLoader(sources).load()
        model = config["agent.model"]
        return PydanticAIRuntime(model=model)
    except ImportError:
        return _NotImplementedAgentRuntime()
