"""AgentService: entry point for goal-driven autonomous execution."""

from __future__ import annotations

from typing import Any

from molexp.workspace import Workspace

from .policy import ApprovalPolicy
from .runtime import AgentRuntime, create_default_agent_runtime
from .tools import Tool
from .types import AgentSession, Goal


class AgentService:
    """Service for creating and managing agent sessions.

    Example::

        service = AgentService.from_workspace("./lab")
        session = await service.run(goal)
        print(session.produced_runs)
    """

    def __init__(
        self,
        workspace: Any,
        runtime: AgentRuntime,
        extra_tools: list[Tool] | None = None,
        approval_policy: ApprovalPolicy | None = None,
    ) -> None:
        self._workspace = workspace
        self._runtime = runtime
        self._extra_tools: list[Tool] = extra_tools or []
        self._approval_policy = approval_policy or ApprovalPolicy()

    @classmethod
    def from_workspace(
        cls,
        workspace_path: str,
        extra_tools: list[Tool] | None = None,
        approval_policy: ApprovalPolicy | None = None,
    ) -> AgentService:
        """Create an AgentService from a workspace directory path."""
        workspace = Workspace(workspace_path)
        runtime = create_default_agent_runtime()
        return cls(
            workspace=workspace,
            runtime=runtime,
            extra_tools=extra_tools,
            approval_policy=approval_policy,
        )

    async def run(self, goal: Goal) -> AgentSession:
        """Run a goal to completion and return the completed session."""
        session = await self.start(goal)
        async for _ in session.stream_events():
            pass  # drain until SessionCompletedEvent
        return session

    async def start(self, goal: Goal) -> AgentSession:
        """Start a goal-driven agent session asynchronously."""
        return await self._runtime.start_session(
            goal=goal,
            workspace=self._workspace,
            extra_tools=self._extra_tools,
            approval_policy=self._approval_policy,
        )

    async def resume(self, session_id: str) -> AgentSession:
        """Resume a previously suspended session."""
        return await self._runtime.resume_session(
            session_id=session_id,
            workspace=self._workspace,
        )

    async def get_session_history(self, session_id: str) -> Any:
        """Retrieve the full event timeline for a completed session."""
        return await self._runtime.get_session_history(session_id)
