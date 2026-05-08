"""``AgentRunner`` — public orchestration entry point.

Takes a mode + a model string + optional tools/workspace; constructs the
private ``PydanticAIHarness`` lazily on first :meth:`run`; injects the
harness into the mode. Users never see the harness directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from molexp.agent.mode import AgentMode, AgentRunResult
    from molexp.agent.session import AgentSession


class AgentRunner:
    """Drive an ``AgentMode`` end-to-end.

    Construction performs no network IO — the underlying pydantic-ai
    Agent is built lazily on first :meth:`run`.
    """

    def __init__(
        self,
        *,
        mode: AgentMode,
        model: str,
        tools: tuple[Any, ...] = (),
        workspace: Path | None = None,
    ) -> None:
        self.mode = mode
        self.model = model
        self.tools = tools
        self.workspace = workspace
        self._harness: Any | None = None  # PydanticAIHarness; lazy.

    async def run(self, session: AgentSession, user_input: str) -> AgentRunResult:
        if self._harness is None:
            from molexp.agent._pydanticai.harness import PydanticAIHarness

            self._harness = PydanticAIHarness(
                model=self.model,
                tools=self.tools,
                workspace=self.workspace,
            )
        return await self.mode.run(
            harness=self._harness,
            session=session,
            user_input=user_input,
        )


__all__ = ["AgentRunner"]
