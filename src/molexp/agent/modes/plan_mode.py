"""``PlanMode`` — workflow-backed planning mode.

Drives a private :class:`PlanGraph` (which wraps pydantic-graph
internally). The user-facing surface is the :class:`AgentMode` ABC; the
graph itself never escapes ``molexp.agent``'s private subpackages.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from molexp.agent.mode import AgentMode, AgentRunResult
from molexp.agent.types import Message

if TYPE_CHECKING:
    from molexp.agent._pydanticai.harness import PydanticAIHarness
    from molexp.agent.session import AgentSession


class PlanModeConfig(BaseModel):
    """Tunables for :class:`PlanMode`."""

    model_config = ConfigDict(frozen=True)

    artifacts_root: Path | None = None
    max_iterations: int = 8
    temperature: float | None = None


class PlanMode(AgentMode):
    """Multi-step planning mode; dispatches into a private :class:`PlanGraph`."""

    name = "plan"

    def __init__(
        self,
        *,
        artifacts_root: Path | None = None,
        max_iterations: int = 8,
        temperature: float | None = None,
    ) -> None:
        self.config = PlanModeConfig(
            artifacts_root=artifacts_root,
            max_iterations=max_iterations,
            temperature=temperature,
        )

    async def run(
        self,
        *,
        harness: PydanticAIHarness,
        session: AgentSession,
        user_input: str,
    ) -> AgentRunResult:
        from molexp.agent._pydantic_graph.plan_graph import PlanGraph

        session.append(Message(role="user", content=user_input))
        graph = PlanGraph(
            artifacts_root=self.config.artifacts_root,
            max_iterations=self.config.max_iterations,
            temperature=self.config.temperature,
        )
        graph_result = await graph.run(
            harness=harness,
            session=session,
            user_input=user_input,
        )
        session.append(Message(role="assistant", content=graph_result.summary))
        session.mode_state["plan"] = graph_result.plan
        return AgentRunResult(
            text=graph_result.summary,
            messages=tuple(session.history),
            mode_state={"plan": graph_result.plan},
        )


__all__ = ["PlanMode", "PlanModeConfig"]
