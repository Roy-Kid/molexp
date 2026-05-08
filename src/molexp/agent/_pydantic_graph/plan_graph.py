"""``PlanGraph`` — multi-step planning workflow built on pydantic-graph.

Sole ``import pydantic_graph`` site inside ``molexp.agent``. The graph
runs three nodes in sequence — intake → design → output — each calling
the harness to produce a structured plan section. The internal nodes
are private; ``PlanMode`` only sees :meth:`PlanGraph.run`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict
from pydantic_graph import BaseNode, End, Graph, GraphRunContext

if TYPE_CHECKING:
    from molexp.agent._pydanticai.harness import PydanticAIHarness
    from molexp.agent.session import AgentSession


class PlanGraphResult(BaseModel):
    """Outcome of one ``PlanGraph.run`` invocation."""

    model_config = ConfigDict(frozen=True)

    plan: dict[str, Any]
    summary: str = ""


@dataclass
class _PlanState:
    """Mutable state threaded through every plan-graph node."""

    user_input: str
    harness: Any  # PydanticAIHarness — type-checked at the call boundary
    session: Any  # AgentSession
    plan: dict[str, Any] = field(default_factory=dict)


@dataclass
class _IntakeNode(BaseNode[_PlanState, None, PlanGraphResult]):
    """Extract goals + constraints from the raw user input."""

    async def run(self, ctx: GraphRunContext[_PlanState, None]) -> _DesignNode:
        prompt = (
            "You are the intake step of a planning workflow. "
            "Extract the goal and any constraints from the user request below. "
            f"Request: {ctx.state.user_input}"
        )
        result = await ctx.state.harness.complete(prompt)
        ctx.state.plan["intake"] = result.text
        return _DesignNode()


@dataclass
class _DesignNode(BaseNode[_PlanState, None, PlanGraphResult]):
    """Draft a multi-step plan from the intake summary."""

    async def run(self, ctx: GraphRunContext[_PlanState, None]) -> _OutputNode:
        prompt = (
            "You are the design step of a planning workflow. "
            f"Given this intake summary: {ctx.state.plan.get('intake', '')}\n"
            "Produce a numbered plan."
        )
        result = await ctx.state.harness.complete(prompt)
        ctx.state.plan["design"] = result.text
        return _OutputNode()


@dataclass
class _OutputNode(BaseNode[_PlanState, None, PlanGraphResult]):
    """Assemble the final :class:`PlanGraphResult`."""

    async def run(self, ctx: GraphRunContext[_PlanState, None]) -> End[PlanGraphResult]:
        summary = str(ctx.state.plan.get("design", ""))
        return End(PlanGraphResult(plan=dict(ctx.state.plan), summary=summary))


class PlanGraph:
    """Encapsulates the three-step planning workflow.

    ``PlanMode`` is the only caller; users never touch this class.
    """

    def __init__(
        self,
        *,
        artifacts_root: Path | None = None,
        max_iterations: int = 8,
        temperature: float | None = None,
    ) -> None:
        self.artifacts_root = artifacts_root
        self.max_iterations = max_iterations
        self.temperature = temperature
        self._graph = Graph[_PlanState, None, PlanGraphResult](
            nodes=(_IntakeNode, _DesignNode, _OutputNode)
        )

    async def run(
        self,
        *,
        harness: PydanticAIHarness,
        session: AgentSession,
        user_input: str,
    ) -> PlanGraphResult:
        state = _PlanState(user_input=user_input, harness=harness, session=session)
        run_result = await self._graph.run(_IntakeNode(), state=state)
        return run_result.output


__all__ = ["PlanGraph", "PlanGraphResult"]
