"""``PlanGraph`` contract (spec ac-005 / behavior)."""

from __future__ import annotations

import pytest

from molexp.agent._pydantic_graph.plan_graph import PlanGraph, PlanGraphResult


def test_plan_graph_carries_config() -> None:
    graph = PlanGraph(max_iterations=3)
    assert graph.max_iterations == 3


def test_plan_graph_result_is_frozen() -> None:
    result = PlanGraphResult(plan={"step1": "intake"})
    with pytest.raises(Exception):
        result.plan = {}  # type: ignore[misc]


@pytest.mark.asyncio
async def test_plan_graph_run_returns_non_empty_plan() -> None:
    pytest.importorskip("pydantic_ai")
    from pydantic_ai.models.test import TestModel

    from molexp.agent._pydanticai.harness import PydanticAIHarness
    from molexp.agent.session import AgentSession

    harness = PydanticAIHarness(model=TestModel())  # type: ignore[arg-type]
    graph = PlanGraph()
    result = await graph.run(
        harness=harness,
        session=AgentSession(),
        user_input="plan something",
    )
    assert isinstance(result, PlanGraphResult)
    assert result.plan
