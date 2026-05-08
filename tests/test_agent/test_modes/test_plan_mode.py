"""``PlanMode`` unit tests (spec ac-009)."""

from __future__ import annotations

import pytest

from molexp.agent.modes import PlanMode, PlanModeConfig


def test_plan_mode_carries_config() -> None:
    mode = PlanMode(max_iterations=5)
    assert mode.name == "plan"
    assert mode.config.max_iterations == 5


def test_plan_mode_config_is_frozen() -> None:
    cfg = PlanModeConfig()
    with pytest.raises(Exception):
        cfg.max_iterations = 99  # type: ignore[misc]


@pytest.mark.asyncio
async def test_plan_mode_runs_full_graph() -> None:
    """ac-009 — PlanMode drives the multi-step graph and returns a plan."""

    pytest.importorskip("pydantic_ai")
    from pydantic_ai.models.test import TestModel

    from molexp.agent._pydanticai.harness import PydanticAIHarness
    from molexp.agent.mode import AgentRunResult
    from molexp.agent.session import AgentSession

    harness = PydanticAIHarness(model=TestModel())  # type: ignore[arg-type]
    mode = PlanMode()
    result = await mode.run(
        harness=harness,
        session=AgentSession(),
        user_input="design a workflow",
    )
    assert isinstance(result, AgentRunResult)
    assert result.mode_state is not None
    assert result.mode_state.get("plan")
