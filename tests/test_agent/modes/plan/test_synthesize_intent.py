"""``SynthesizeIntent`` / ``ClarifyIntent`` stage tests (ac-002).

The first two stages: project free-text ``user_input`` into a typed
``IntentSpec`` via the structured router, and route a plan with a
blocking ``MissingInfoItem`` to ``PlanState.needs_clarification``.
"""

from __future__ import annotations

import pytest

from molexp.agent.modes._planning import (
    IntentSpec,
    MissingInfoItem,
    PlanState,
    ResourceBudget,
    RiskLevel,
)
from molexp.agent.modes.plan.tasks_planning import (
    clarify_intent,
    synthesize_intent,
)
from molexp.agent.router import ModelTier

from .conftest import ScriptedStructuredRouter


def _intent(*, missing: tuple[MissingInfoItem, ...] = ()) -> IntentSpec:
    return IntentSpec(
        objective="run an MD simulation of water",
        non_goals=("no GPU tuning",),
        required_outputs=("trajectory",),
        constraints=(),
        assumptions=(),
        missing_information=missing,
        success_criteria=(),
        allowed_side_effects=(),
        budget=ResourceBudget(max_cost_usd=None, max_wall_seconds=None, model_tier=None),
        risk_level=RiskLevel.low,
    )


@pytest.mark.asyncio
async def test_synthesize_intent_returns_typed_intent_spec() -> None:
    router = ScriptedStructuredRouter(responses=[_intent()])
    intent = await synthesize_intent(router=router, user_input="simulate water")  # type: ignore[arg-type]
    assert isinstance(intent, IntentSpec)
    assert intent.objective == "run an MD simulation of water"


@pytest.mark.asyncio
async def test_synthesize_intent_uses_heavy_tier() -> None:
    router = ScriptedStructuredRouter(responses=[_intent()])
    await synthesize_intent(router=router, user_input="x")  # type: ignore[arg-type]
    assert router.calls[0]["tier"] is ModelTier.HEAVY
    assert router.calls[0]["schema"] is IntentSpec


@pytest.mark.asyncio
async def test_clarify_intent_no_blocking_items_stays_exploring() -> None:
    intent = _intent(missing=(MissingInfoItem(question="which forcefield?", blocking=False),))
    next_state, blocking = clarify_intent(intent=intent)
    assert next_state is PlanState.exploring
    assert blocking == ()


@pytest.mark.asyncio
async def test_clarify_intent_blocking_item_routes_to_needs_clarification() -> None:
    blocking_item = MissingInfoItem(question="what temperature?", blocking=True)
    intent = _intent(missing=(blocking_item,))
    next_state, blocking = clarify_intent(intent=intent)
    assert next_state is PlanState.needs_clarification
    assert blocking == (blocking_item,)


@pytest.mark.asyncio
async def test_clarify_intent_empty_missing_info_stays_exploring() -> None:
    next_state, blocking = clarify_intent(intent=_intent())
    assert next_state is PlanState.exploring
    assert blocking == ()
