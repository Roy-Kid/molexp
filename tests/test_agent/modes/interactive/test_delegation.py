"""PlanMode delegation — ``/plan`` funnel + ``run_plan_pipeline`` (ac-006/ac-007).

A scripted structured router drives the real :class:`PlanMode` straight
to its ``needs_clarification`` short-path (one ``IntentSpec`` carrying a
blocking ``MissingInfoItem``) — enough to prove the delegation seam:
PlanMode runs on the parent harness and its events surface in the
parent stream.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from molexp.agent.harness.events import ModeCompletedEvent, ModeStartedEvent
from molexp.agent.harness.harness import AgentHarness
from molexp.agent.harness.session import Session
from molexp.agent.harness.session_storage import InMemorySessionStorage
from molexp.agent.modes._planning.intent import (
    IntentSpec,
    MissingInfoItem,
    ResourceBudget,
    RiskLevel,
)
from molexp.agent.modes.interactive.delegation import delegate_to_plan, run_plan_pipeline_tool
from molexp.agent.router import ModelTier, RouterTextResult
from molexp.agent.types import UsageBreakdown

pytestmark = pytest.mark.asyncio


def _blocking_intent() -> IntentSpec:
    """An IntentSpec with a blocking question — PlanMode stops at stage 2."""
    return IntentSpec(
        objective="run a molecular-dynamics experiment",
        non_goals=(),
        required_outputs=(),
        constraints=(),
        assumptions=(),
        missing_information=(
            MissingInfoItem(question="which force field should be used?", blocking=True),
        ),
        success_criteria=(),
        allowed_side_effects=(),
        budget=ResourceBudget(max_cost_usd=None, max_wall_seconds=None, model_tier=None),
        risk_level=RiskLevel.low,
    )


class _ClarifyRouter:
    """A ``Router`` stub driving PlanMode to ``needs_clarification``.

    ``complete_structured`` returns one ``IntentSpec`` with a blocking
    ``MissingInfoItem``; PlanMode then stops before capability
    exploration. Records the ``node_id`` of every structured call.
    """

    def __init__(self) -> None:
        self.structured_calls: list[str] = []

    async def complete_text(
        self,
        *,
        prompt: str,
        system: str = "",
        message_history: tuple[object, ...] = (),
        tier: ModelTier = ModelTier.DEFAULT,
    ) -> RouterTextResult:
        return RouterTextResult(text=f"echo:{prompt}")

    async def complete_structured(
        self,
        *,
        tier: ModelTier,
        system: str,
        user: str,
        schema: type,
        node_id: str = "",
    ) -> object:
        self.structured_calls.append(node_id)
        if schema is IntentSpec:
            return _blocking_intent()
        raise AssertionError(f"unexpected structured schema {schema.__name__}")

    def clear_usage(self) -> None:
        return None

    def snapshot_usage(self) -> UsageBreakdown:
        return UsageBreakdown()


def _harness(router: object) -> tuple[AgentHarness, list[object]]:
    events: list[object] = []

    async def sink(event: object) -> None:
        events.append(event)

    session = Session(storage=InMemorySessionStorage(), session_id="interactive-deleg")
    harness = AgentHarness(session=session, event_sink=sink, router=router)  # type: ignore[arg-type]
    return harness, events


async def test_delegate_to_plan_runs_planmode_on_the_parent_harness(tmp_path: Path) -> None:
    router = _ClarifyRouter()
    harness, events = _harness(router)

    summary = await delegate_to_plan(
        harness, "build an MD pipeline", workspace_root=tmp_path / "lab"
    )

    # PlanMode actually ran — its first structured stage fired.
    assert "SynthesizeIntent" in router.structured_calls
    # PlanMode's own events surfaced in the parent harness stream.
    assert any(isinstance(e, ModeStartedEvent) and e.mode_name == "plan" for e in events)
    assert any(isinstance(e, ModeCompletedEvent) for e in events)
    # the summary is non-empty and reflects the clarification outcome.
    assert summary
    assert "clarification" in summary.lower()


async def test_delegate_to_plan_persists_a_plan_folder(tmp_path: Path) -> None:
    router = _ClarifyRouter()
    harness, _ = _harness(router)

    await delegate_to_plan(harness, "study glass transition", workspace_root=tmp_path / "lab")

    plans_dir = tmp_path / "lab" / "plans"
    assert plans_dir.is_dir()
    intent_files = list(plans_dir.glob("*/intent.json"))
    assert intent_files, "PlanMode should have persisted a typed IntentSpec"


async def test_run_plan_pipeline_tool_delegates_and_returns_summary(tmp_path: Path) -> None:
    router = _ClarifyRouter()
    harness, events = _harness(router)

    tool = run_plan_pipeline_tool(harness, tmp_path / "lab")
    assert tool.__name__ == "run_plan_pipeline"

    summary = await tool("explore protein folding kinetics")

    assert summary
    assert "SynthesizeIntent" in router.structured_calls
    assert any(isinstance(e, ModeStartedEvent) and e.mode_name == "plan" for e in events)
