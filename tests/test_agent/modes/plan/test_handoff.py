"""``ApprovedPlanHandoff`` contract tests (ac-001).

The handoff is PlanMode's sole terminal output object and the seam
AuthorMode imports. It must be a frozen pydantic model with
``extra="forbid"`` carrying exactly six fields.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from molexp.agent.modes._planning import (
    CapabilityGraph,
    IntentSpec,
    PlanGraph,
    PlanState,
    ResourceBudget,
    RiskLevel,
)
from molexp.agent.modes.plan import ApprovedPlanHandoff


def _intent() -> IntentSpec:
    return IntentSpec(
        objective="do a thing",
        non_goals=(),
        required_outputs=("result",),
        constraints=(),
        assumptions=(),
        missing_information=(),
        success_criteria=(),
        allowed_side_effects=(),
        budget=ResourceBudget(max_cost_usd=None, max_wall_seconds=None, model_tier=None),
        risk_level=RiskLevel.low,
    )


def _plan_graph() -> PlanGraph:
    return PlanGraph(
        plan_id="p1",
        intent_ref="i1",
        steps=(),
        state=PlanState.approved,
        compiled_contract_ref=None,
        notes="",
    )


def _capability_graph() -> CapabilityGraph:
    return CapabilityGraph(nodes=(), edges=())


def test_approved_plan_handoff_has_exactly_six_fields() -> None:
    fields = set(ApprovedPlanHandoff.model_fields)
    assert fields == {
        "plan_id",
        "intent",
        "plan_graph",
        "capability_graph",
        "plan_folder_path",
        "direction_approved_at",
    }


def test_approved_plan_handoff_constructs() -> None:
    handoff = ApprovedPlanHandoff(
        plan_id="p1",
        intent=_intent(),
        plan_graph=_plan_graph(),
        capability_graph=_capability_graph(),
        plan_folder_path=Path("/tmp/plans/p1"),
        direction_approved_at=datetime(2026, 5, 20, 12, 0, 0),
    )
    assert handoff.plan_id == "p1"
    assert handoff.intent.objective == "do a thing"
    assert handoff.plan_graph.state is PlanState.approved


def test_approved_plan_handoff_is_frozen() -> None:
    handoff = ApprovedPlanHandoff(
        plan_id="p1",
        intent=_intent(),
        plan_graph=_plan_graph(),
        capability_graph=_capability_graph(),
        plan_folder_path=Path("/tmp/plans/p1"),
        direction_approved_at=datetime(2026, 5, 20, 12, 0, 0),
    )
    with pytest.raises(ValidationError):
        handoff.plan_id = "p2"  # type: ignore[misc]


def test_approved_plan_handoff_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ApprovedPlanHandoff(
            plan_id="p1",
            intent=_intent(),
            plan_graph=_plan_graph(),
            capability_graph=_capability_graph(),
            plan_folder_path=Path("/tmp/plans/p1"),
            direction_approved_at=datetime(2026, 5, 20, 12, 0, 0),
            extra_field="nope",  # type: ignore[call-arg]
        )


def test_plan_run_handoff_is_gone() -> None:
    import molexp.agent.modes.plan as plan_pkg

    assert not hasattr(plan_pkg, "PlanRunHandoff")
    assert "ApprovedPlanHandoff" in plan_pkg.__all__
