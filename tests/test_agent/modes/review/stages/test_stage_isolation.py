"""Per-stage isolation tests for ReviewMode (ac-007)."""

from __future__ import annotations

from pathlib import Path

from molexp.agent.modes._planning import (
    CapabilityGraph,
    IntentSpec,
    PlanGraph,
    PlanState,
    ResourceBudget,
    RiskLevel,
)
from molexp.agent.modes.plan.plan_folder import PlanFolder
from molexp.agent.modes.review._mode import ReviewMode
from molexp.agent.modes.review.stages import (
    IngestReviewTarget,
    RenderReviewVerdict,
    RunReviewChecks,
)
from molexp.workspace import Workspace


def _minimal_review_mode(tmp_path: Path) -> ReviewMode:
    ws = Workspace(tmp_path / "lab")
    pf = ws.add_folder(PlanFolder(name="plan-1"))
    return ReviewMode(
        plan_folder=pf,
        intent=IntentSpec(
            objective="x",
            non_goals=(),
            required_outputs=(),
            constraints=(),
            assumptions=(),
            missing_information=(),
            success_criteria=(),
            allowed_side_effects=(),
            budget=ResourceBudget(max_cost_usd=None, max_wall_seconds=None, model_tier=None),
            risk_level=RiskLevel.low,
        ),
        plan_graph=PlanGraph(
            plan_id="p1",
            intent_ref="i1",
            steps=(),
            state=PlanState.draft_plan,
            compiled_contract_ref=None,
            notes="",
        ),
        capability_graph=CapabilityGraph(nodes=(), edges=()),
    )


def test_review_mode_pipeline_carries_three_stages_in_order(tmp_path: Path) -> None:
    mode = _minimal_review_mode(tmp_path)
    stages = mode.pipeline.stages
    assert len(stages) == 3
    assert isinstance(stages[0], IngestReviewTarget)
    assert isinstance(stages[1], RunReviewChecks)
    assert isinstance(stages[2], RenderReviewVerdict)
    assert tuple(s.name for s in stages) == (
        "IngestReviewTarget",
        "RunReviewChecks",
        "RenderReviewVerdict",
    )
