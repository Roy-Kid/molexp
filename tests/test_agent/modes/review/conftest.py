"""Shared fixtures for the (slimmed) ReviewMode test suite.

The deep ReviewMode test files have been retired alongside the
``plan-mode-pydanticai-rewrite`` schema migration — the remaining tests
under this directory only need a couple of helpers to construct a
satisfying :class:`PlanGraph` and a shared :class:`IntentSpec`.
"""

from __future__ import annotations

from molexp.agent.modes._planning import (
    ApprovalGate,
    IntentConstraint,
    IntentSpec,
    IsolatedTestSketch,
    PlanCheck,
    PlanGraph,
    PlanState,
    PlanStep,
    PlanStepArtifact,
    PlanStepInput,
    PlanStepIO,
    ResourceBudget,
    RetryPolicy,
    RiskLevel,
    SuccessCriterion,
)


def make_intent() -> IntentSpec:
    """A shared :class:`IntentSpec` review fixtures judge plans against."""
    return IntentSpec(
        objective="produce a charge-density report for the molecule",
        non_goals=("retrain any model", "write to the production database"),
        required_outputs=("report.pdf", "density.npz"),
        constraints=(IntentConstraint(kind="time", detail="finish within one hour"),),
        assumptions=("the input geometry is already optimized",),
        missing_information=(),
        success_criteria=(
            SuccessCriterion(summary="report.pdf is generated", verifiable=True),
            SuccessCriterion(summary="density.npz is generated", verifiable=True),
        ),
        allowed_side_effects=("write files under the run directory",),
        budget=ResourceBudget(max_cost_usd=10.0, max_wall_seconds=3600.0, model_tier=None),
        risk_level=RiskLevel.low,
    )


def _step(
    step_id: str,
    *,
    depends_on: tuple[str, ...] = (),
    inputs: tuple[PlanStepInput, ...] = (),
    outputs: tuple[str, ...] = (),
    artifacts: tuple[PlanStepArtifact, ...] = (),
    api_refs: tuple[str, ...] = ("molpy.System",),
    checks: tuple[PlanCheck, ...] = (),
) -> PlanStep:
    return PlanStep(
        id=step_id,
        depends_on=depends_on,
        io=PlanStepIO(inputs=inputs, outputs=outputs),
        artifacts=artifacts,
        api_refs=api_refs,
        composition_notes="fixture step",
        checks=checks,
        retry_policy=RetryPolicy(max_attempts=1, on=()),
        rollback=None,
        approval_gate=ApprovalGate.approve_direction,
        estimated_cost_usd=None,
        risk_level=RiskLevel.low,
        unknowns=(),
        test_sketch=IsolatedTestSketch(
            is_isolated_testable=True,
            synthetic_inputs=(),
            assertion_sketch=(),
            rationale="",
        ),
    )


def make_satisfying_plan(*, state: PlanState = PlanState.approved) -> PlanGraph:
    """A :class:`PlanGraph` whose artefacts cover every required output."""
    compute = _step(
        "compute",
        outputs=("density",),
        artifacts=(PlanStepArtifact(path="density.npz", description="electron density"),),
        api_refs=("molpy.compute.density",),
    )
    render = _step(
        "render",
        depends_on=("compute",),
        inputs=(PlanStepInput(name="density", source_step="compute"),),
        outputs=("report",),
        artifacts=(PlanStepArtifact(path="report.pdf", description="charge-density report"),),
        api_refs=("molpy.report.render",),
    )
    return PlanGraph(
        plan_id="satisfying-plan",
        intent_ref="charge-density-intent",
        steps=(compute, render),
        state=state,
        compiled_contract_ref=None,
        notes="",
    )
