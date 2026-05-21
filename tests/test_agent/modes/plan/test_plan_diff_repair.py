"""Plan-diff repair-loop tests (ac-006).

A preflight failure or a rejected ``approve_direction`` gate produces a
``PlanDiff`` applied via ``PlanDiff.apply``; ``state.py``'s
``RepairSignal`` / ``PlanRuntimeState`` carry a ``PlanDiff`` payload.
"""

from __future__ import annotations

from molexp.agent.modes._planning import (
    ApprovalGate,
    CapabilityGraph,
    CapabilityNode,
    DiffOpKind,
    EvidenceState,
    IntentSpec,
    IsolatedTestSketch,
    PlanGraph,
    PlanNodeOp,
    PlanState,
    PlanStep,
    PlanStepIO,
    ResourceBudget,
    RetryPolicy,
    RiskLevel,
)
from molexp.agent.modes.plan.plan_graph_preflight import run_plan_graph_preflight
from molexp.agent.modes.plan.state import PlanRuntimeState, RepairSignal
from molexp.agent.modes.plan.tasks_planning import build_repair_diff


def _intent() -> IntentSpec:
    return IntentSpec(
        objective="run MD",
        non_goals=(),
        required_outputs=("trajectory",),
        constraints=(),
        assumptions=(),
        missing_information=(),
        success_criteria=(),
        allowed_side_effects=(),
        budget=ResourceBudget(max_cost_usd=None, max_wall_seconds=None, model_tier=None),
        risk_level=RiskLevel.low,
    )


def _capabilities() -> CapabilityGraph:
    return CapabilityGraph(
        nodes=(
            CapabilityNode(
                id="cap_md",
                capability="run MD",
                evidence_state=EvidenceState.evidenced,
                confidence=0.9,
                api_refs=("molpy.run_md",),
                usage_limits=(),
                needs_user_confirmation=False,
            ),
        ),
        edges=(),
    )


def _step(step_id: str, *, capability_id: str | None = "cap_md") -> PlanStep:
    return PlanStep(
        id=step_id,
        depends_on=(),
        io=PlanStepIO(inputs=(), outputs=("trajectory",)),
        artifacts=(),
        capability_id=capability_id,
        tool_binding=None,
        checks=(),
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


def _plan(steps: tuple[PlanStep, ...]) -> PlanGraph:
    return PlanGraph(
        plan_id="p1",
        intent_ref="i1",
        steps=steps,
        state=PlanState.draft_plan,
        compiled_contract_ref=None,
        notes="",
    )


# ── RepairSignal / PlanRuntimeState carry a PlanDiff ───────────────────────


def test_runtime_state_default_has_no_diff() -> None:
    state = PlanRuntimeState()
    assert state.iteration == 0
    assert state.repair_signal is None


def test_repair_signal_carries_plan_diff() -> None:
    diff = build_repair_diff(
        failed_invariant="capability_evidenced",
        plan_graph=_plan((_step("s1", capability_id="cap_ghost"),)),
        rationale="rebind to an evidenced capability",
        operations=(PlanNodeOp(kind=DiffOpKind.replace, node_id="s1", step=_step("s1")),),
    )
    signal = RepairSignal(plan_diff=diff)
    assert signal.plan_diff.failed_invariant == "capability_evidenced"
    assert signal.plan_diff.operations[0].kind is DiffOpKind.replace


def test_runtime_state_records_repair_signal() -> None:
    diff = build_repair_diff(
        failed_invariant="x",
        plan_graph=_plan((_step("s1"),)),
        rationale="r",
        operations=(),
    )
    state = PlanRuntimeState(iteration=1, repair_signal=RepairSignal(plan_diff=diff))
    assert state.repair_signal is not None
    assert state.repair_signal.plan_diff.rationale == "r"


# ── repair diff fixes a failing preflight ──────────────────────────────────


def test_repair_diff_applied_fixes_preflight() -> None:
    # A plan binding an unknown capability fails preflight.
    bad = _plan((_step("s1", capability_id="cap_ghost"),))
    report = run_plan_graph_preflight(
        plan_graph=bad,
        intent=_intent(),
        capabilities=_capabilities(),
    )
    assert not report.passed

    # Build a diff that replaces the step with one bound to an evidenced cap.
    diff = build_repair_diff(
        failed_invariant="capability_evidenced",
        plan_graph=bad,
        rationale="rebind to cap_md",
        operations=(
            PlanNodeOp(
                kind=DiffOpKind.replace,
                node_id="s1",
                step=_step("s1", capability_id="cap_md"),
            ),
        ),
    )
    repaired = diff.apply(bad)
    repaired_report = run_plan_graph_preflight(
        plan_graph=repaired,
        intent=_intent(),
        capabilities=_capabilities(),
    )
    assert repaired_report.passed
