"""Plan-graph preflight tests (ac-004).

``run_plan_graph_preflight`` runs pure structural checks over a typed
``PlanGraph`` against its ``IntentSpec`` + ``CapabilityGraph``. Every
defect class — cycle, unconsumed output, unevidenced API binding,
unsatisfiable requirement, external-resource need, disallowed side
effect — must independently produce a failing report.
"""

from __future__ import annotations

from molexp.agent.modes._planning import (
    ApprovalGate,
    CapabilityGraph,
    CapabilityNode,
    EvidenceState,
    IntentSpec,
    PlanGraph,
    PlanState,
    PlanStep,
    PlanStepInput,
    PlanStepIO,
    ResourceBudget,
    RetryPolicy,
    RiskLevel,
)
from molexp.agent.modes.plan.plan_graph_preflight import (
    PlanGraphPreflightReport,
    run_plan_graph_preflight,
)


def _intent(
    *,
    required_outputs: tuple[str, ...] = ("trajectory",),
    allowed_side_effects: tuple[str, ...] = ("filesystem_write",),
) -> IntentSpec:
    return IntentSpec(
        objective="run MD",
        non_goals=(),
        required_outputs=required_outputs,
        constraints=(),
        assumptions=(),
        missing_information=(),
        success_criteria=(),
        allowed_side_effects=allowed_side_effects,
        budget=ResourceBudget(max_cost_usd=None, max_wall_seconds=None, model_tier=None),
        risk_level=RiskLevel.low,
    )


def _capabilities(*, evidenced: bool = True) -> CapabilityGraph:
    state = EvidenceState.evidenced if evidenced else EvidenceState.missing
    return CapabilityGraph(
        nodes=(
            CapabilityNode(
                id="cap_build",
                capability="build a system",
                evidence_state=state,
                confidence=0.9 if evidenced else 0.0,
                api_refs=("molpy.System",),
                usage_limits=(),
                needs_user_confirmation=False,
            ),
        ),
        edges=(),
    )


def _step(
    step_id: str,
    *,
    depends_on: tuple[str, ...] = (),
    inputs: tuple[PlanStepInput, ...] = (),
    outputs: tuple[str, ...] = (),
    capability_id: str | None = "cap_build",
    rollback: str | None = None,
) -> PlanStep:
    return PlanStep(
        id=step_id,
        depends_on=depends_on,
        io=PlanStepIO(inputs=inputs, outputs=outputs),
        artifacts=(),
        capability_id=capability_id,
        tool_binding=None,
        checks=(),
        retry_policy=RetryPolicy(max_attempts=1, on=()),
        rollback=rollback,
        approval_gate=ApprovalGate.approve_direction,
        estimated_cost_usd=None,
        risk_level=RiskLevel.low,
        unknowns=(),
    )


def _graph(steps: tuple[PlanStep, ...]) -> PlanGraph:
    return PlanGraph(
        plan_id="p1",
        intent_ref="i1",
        steps=steps,
        state=PlanState.draft_plan,
        compiled_contract_ref=None,
        notes="",
    )


# ── happy path ─────────────────────────────────────────────────────────────


def test_clean_plan_graph_passes_preflight() -> None:
    graph = _graph((_step("s1", outputs=("trajectory",)),))
    report = run_plan_graph_preflight(
        plan_graph=graph,
        intent=_intent(),
        capabilities=_capabilities(),
    )
    assert isinstance(report, PlanGraphPreflightReport)
    assert report.passed
    assert all(check.passed for check in report.checks)


# ── cycle ──────────────────────────────────────────────────────────────────


def test_cycle_fails_preflight() -> None:
    graph = _graph(
        (
            _step("s1", depends_on=("s2",), outputs=("trajectory",)),
            _step("s2", depends_on=("s1",)),
        )
    )
    report = run_plan_graph_preflight(
        plan_graph=graph,
        intent=_intent(),
        capabilities=_capabilities(),
    )
    assert not report.passed
    assert any("acyclic" in c.name and not c.passed for c in report.checks)


def test_dangling_dependency_fails_preflight() -> None:
    graph = _graph((_step("s1", depends_on=("missing",), outputs=("trajectory",)),))
    report = run_plan_graph_preflight(
        plan_graph=graph,
        intent=_intent(),
        capabilities=_capabilities(),
    )
    assert not report.passed
    assert any("closed" in c.name and not c.passed for c in report.checks)


# ── unconsumed output ──────────────────────────────────────────────────────


def test_unconsumed_output_fails_preflight() -> None:
    # s1 emits "scratch" — not consumed downstream and not a required output.
    graph = _graph(
        (
            _step("s1", outputs=("trajectory", "scratch")),
            _step(
                "s2",
                depends_on=("s1",),
                inputs=(PlanStepInput(name="trajectory", source_step="s1"),),
            ),
        )
    )
    report = run_plan_graph_preflight(
        plan_graph=graph,
        intent=_intent(),
        capabilities=_capabilities(),
    )
    assert not report.passed
    assert any("output" in c.name and not c.passed for c in report.checks)


# ── unevidenced API binding ────────────────────────────────────────────────


def test_unevidenced_capability_binding_fails_preflight() -> None:
    graph = _graph((_step("s1", outputs=("trajectory",), capability_id="cap_build"),))
    report = run_plan_graph_preflight(
        plan_graph=graph,
        intent=_intent(),
        capabilities=_capabilities(evidenced=False),
    )
    assert not report.passed
    assert any("evidenc" in c.name and not c.passed for c in report.checks)


def test_unknown_capability_binding_fails_preflight() -> None:
    graph = _graph((_step("s1", outputs=("trajectory",), capability_id="cap_ghost"),))
    report = run_plan_graph_preflight(
        plan_graph=graph,
        intent=_intent(),
        capabilities=_capabilities(),
    )
    assert not report.passed
    assert any("evidenc" in c.name and not c.passed for c in report.checks)


# ── unsatisfiable requirement ──────────────────────────────────────────────


def test_unsatisfiable_requirement_fails_preflight() -> None:
    # The plan never produces the "energy_report" the intent requires.
    graph = _graph((_step("s1", outputs=("trajectory",)),))
    report = run_plan_graph_preflight(
        plan_graph=graph,
        intent=_intent(required_outputs=("trajectory", "energy_report")),
        capabilities=_capabilities(),
    )
    assert not report.passed
    assert any("requirement" in c.name and not c.passed for c in report.checks)


# ── external resource ──────────────────────────────────────────────────────


def test_external_resource_need_is_flagged() -> None:
    cap = CapabilityGraph(
        nodes=(
            CapabilityNode(
                id="cap_build",
                capability="build a system",
                evidence_state=EvidenceState.evidenced,
                confidence=0.9,
                api_refs=("molpy.System",),
                usage_limits=("requires external LAMMPS install",),
                needs_user_confirmation=True,
            ),
        ),
        edges=(),
    )
    graph = _graph((_step("s1", outputs=("trajectory",)),))
    report = run_plan_graph_preflight(
        plan_graph=graph,
        intent=_intent(),
        capabilities=cap,
    )
    assert not report.passed
    assert any("external" in c.name and not c.passed for c in report.checks)


# ── disallowed side effect ─────────────────────────────────────────────────


def test_disallowed_side_effect_fails_preflight() -> None:
    # The step has a rollback (a side effect) but the intent allows none.
    graph = _graph((_step("s1", outputs=("trajectory",), rollback="delete written files"),))
    report = run_plan_graph_preflight(
        plan_graph=graph,
        intent=_intent(allowed_side_effects=()),
        capabilities=_capabilities(),
    )
    assert not report.passed
    assert any("side_effect" in c.name and not c.passed for c in report.checks)


def test_report_is_frozen() -> None:
    import pytest
    from pydantic import ValidationError

    report = run_plan_graph_preflight(
        plan_graph=_graph((_step("s1", outputs=("trajectory",)),)),
        intent=_intent(),
        capabilities=_capabilities(),
    )
    with pytest.raises(ValidationError):
        report.passed = False  # type: ignore[misc]
