"""Candidate synthesis + selection tests (ac-005).

Stage 4 (``SynthesizeCandidates``) emits one candidate ``PlanGraph`` for
a simple task and three (A/B/C) with a self-critique for a complex one;
stage 5 (``SelectPlan``) chooses one whose every ``PlanStep`` binds to a
``CapabilityGraph`` node.
"""

from __future__ import annotations

import pytest

from molexp.agent.modes._planning import (
    ApprovalGate,
    CapabilityGraph,
    CapabilityNode,
    EvidenceState,
    IntentSpec,
    PlanGraph,
    PlanState,
    PlanStep,
    PlanStepIO,
    ResourceBudget,
    RetryPolicy,
    RiskLevel,
)
from molexp.agent.modes.plan.tasks_planning import (
    CandidateSet,
    PlanCandidate,
    SelectionResult,
    select_plan,
    synthesize_candidates,
)

from .conftest import ScriptedStructuredRouter


def _intent(*, complex_task: bool) -> IntentSpec:
    return IntentSpec(
        objective="run MD" if not complex_task else "build a full production MD pipeline",
        non_goals=(),
        required_outputs=("trajectory",),
        constraints=(),
        assumptions=(),
        missing_information=(),
        success_criteria=(),
        allowed_side_effects=(),
        budget=ResourceBudget(max_cost_usd=None, max_wall_seconds=None, model_tier=None),
        risk_level=RiskLevel.high if complex_task else RiskLevel.low,
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
    )


def _plan(plan_id: str) -> PlanGraph:
    return PlanGraph(
        plan_id=plan_id,
        intent_ref="i1",
        steps=(_step("s1"),),
        state=PlanState.draft_plan,
        compiled_contract_ref=None,
        notes="",
    )


# ── candidate synthesis ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_simple_task_yields_single_candidate() -> None:
    canned = CandidateSet(
        candidates=(PlanCandidate(label="A", plan_graph=_plan("p-a"), self_critique=""),),
        is_complex=False,
    )
    router = ScriptedStructuredRouter(responses=[canned])
    result = await synthesize_candidates(  # type: ignore[arg-type]
        router=router,
        intent=_intent(complex_task=False),
        capabilities=_capabilities(),
    )
    assert isinstance(result, CandidateSet)
    assert len(result.candidates) == 1


@pytest.mark.asyncio
async def test_complex_task_yields_three_candidates_with_critique() -> None:
    canned = CandidateSet(
        candidates=(
            PlanCandidate(label="A", plan_graph=_plan("p-a"), self_critique="conservative"),
            PlanCandidate(label="B", plan_graph=_plan("p-b"), self_critique="faster"),
            PlanCandidate(label="C", plan_graph=_plan("p-c"), self_critique="full production"),
        ),
        is_complex=True,
    )
    router = ScriptedStructuredRouter(responses=[canned])
    result = await synthesize_candidates(  # type: ignore[arg-type]
        router=router,
        intent=_intent(complex_task=True),
        capabilities=_capabilities(),
    )
    assert len(result.candidates) == 3
    assert {c.label for c in result.candidates} == {"A", "B", "C"}
    assert all(c.self_critique for c in result.candidates)


# ── selection ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_select_plan_picks_one_candidate() -> None:
    candidates = CandidateSet(
        candidates=(
            PlanCandidate(label="A", plan_graph=_plan("p-a"), self_critique="x"),
            PlanCandidate(label="B", plan_graph=_plan("p-b"), self_critique="y"),
        ),
        is_complex=True,
    )
    router = ScriptedStructuredRouter(
        responses=[SelectionResult(chosen_label="B", rationale="best validation")]
    )
    chosen = await select_plan(  # type: ignore[arg-type]
        router=router,
        candidates=candidates,
        capabilities=_capabilities(),
    )
    assert isinstance(chosen, PlanGraph)
    assert chosen.plan_id == "p-b"


@pytest.mark.asyncio
async def test_select_plan_single_candidate_skips_router() -> None:
    candidates = CandidateSet(
        candidates=(PlanCandidate(label="A", plan_graph=_plan("p-a"), self_critique=""),),
        is_complex=False,
    )
    router = ScriptedStructuredRouter(responses=[])  # router must not be called
    chosen = await select_plan(  # type: ignore[arg-type]
        router=router,
        candidates=candidates,
        capabilities=_capabilities(),
    )
    assert chosen.plan_id == "p-a"
    assert router.calls == []


@pytest.mark.asyncio
async def test_selected_plan_steps_bind_to_capability_nodes() -> None:
    candidates = CandidateSet(
        candidates=(PlanCandidate(label="A", plan_graph=_plan("p-a"), self_critique=""),),
        is_complex=False,
    )
    chosen = await select_plan(  # type: ignore[arg-type]
        router=ScriptedStructuredRouter(),
        candidates=candidates,
        capabilities=_capabilities(),
    )
    caps = {n.id for n in _capabilities().nodes}
    for step in chosen.steps:
        assert step.capability_id in caps
