"""Per-stage isolation tests for ``agent-mode-stage-pipeline-02`` ac-006.

Each of PlanMode's seven Stage subclasses is driven once with a stub
input and the typed terminal output + emitted-events are asserted —
no other stage runs. This is the testability win the substrate
migration buys: previously a single stage method couldn't be
exercised without setting up the whole ``_StageOutcome`` scratchpad.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from molexp.agent.harness.events import (
    ArtifactWrittenEvent,
    ClarificationRequiredEvent,
    PlanEmittedEvent,
    PreflightFailedEvent,
    RepairProposedEvent,
)
from molexp.agent.modes._planning import (
    ApprovalGate,
    CapabilityGraph,
    CapabilityNode,
    EvidenceState,
    IntentSpec,
    IsolatedTestSketch,
    MissingInfoItem,
    PlanGraph,
    PlanState,
    PlanStep,
    PlanStepIO,
    ResourceBudget,
    RetryPolicy,
    RiskLevel,
)
from molexp.agent.modes.plan import PlanMode, PlanModeConfig
from molexp.agent.modes.plan.plan_folder import PlanFolder
from molexp.agent.modes.plan.stages import (
    ClarifyIntent,
    EmitApprovedPlan,
    ExploreCapabilities,
    PlanThreadState,
    PreflightPlanGraph,
    SelectPlan,
    SynthesizeCandidates,
    SynthesizeIntent,
)
from molexp.agent.modes.plan.tasks_planning import (
    CandidateSet,
    PlanCandidate,
    SelectionResult,
)
from molexp.agent.review import ReviewDecision
from molexp.workspace import Workspace

from ..conftest import (
    ScriptedStructuredRouter,
    StubCapabilityProbe,
    make_harness,
    make_probe_result,
)

# ── builders (parallel to test_plan_mode.py) ────────────────────────────────


def _intent(*, missing: tuple[MissingInfoItem, ...] = ()) -> IntentSpec:
    return IntentSpec(
        objective="run an MD simulation",
        non_goals=(),
        required_outputs=("trajectory",),
        constraints=(),
        assumptions=(),
        missing_information=missing,
        success_criteria=(),
        allowed_side_effects=(),
        budget=ResourceBudget(max_cost_usd=None, max_wall_seconds=None, model_tier=None),
        risk_level=RiskLevel.low,
    )


def _step(step_id: str, *, capability_id: str | None) -> PlanStep:
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


def _plan(plan_id: str, *, capability_id: str | None) -> PlanGraph:
    return PlanGraph(
        plan_id=plan_id,
        intent_ref="i1",
        steps=(_step("s1", capability_id=capability_id),),
        state=PlanState.draft_plan,
        compiled_contract_ref=None,
        notes="",
    )


def _capabilities() -> CapabilityGraph:
    return CapabilityGraph(
        nodes=(
            CapabilityNode(
                id="build_system",
                capability="construct a molecular system",
                evidence_state=EvidenceState.evidenced,
                confidence=0.95,
                api_refs=("molpy.System",),
                usage_limits=(),
                needs_user_confirmation=False,
            ),
        ),
        edges=(),
    )


def _make_plan_mode(tmp_path: Path, *, max_iter: int = 2) -> PlanMode:
    ws = Workspace(tmp_path / "lab")
    pf = ws.add_folder(PlanFolder(name="plan-1"))
    return PlanMode(
        config=PlanModeConfig(max_repair_iterations=max_iter),
        plan_folder=pf,
        capability_probe=StubCapabilityProbe(make_probe_result()),
    )


def _stage(plan_mode: PlanMode, cls: type) -> object:
    """Locate the Stage instance of ``cls`` on the mode's pipeline."""
    for stage in plan_mode.pipeline.stages:
        if isinstance(stage, cls):
            return stage
    raise LookupError(f"no {cls.__name__} on the pipeline")


# ── stage 1: SynthesizeIntent ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_synthesize_intent_emits_intent_and_writes_artefact(tmp_path: Path) -> None:
    plan_mode = _make_plan_mode(tmp_path)
    router = ScriptedStructuredRouter(responses=[_intent()])
    harness, _ = make_harness(router)
    stage = cast(SynthesizeIntent, _stage(plan_mode, SynthesizeIntent))

    items = [item async for item in stage.run(harness=harness, input="simulate water")]

    assert any(isinstance(item, ArtifactWrittenEvent) for item in items)
    terminal = items[-1]
    assert isinstance(terminal, PlanThreadState)
    assert isinstance(terminal.intent, IntentSpec)
    assert terminal.user_input == "simulate water"


# ── stage 2: ClarifyIntent ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_clarify_intent_passes_through_when_no_blocking(tmp_path: Path) -> None:
    plan_mode = _make_plan_mode(tmp_path)
    plan_mode._transition(
        PlanState.exploring
    )  # ClarifyIntent transitions from intake → exploring; pre-set so the legal-move check passes
    plan_mode._transition(PlanState.draft_plan)  # not used here, just demonstrating
    # Actually reset:
    plan_mode = _make_plan_mode(tmp_path / "v2")
    # plan_folder starts at intake
    harness, _ = make_harness(ScriptedStructuredRouter())
    stage = cast(ClarifyIntent, _stage(plan_mode, ClarifyIntent))
    state = PlanThreadState(user_input="x", intent=_intent())

    items = [item async for item in stage.run(harness=harness, input=state)]

    assert not any(isinstance(item, ClarificationRequiredEvent) for item in items)
    terminal = items[-1]
    assert isinstance(terminal, PlanThreadState)
    assert plan_mode.plan_folder.plan_state is PlanState.exploring


@pytest.mark.asyncio
async def test_clarify_intent_yields_clarification_required_when_blocking(tmp_path: Path) -> None:
    plan_mode = _make_plan_mode(tmp_path)
    harness, _ = make_harness(ScriptedStructuredRouter())
    stage = cast(ClarifyIntent, _stage(plan_mode, ClarifyIntent))
    state = PlanThreadState(
        user_input="x",
        intent=_intent(missing=(MissingInfoItem(question="temp?", blocking=True),)),
    )

    items = [item async for item in stage.run(harness=harness, input=state)]

    events = [i for i in items if isinstance(i, ClarificationRequiredEvent)]
    assert len(events) == 1
    assert events[0].questions == "temp?"
    assert plan_mode.plan_folder.plan_state is PlanState.needs_clarification


# ── stage 3: ExploreCapabilities ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_explore_capabilities_emits_capability_graph(tmp_path: Path) -> None:
    plan_mode = _make_plan_mode(tmp_path)
    plan_mode._transition(PlanState.exploring)
    plan_mode._probe = StubCapabilityProbe(make_probe_result())
    harness, _ = make_harness(ScriptedStructuredRouter())
    stage = cast(ExploreCapabilities, _stage(plan_mode, ExploreCapabilities))
    state = PlanThreadState(user_input="x", intent=_intent())

    items = [item async for item in stage.run(harness=harness, input=state)]

    assert any(isinstance(item, ArtifactWrittenEvent) for item in items)
    terminal = items[-1]
    assert isinstance(terminal, PlanThreadState)
    assert isinstance(terminal.capabilities, CapabilityGraph)


# ── stage 4: SynthesizeCandidates ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_synthesize_candidates_returns_candidate_set(tmp_path: Path) -> None:
    plan_mode = _make_plan_mode(tmp_path)
    plan_mode._transition(PlanState.exploring)
    plan_mode._transition(PlanState.draft_plan)
    router = ScriptedStructuredRouter(
        responses=[
            CandidateSet(
                candidates=(
                    PlanCandidate(
                        label="A",
                        plan_graph=_plan("p-a", capability_id="build_system"),
                        self_critique="",
                    ),
                ),
                is_complex=False,
            )
        ]
    )
    harness, _ = make_harness(router)
    stage = cast(SynthesizeCandidates, _stage(plan_mode, SynthesizeCandidates))
    state = PlanThreadState(user_input="x", intent=_intent(), capabilities=_capabilities())

    items = [item async for item in stage.run(harness=harness, input=state)]

    terminal = items[-1]
    assert isinstance(terminal, PlanThreadState)
    assert terminal.candidates is not None
    assert len(terminal.candidates.candidates) == 1


# ── stage 5: SelectPlan ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_select_plan_yields_plan_emitted_event(tmp_path: Path) -> None:
    plan_mode = _make_plan_mode(tmp_path)
    plan_mode._transition(PlanState.exploring)
    plan_mode._transition(PlanState.draft_plan)
    router = ScriptedStructuredRouter(
        responses=[SelectionResult(chosen_label="A", rationale="only")]
    )
    harness, _ = make_harness(router)
    stage = cast(SelectPlan, _stage(plan_mode, SelectPlan))
    state = PlanThreadState(
        user_input="x",
        intent=_intent(),
        capabilities=_capabilities(),
        candidates=CandidateSet(
            candidates=(
                PlanCandidate(
                    label="A",
                    plan_graph=_plan("p-a", capability_id="build_system"),
                    self_critique="",
                ),
            ),
            is_complex=False,
        ),
    )

    items = [item async for item in stage.run(harness=harness, input=state)]

    assert any(isinstance(item, PlanEmittedEvent) for item in items)
    terminal = items[-1]
    assert isinstance(terminal, PlanThreadState)
    assert terminal.selected is not None


# ── stage 6: PreflightPlanGraph ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_preflight_plan_graph_passes_when_capability_evidenced(tmp_path: Path) -> None:
    plan_mode = _make_plan_mode(tmp_path)
    plan_mode._transition(PlanState.exploring)
    plan_mode._transition(PlanState.draft_plan)
    harness, _ = make_harness(ScriptedStructuredRouter())
    stage = cast(PreflightPlanGraph, _stage(plan_mode, PreflightPlanGraph))
    selected = _plan("p1", capability_id="build_system")
    state = PlanThreadState(
        user_input="x",
        intent=_intent(),
        capabilities=_capabilities(),
        selected=selected,
    )

    items = [item async for item in stage.run(harness=harness, input=state)]

    assert not any(isinstance(item, PreflightFailedEvent) for item in items)
    terminal = items[-1]
    assert isinstance(terminal, PlanThreadState)
    assert terminal.preflight is not None
    assert terminal.preflight.passed


@pytest.mark.asyncio
async def test_preflight_plan_graph_yields_preflight_failed_when_unevidenced(
    tmp_path: Path,
) -> None:
    plan_mode = _make_plan_mode(tmp_path)
    plan_mode._transition(PlanState.exploring)
    plan_mode._transition(PlanState.draft_plan)
    harness, _ = make_harness(ScriptedStructuredRouter())
    stage = cast(PreflightPlanGraph, _stage(plan_mode, PreflightPlanGraph))
    selected = _plan("p1", capability_id="cap_ghost")
    state = PlanThreadState(
        user_input="x",
        intent=_intent(),
        capabilities=_capabilities(),
        selected=selected,
    )

    items = [item async for item in stage.run(harness=harness, input=state)]

    assert any(isinstance(item, PreflightFailedEvent) for item in items)
    assert plan_mode.plan_folder.plan_state is PlanState.preflight_failed
    assert plan_mode._runtime.repair_signal is not None


# ── stage 7: EmitApprovedPlan ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_emit_approved_plan_builds_handoff_on_approve(tmp_path: Path) -> None:
    plan_mode = _make_plan_mode(tmp_path)
    plan_mode._transition(PlanState.exploring)
    plan_mode._transition(PlanState.draft_plan)
    plan_mode._transition(PlanState.awaiting_approval)
    harness, _ = make_harness(ScriptedStructuredRouter())
    stage = cast(EmitApprovedPlan, _stage(plan_mode, EmitApprovedPlan))
    state = PlanThreadState(
        user_input="x",
        intent=_intent(),
        capabilities=_capabilities(),
        selected=_plan("p1", capability_id="build_system"),
    )

    items = [item async for item in stage.run(harness=harness, input=state)]

    assert not any(isinstance(item, RepairProposedEvent) for item in items)
    assert plan_mode.plan_folder.plan_state is PlanState.approved
    assert plan_mode._handoff is not None
    assert plan_mode._handoff.plan_graph.state is PlanState.approved


@pytest.mark.asyncio
async def test_emit_approved_plan_plants_repair_on_reject(tmp_path: Path) -> None:
    plan_mode = _make_plan_mode(tmp_path)
    plan_mode._transition(PlanState.exploring)
    plan_mode._transition(PlanState.draft_plan)
    plan_mode._transition(PlanState.awaiting_approval)
    harness, _ = make_harness(ScriptedStructuredRouter())

    from molexp.agent.harness.hooks import HookPoint

    async def always_reject(ctx: object) -> ReviewDecision:
        return ReviewDecision(approved=False, reason="no")

    harness.hooks.register(HookPoint.before_approval, always_reject)
    stage = cast(EmitApprovedPlan, _stage(plan_mode, EmitApprovedPlan))
    state = PlanThreadState(
        user_input="x",
        intent=_intent(),
        capabilities=_capabilities(),
        selected=_plan("p1", capability_id="build_system"),
    )

    items = [item async for item in stage.run(harness=harness, input=state)]

    assert any(isinstance(item, RepairProposedEvent) for item in items)
    # Stage does NOT transition on reject — lifecycle validator handles
    # rewind transition; PlanMode post-pipeline handles exhausted-rejection.
    assert plan_mode.plan_folder.plan_state is PlanState.awaiting_approval
    assert plan_mode._runtime.repair_signal is not None
