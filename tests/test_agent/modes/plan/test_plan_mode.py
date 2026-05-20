"""End-to-end ``PlanMode`` tests (ac-007).

A full ``PlanMode.run`` on a scripted router + stub probe ends at
``PlanState.approved`` with an ``ApprovedPlanHandoff``; the ``PlanFolder``
holds no ``src/`` / ``tests/`` / ``ir/`` directories and no executable
code. Edge cases: blocking missing-info routes to
``needs_clarification``; a rejected ``approve_direction`` gate produces a
``PlanDiff``; a preflight failure routes to ``preflight_failed``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from molexp.agent.harness.events import (
    ApprovalRequestedEvent,
    ModeCompletedEvent,
    PlanEmittedEvent,
    PreflightFailedEvent,
    RepairProposedEvent,
)
from molexp.agent.harness.hooks import HookPoint
from molexp.agent.modes._planning import (
    ApprovalGate,
    IntentSpec,
    MissingInfoItem,
    PlanGraph,
    PlanState,
    PlanStep,
    PlanStepIO,
    ResourceBudget,
    RetryPolicy,
    RiskLevel,
)
from molexp.agent.modes.plan import ApprovedPlanHandoff, PlanMode, PlanModeConfig
from molexp.agent.modes.plan.plan_folder import PlanFolder
from molexp.agent.modes.plan.tasks_planning import (
    CandidateSet,
    PlanCandidate,
    SelectionResult,
)
from molexp.agent.review import ReviewDecision
from molexp.workspace import Workspace

from .conftest import ScriptedStructuredRouter, make_harness

# ── builders ───────────────────────────────────────────────────────────────


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


def _happy_router(*, capability_id: str = "build_system") -> ScriptedStructuredRouter:
    """A router scripted for a clean single-candidate happy path.

    ``build_system`` is evidenced by the stub probe with no external
    resource marker, so the plan passes the structural preflight.
    """
    return ScriptedStructuredRouter(
        responses=[
            _intent(),
            CandidateSet(
                candidates=(
                    PlanCandidate(
                        label="A",
                        plan_graph=_plan("p-a", capability_id=capability_id),
                        self_critique="",
                    ),
                ),
                is_complex=False,
            ),
            SelectionResult(chosen_label="A", rationale="only candidate"),
        ]
    )


def _plan_folder(tmp_path: Path) -> PlanFolder:
    ws = Workspace(tmp_path / "lab")
    return ws.add_folder(PlanFolder(name="plan-1"))


# ── happy path ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_plan_mode_happy_path_emits_approved_handoff(
    tmp_path: Path, stub_probe: object
) -> None:
    pf = _plan_folder(tmp_path)
    mode = PlanMode(
        config=PlanModeConfig(),
        plan_folder=pf,
        capability_probe=stub_probe,  # type: ignore[arg-type]
    )
    harness, _ = make_harness(_happy_router())
    events = [ev async for ev in mode.run(harness=harness, user_input="simulate water")]

    completed = events[-1]
    assert isinstance(completed, ModeCompletedEvent)
    assert completed.result is not None
    handoff = completed.result["mode_state"]["handoff"]
    assert handoff["plan_id"] == pf.plan_id

    # The typed handoff round-trips.
    rebuilt = ApprovedPlanHandoff.model_validate(handoff)
    assert rebuilt.plan_graph.state is PlanState.approved
    assert pf.plan_state is PlanState.approved


@pytest.mark.asyncio
async def test_plan_mode_emits_plan_and_approval_events(tmp_path: Path, stub_probe: object) -> None:
    pf = _plan_folder(tmp_path)
    mode = PlanMode(config=PlanModeConfig(), plan_folder=pf, capability_probe=stub_probe)  # type: ignore[arg-type]
    harness, sink = make_harness(_happy_router())
    events = [ev async for ev in mode.run(harness=harness, user_input="x")]
    all_events = list(events) + list(sink)
    assert any(isinstance(e, PlanEmittedEvent) for e in all_events)
    assert any(isinstance(e, ApprovalRequestedEvent) for e in all_events)


@pytest.mark.asyncio
async def test_plan_mode_writes_no_codegen_dirs(tmp_path: Path, stub_probe: object) -> None:
    pf = _plan_folder(tmp_path)
    mode = PlanMode(config=PlanModeConfig(), plan_folder=pf, capability_probe=stub_probe)  # type: ignore[arg-type]
    harness, _ = make_harness(_happy_router())
    async for _ in mode.run(harness=harness, user_input="x"):
        pass
    root = Path(pf.path())
    for forbidden in ("src", "tests", "ir"):
        assert not (root / forbidden).exists()
    # No .py files anywhere under the plan folder.
    assert not list(root.rglob("*.py"))


@pytest.mark.asyncio
async def test_plan_mode_persists_typed_artefacts(tmp_path: Path, stub_probe: object) -> None:
    pf = _plan_folder(tmp_path)
    mode = PlanMode(config=PlanModeConfig(), plan_folder=pf, capability_probe=stub_probe)  # type: ignore[arg-type]
    harness, _ = make_harness(_happy_router())
    async for _ in mode.run(harness=harness, user_input="x"):
        pass
    root = Path(pf.path())
    # intent, capability-graph, selected plan, preflight report all on disk.
    assert list(root.rglob("intent.json"))
    assert list(root.rglob("capability_graph.json"))
    assert list(root.rglob("selected_plan.json"))


# ── blocking missing-info ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_blocking_missing_info_routes_to_needs_clarification(
    tmp_path: Path, stub_probe: object
) -> None:
    pf = _plan_folder(tmp_path)
    mode = PlanMode(config=PlanModeConfig(), plan_folder=pf, capability_probe=stub_probe)  # type: ignore[arg-type]
    router = ScriptedStructuredRouter(
        responses=[_intent(missing=(MissingInfoItem(question="temp?", blocking=True),))]
    )
    harness, _ = make_harness(router)
    events = [ev async for ev in mode.run(harness=harness, user_input="x")]
    completed = events[-1]
    assert isinstance(completed, ModeCompletedEvent)
    assert pf.plan_state is PlanState.needs_clarification
    # Stage stops before candidate synthesis — probe never ran.
    assert stub_probe.calls == []  # type: ignore[attr-defined]


# ── rejected direction → repair ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rejected_direction_produces_repair_diff(tmp_path: Path, stub_probe: object) -> None:
    pf = _plan_folder(tmp_path)
    mode = PlanMode(
        config=PlanModeConfig(max_repair_iterations=1),
        plan_folder=pf,
        capability_probe=stub_probe,  # type: ignore[arg-type]
    )
    # The happy router needs enough responses for the re-run after repair:
    # intent / candidates / select, then candidates / select again.
    router = ScriptedStructuredRouter(
        responses=[
            _intent(),
            CandidateSet(
                candidates=(
                    PlanCandidate(
                        label="A",
                        plan_graph=_plan("p-a", capability_id="build_system"),
                        self_critique="",
                    ),
                ),
                is_complex=False,
            ),
            SelectionResult(chosen_label="A", rationale="r"),
            CandidateSet(
                candidates=(
                    PlanCandidate(
                        label="A",
                        plan_graph=_plan("p-a2", capability_id="build_system"),
                        self_critique="",
                    ),
                ),
                is_complex=False,
            ),
            SelectionResult(chosen_label="A", rationale="r2"),
        ]
    )
    harness, sink = make_harness(router)

    # A before_approval hook rejects the first gate, approves the rest.
    rejections = {"count": 0}

    async def reject_once(ctx: object) -> ReviewDecision:
        rejections["count"] += 1
        if rejections["count"] == 1:
            return ReviewDecision(approved=False, reason="wrong direction")
        return ReviewDecision(approved=True, reason="ok")

    harness.hooks.register(HookPoint.before_approval, reject_once)

    events = [ev async for ev in mode.run(harness=harness, user_input="x")]
    all_events = list(events) + list(sink)
    assert any(isinstance(e, RepairProposedEvent) for e in all_events)


@pytest.mark.asyncio
async def test_rejected_direction_exhausts_budget_ends_rejected(
    tmp_path: Path, stub_probe: object
) -> None:
    pf = _plan_folder(tmp_path)
    mode = PlanMode(
        config=PlanModeConfig(max_repair_iterations=0),
        plan_folder=pf,
        capability_probe=stub_probe,  # type: ignore[arg-type]
    )
    harness, _ = make_harness(_happy_router())

    async def always_reject(ctx: object) -> ReviewDecision:
        return ReviewDecision(approved=False, reason="no")

    harness.hooks.register(HookPoint.before_approval, always_reject)
    events = [ev async for ev in mode.run(harness=harness, user_input="x")]
    assert isinstance(events[-1], ModeCompletedEvent)
    assert pf.plan_state is PlanState.rejected


# ── preflight failure ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_preflight_failure_routes_to_preflight_failed(
    tmp_path: Path, stub_probe: object
) -> None:
    pf = _plan_folder(tmp_path)
    mode = PlanMode(
        config=PlanModeConfig(max_repair_iterations=0),
        plan_folder=pf,
        capability_probe=stub_probe,  # type: ignore[arg-type]
    )
    # capability_id="cap_ghost" is not evidenced by the stub probe → preflight fails.
    router = ScriptedStructuredRouter(
        responses=[
            _intent(),
            CandidateSet(
                candidates=(
                    PlanCandidate(
                        label="A",
                        plan_graph=_plan("p-a", capability_id="cap_ghost"),
                        self_critique="",
                    ),
                ),
                is_complex=False,
            ),
            SelectionResult(chosen_label="A", rationale="r"),
        ]
    )
    harness, sink = make_harness(router)
    events = [ev async for ev in mode.run(harness=harness, user_input="x")]
    all_events = list(events) + list(sink)
    assert any(isinstance(e, PreflightFailedEvent) for e in all_events)
    assert pf.plan_state is PlanState.preflight_failed
