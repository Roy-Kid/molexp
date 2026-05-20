"""``PlanFolder`` + ``PlanState`` lifecycle tests (ac-006).

``plan_folder.py`` uses 01's ``PlanState`` enum — no ``PlanStatus``
``Literal``; illegal transitions raise ``IllegalPlanTransitionError``;
typed-plan artefact writers persist the plan substrate without writing
``src/`` / ``tests/`` / ``ir/``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from molexp.agent.modes._planning import (
    CapabilityGraph,
    IllegalPlanTransitionError,
    IntentSpec,
    PlanGraph,
    PlanState,
    ResourceBudget,
    RiskLevel,
)
from molexp.agent.modes.plan.plan_folder import AGENT_PLAN_KIND, PlanFolder
from molexp.agent.modes.plan.plan_graph_preflight import run_plan_graph_preflight
from molexp.workspace import Workspace


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


def _plan(state: PlanState = PlanState.draft_plan) -> PlanGraph:
    return PlanGraph(
        plan_id="p1",
        intent_ref="i1",
        steps=(),
        state=state,
        compiled_contract_ref=None,
        notes="",
    )


def test_plan_folder_kind() -> None:
    assert AGENT_PLAN_KIND == "agent.plan"
    pf = PlanFolder(name="my-plan")
    assert pf.metadata.kind == AGENT_PLAN_KIND


def test_plan_folder_metadata_uses_plan_state() -> None:
    pf = PlanFolder(name="my-plan")
    assert pf.plan_state is PlanState.intake


def test_plan_folder_no_plan_status_literal() -> None:
    import molexp.agent.modes.plan.plan_folder as mod

    assert not hasattr(mod, "PlanStatus")


def test_plan_folder_legal_transition() -> None:
    ws = Workspace(Path("/tmp/lab-test-lifecycle"))  # not materialized; in-memory checks below
    del ws
    pf = PlanFolder(name="my-plan")
    pf.transition_to(PlanState.exploring)
    assert pf.plan_state is PlanState.exploring


def test_plan_folder_illegal_transition_raises() -> None:
    pf = PlanFolder(name="my-plan")
    with pytest.raises(IllegalPlanTransitionError):
        pf.transition_to(PlanState.approved)  # intake -> approved is illegal


def test_plan_folder_writers_persist_typed_artefacts(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "lab")
    pf = ws.add_folder(PlanFolder(name="my-plan"))
    intent_path = pf.write_intent(_intent())
    cap_path = pf.write_capability_graph(CapabilityGraph(nodes=(), edges=()))
    candidate_path = pf.write_candidate("A", _plan())
    selected_path = pf.write_selected_plan(_plan(PlanState.approved))
    preflight_path = pf.write_preflight_report(
        run_plan_graph_preflight(
            plan_graph=_plan(),
            intent=_intent(),
            capabilities=CapabilityGraph(nodes=(), edges=()),
        )
    )
    for path in (intent_path, cap_path, candidate_path, selected_path, preflight_path):
        assert Path(path).exists()


def test_plan_folder_writes_no_codegen_dirs(tmp_path: Path) -> None:
    """PlanMode is read-only: no src/ tests/ ir/ writers on the folder."""
    ws = Workspace(tmp_path / "lab")
    pf = ws.add_folder(PlanFolder(name="my-plan"))
    pf.write_intent(_intent())
    pf.write_selected_plan(_plan())
    root = Path(pf.path())
    for forbidden in ("src", "tests", "ir"):
        assert not (root / forbidden).exists()
    # The retired generated-source writers are gone.
    for retired in ("write_test_module", "write_task_implementation", "src_dir", "tests_dir"):
        assert not hasattr(pf, retired)


def test_plan_folder_round_trips_from_disk(tmp_path: Path) -> None:
    ws = Workspace(tmp_path / "lab")
    pf = ws.add_folder(PlanFolder(name="my-plan"))
    pf.transition_to(PlanState.exploring)
    pf.save()
    ws2 = Workspace(tmp_path / "lab")
    reloaded = ws2.get_folder("my-plan", cls=PlanFolder)
    assert reloaded.plan_state is PlanState.exploring
