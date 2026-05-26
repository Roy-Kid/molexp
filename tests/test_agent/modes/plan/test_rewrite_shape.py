"""Structural shape tests for the PlanMode pydantic-ai-native rewrite.

These tests pin the *static* contract the rewrite must satisfy:

- Pipeline collapsed to 5 stages with specific names + order
- ``PlanStep`` carries ``api_refs`` + ``composition_notes`` (drops
  ``capability_id`` + ``tool_binding``)
- ``ApprovedPlanHandoff`` no longer carries a ``capability_graph`` field
- Preflight report names exactly the 5 new checks
- ``PlanMode.__init__`` drops the probe surface (``capability_probe`` /
  ``probe_model`` kwargs)
- Capability-graph era types are removed from ``modes/_planning``
- ``research_planner.py`` exists and exposes the build function

All start RED and turn GREEN as tasks 2-10 of the spec land.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import pytest

# ── ac-001 — 5-stage pipeline shape ────────────────────────────────────────


def _make_planmode_for_shape():
    """Construct a bare PlanMode against an empty PlanFolder to read its pipeline."""
    from molexp.agent.modes.plan import PlanMode
    from molexp.agent.modes.plan.plan_folder import PlanFolder
    from molexp.workspace import Workspace

    workspace = Workspace(Path("/tmp") / "_shape_check_ws")
    plan_folder = PlanFolder(name="shape-check")
    workspace.add_folder(plan_folder)
    return PlanMode(plan_folder=plan_folder, workspace=workspace.root)


def test_pipeline_has_exactly_five_stages_in_order() -> None:
    mode = _make_planmode_for_shape()
    names = tuple(stage.name for stage in mode.pipeline.stages)
    assert names == (
        "SynthesizeIntent",
        "ClarifyIntent",
        "ResearchAndPlan",
        "PreflightPlanGraph",
        "EmitApprovedPlan",
    )


def test_pipeline_carries_two_repair_policies_both_rewinding_to_research_and_plan() -> None:
    mode = _make_planmode_for_shape()
    repairs = mode.pipeline.repairs
    triggers = {policy.trigger_event_kind for policy in repairs}
    assert {"preflight_failed", "repair_proposed"}.issubset(triggers)
    # The two real repair policies both rewind to ResearchAndPlan. The
    # ``clarification_required`` policy that may also be present is a
    # structural event-router (max_iterations=0 → straight to terminal),
    # not a true repair, so its rewind target can be the terminal name.
    real_repairs = tuple(
        policy
        for policy in repairs
        if policy.trigger_event_kind in {"preflight_failed", "repair_proposed"}
    )
    for policy in real_repairs:
        assert policy.rewind_to == "ResearchAndPlan"


# ── ac-003 — PlanStep schema ───────────────────────────────────────────────


def test_planstep_has_api_refs_and_composition_notes() -> None:
    from molexp.agent.modes._planning import PlanStep

    fields = PlanStep.model_fields
    assert "api_refs" in fields, "PlanStep must carry api_refs"
    assert "composition_notes" in fields, "PlanStep must carry composition_notes"


def test_planstep_drops_capability_id_and_tool_binding() -> None:
    from molexp.agent.modes._planning import PlanStep

    fields = PlanStep.model_fields
    assert "capability_id" not in fields, "capability_id should be removed"
    assert "tool_binding" not in fields, "tool_binding should be removed"


# ── ac-004 — ApprovedPlanHandoff shape ─────────────────────────────────────


def test_handoff_does_not_carry_capability_graph() -> None:
    from molexp.agent.modes.plan.handoff import ApprovedPlanHandoff

    fields = ApprovedPlanHandoff.model_fields
    assert "capability_graph" not in fields, (
        "ApprovedPlanHandoff must not reference CapabilityGraph"
    )
    assert set(fields) == {
        "plan_id",
        "intent",
        "plan_graph",
        "plan_folder_path",
        "direction_approved_at",
    }


# ── ac-002 — CapabilityGraph & friends removed ─────────────────────────────


def test_capability_graph_is_no_longer_exported() -> None:
    planning = importlib.import_module("molexp.agent.modes._planning")
    for name in (
        "CapabilityGraph",
        "CapabilityNode",
        "CapabilityEdge",
        "EvidenceState",
        "CapabilityEdgeKind",
    ):
        assert not hasattr(planning, name), f"{name} must be removed from _planning"


def test_obsolete_files_deleted() -> None:
    src = Path(__file__).resolve().parents[4] / "src" / "molexp" / "agent"
    for relpath in (
        "_pydanticai/capability_probe.py",
        "_pydanticai/capability_probe_factory.py",
        "modes/plan/protocols.py",
        "modes/plan/capability_probe_null.py",
        "modes/plan/capability_evidence.py",
        "modes/plan/capability_projection.py",
        "modes/_planning/capability_graph.py",
        "modes/plan/stages/explore_capabilities.py",
        "modes/plan/stages/synthesize_candidates.py",
        "modes/plan/stages/select_plan.py",
    ):
        assert not (src / relpath).exists(), f"{relpath} should be deleted"


# ── ac-005 — research_planner.py exists ────────────────────────────────────


def test_research_planner_module_exists_with_build_function() -> None:
    mod = importlib.import_module("molexp.agent._pydanticai.research_planner")
    assert hasattr(mod, "build_research_planner"), (
        "research_planner must export build_research_planner(...)"
    )


def test_research_planner_has_no_outer_iteration() -> None:
    """The research-and-plan agent must be one pydantic-ai Agent call.

    No surrounding Python ``for`` / ``while`` loops over needs /
    candidates / refinements. pydantic-ai drives the tool-call loop.
    """
    src = (
        Path(__file__).resolve().parents[4]
        / "src"
        / "molexp"
        / "agent"
        / "_pydanticai"
        / "research_planner.py"
    )
    text = src.read_text()
    import re

    for_match = re.search(r"^\s*for .+ in .+:", text, re.MULTILINE)
    while_match = re.search(r"^\s*while .+:", text, re.MULTILINE)
    assert for_match is None, f"unexpected `for` loop in research_planner.py: {for_match.group(0)}"
    assert while_match is None, (
        f"unexpected `while` loop in research_planner.py: {while_match.group(0)}"
    )


# ── ac-007 — Preflight has 5 named checks ──────────────────────────────────


def test_preflight_yields_exactly_five_named_checks() -> None:
    from molexp.agent.modes._planning import PlanGraph, PlanState
    from molexp.agent.modes.plan.plan_graph_preflight import preflight_plan_graph

    intent_required: tuple[str, ...] = ()
    empty_plan = PlanGraph(
        plan_id="p",
        intent_ref=None,
        steps=(),
        state=PlanState.draft_plan,
        compiled_contract_ref=None,
        notes="",
    )
    report = preflight_plan_graph(plan_graph=empty_plan, required_outputs=intent_required)
    check_names = {check.name for check in report.checks}
    assert check_names == {
        "graph_closed",
        "graph_acyclic",
        "outputs_consumed",
        "every_step_has_api_refs",
        "every_step_isolated_testable",
    }


# ── ac-008 — PlanMode constructor drops probe surface ──────────────────────


def test_planmode_constructor_drops_probe_kwargs() -> None:
    from molexp.agent.modes.plan import PlanMode

    params = inspect.signature(PlanMode.__init__).parameters
    assert "capability_probe" not in params, "drop capability_probe= kwarg"
    assert "probe_model" not in params, "drop probe_model= kwarg"


# ── ac-009 — AuthorMode has no capability_graph references ────────────────


def test_author_mode_no_capability_graph_references() -> None:
    src = Path(__file__).resolve().parents[4] / "src" / "molexp" / "agent" / "modes" / "author"
    offenders: list[str] = []
    for path in src.rglob("*.py"):
        text = path.read_text()
        if "capability_graph" in text or "CapabilityGraph" in text:
            offenders.append(str(path.relative_to(src.parent.parent)))
    assert offenders == [], (
        f"AuthorMode must not reference CapabilityGraph anymore; offenders: {offenders}"
    )


# ── ac-006 — schema_parse retry stripped ───────────────────────────────────


def test_should_retry_treats_schema_parse_as_terminal() -> None:
    from molexp.agent._pydanticai.errors import ErrorKind
    from molexp.agent._pydanticai.retry import RetryPolicy, should_retry

    policy = RetryPolicy()
    assert should_retry(ErrorKind.schema_parse, policy, attempt=1) is False
    assert should_retry(ErrorKind.schema_parse, policy, attempt=2) is False


# ── Marker for pytest discovery ────────────────────────────────────────────


@pytest.mark.unit
def test_module_loads() -> None:
    """Trivial canary so pytest discovers the file even if every test errors."""
    assert True
