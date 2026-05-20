"""``ReviewTarget`` ingestion across all three kinds (ac-003).

Covers detection of a typed ``PlanGraph`` target, a materialized
workspace path, and a mounted ``Run`` — each ingested into a
``ReviewTarget`` carrying the matching ``ReviewTargetKind``.
"""

from __future__ import annotations

from pathlib import Path

from molexp.agent.modes._planning import IntentSpec
from molexp.agent.modes.review.target import (
    ReviewTarget,
    ReviewTargetKind,
    detect_review_target,
)

from .conftest import make_intent, make_satisfying_plan

# ── ReviewTargetKind enum ────────────────────────────────────────────────


def test_review_target_kind_members() -> None:
    assert {k.value for k in ReviewTargetKind} == {"plan", "workspace", "run"}


# ── plan kind — inline PlanGraph ─────────────────────────────────────────


def test_detect_plan_target_from_inline_graph() -> None:
    plan = make_satisfying_plan()
    target = detect_review_target(
        user_input="review this plan",
        plan_graph=plan,
        workspace_path=None,
        run_ref=None,
    )
    assert isinstance(target, ReviewTarget)
    assert target.kind is ReviewTargetKind.plan
    assert target.plan_graph is plan
    assert target.workspace_path is None
    assert target.run_ref is None


def test_plan_target_carries_only_plan_graph() -> None:
    plan = make_satisfying_plan()
    target = ReviewTarget(kind=ReviewTargetKind.plan, plan_graph=plan)
    assert target.plan_graph is plan
    assert target.workspace_path is None
    assert target.run_ref is None


# ── workspace kind — materialized src tree ───────────────────────────────


def test_detect_workspace_target_from_path(tmp_path: Path) -> None:
    ws_path = tmp_path / "materialized"
    (ws_path / "src").mkdir(parents=True)
    (ws_path / "src" / "workflow.py").write_text("# generated\n")
    target = detect_review_target(
        user_input="review the materialized workspace",
        plan_graph=None,
        workspace_path=ws_path,
        run_ref=None,
    )
    assert target.kind is ReviewTargetKind.workspace
    assert target.workspace_path == ws_path
    assert target.plan_graph is None
    assert target.run_ref is None


# ── run kind — mounted Run reference ─────────────────────────────────────


def test_detect_run_target_from_run_ref() -> None:
    target = detect_review_target(
        user_input="review the completed run",
        plan_graph=None,
        workspace_path=None,
        run_ref="run-001",
    )
    assert target.kind is ReviewTargetKind.run
    assert target.run_ref == "run-001"
    assert target.plan_graph is None
    assert target.workspace_path is None


# ── precedence — plan graph wins when several are supplied ────────────────


def test_inline_plan_graph_takes_precedence(tmp_path: Path) -> None:
    plan = make_satisfying_plan()
    target = detect_review_target(
        user_input="review",
        plan_graph=plan,
        workspace_path=tmp_path,
        run_ref="run-001",
    )
    assert target.kind is ReviewTargetKind.plan


# ── frozen / typed contract ──────────────────────────────────────────────


def test_review_target_is_frozen() -> None:
    target = ReviewTarget(kind=ReviewTargetKind.run, run_ref="run-x")
    try:
        target.run_ref = "run-y"  # type: ignore[misc]
    except (AttributeError, TypeError, ValueError):
        return
    raise AssertionError("ReviewTarget must be frozen")


def test_intent_fixture_round_trips() -> None:
    intent: IntentSpec = make_intent()
    assert intent.objective.startswith("produce a charge-density")
    assert "report.pdf" in intent.required_outputs
