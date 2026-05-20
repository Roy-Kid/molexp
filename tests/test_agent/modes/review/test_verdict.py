"""``build_review_verdict`` outcome derivation + ``PlanDiff`` shape (ac-005).

A satisfying plan folds to ``overall == "pass"`` with no ``PlanDiff``;
a plan dropping a required output folds to ``overall == "needs_changes"``
with a well-formed ``PlanDiff`` (``failed_invariant`` set, non-empty
``affected_nodes`` / ``operations``).
"""

from __future__ import annotations

from molexp.agent.modes._planning import PlanDiff
from molexp.agent.modes.review.checks import (
    check_capability_evidence,
    check_intent_conformance,
    check_lifecycle_consistency,
)
from molexp.agent.modes.review.target import ReviewTargetKind
from molexp.agent.modes.review.verdict import (
    ReviewVerdict,
    StepFinding,
    build_review_verdict,
)

from .conftest import (
    make_capability_graph,
    make_dropped_output_plan,
    make_intent,
    make_satisfying_plan,
)


def _all_findings(intent: object, plan: object, caps: object) -> tuple[StepFinding, ...]:
    """Run all three checkers and concatenate the findings."""
    return (
        *check_intent_conformance(intent, plan),  # type: ignore[arg-type]
        *check_capability_evidence(plan, caps),  # type: ignore[arg-type]
        *check_lifecycle_consistency(plan),  # type: ignore[arg-type]
    )


# ── StepFinding contract ─────────────────────────────────────────────────


def test_step_finding_is_frozen() -> None:
    finding = StepFinding(
        step_id="s1",
        severity="error",
        summary="bad",
        detail="detail",
    )
    try:
        finding.severity = "info"  # type: ignore[misc]
    except (AttributeError, TypeError, ValueError):
        return
    raise AssertionError("StepFinding must be frozen")


# ── overall == pass ──────────────────────────────────────────────────────


def test_satisfying_plan_yields_pass() -> None:
    intent = make_intent()
    plan = make_satisfying_plan()
    caps = make_capability_graph(all_evidenced=True)
    findings = _all_findings(intent, plan, caps)
    verdict = build_review_verdict(
        findings=findings,
        intent=intent,
        plan=plan,
        target_kind=ReviewTargetKind.plan,
    )
    assert isinstance(verdict, ReviewVerdict)
    assert verdict.overall == "pass"
    assert verdict.plan_diff is None
    assert all(f.severity != "error" for f in verdict.findings)
    assert verdict.target_kind is ReviewTargetKind.plan
    assert verdict.intent_ref == plan.intent_ref


# ── overall == needs_changes + PlanDiff ──────────────────────────────────


def test_dropped_output_yields_needs_changes_with_plan_diff() -> None:
    intent = make_intent()
    plan = make_dropped_output_plan()  # missing report.pdf
    caps = make_capability_graph(all_evidenced=True)
    findings = _all_findings(intent, plan, caps)
    verdict = build_review_verdict(
        findings=findings,
        intent=intent,
        plan=plan,
        target_kind=ReviewTargetKind.plan,
    )
    assert verdict.overall == "needs_changes"
    assert verdict.plan_diff is not None
    diff: PlanDiff = verdict.plan_diff
    assert diff.failed_invariant
    assert diff.affected_nodes
    assert diff.operations
    assert diff.rationale


def test_needs_changes_plan_diff_round_trips() -> None:
    intent = make_intent()
    plan = make_dropped_output_plan()
    caps = make_capability_graph(all_evidenced=True)
    findings = _all_findings(intent, plan, caps)
    verdict = build_review_verdict(
        findings=findings,
        intent=intent,
        plan=plan,
        target_kind=ReviewTargetKind.plan,
    )
    payload = verdict.model_dump(mode="json")
    reloaded = ReviewVerdict.model_validate(payload)
    assert reloaded.overall == "needs_changes"
    assert reloaded.plan_diff is not None


# ── empty findings ───────────────────────────────────────────────────────


def test_no_findings_yields_pass() -> None:
    intent = make_intent()
    plan = make_satisfying_plan()
    verdict = build_review_verdict(
        findings=(),
        intent=intent,
        plan=plan,
        target_kind=ReviewTargetKind.plan,
    )
    assert verdict.overall == "pass"
    assert verdict.plan_diff is None
