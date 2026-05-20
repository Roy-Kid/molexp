"""The three ReviewMode checker pure functions (ac-004 / ac-006).

``check_intent_conformance`` — IntentSpec criteria / required outputs /
non-goals / allowed side effects. ``check_capability_evidence`` — each
``PlanStep.capability_id`` still maps to an evidenced ``CapabilityNode``.
``check_lifecycle_consistency`` — every step's ``PlanCheck`` is met and
the ``PlanState`` is consistent with what actually happened.
"""

from __future__ import annotations

from molexp.agent.modes.review.checks import (
    check_capability_evidence,
    check_intent_conformance,
    check_lifecycle_consistency,
)
from molexp.agent.modes.review.verdict import StepFinding

from .conftest import (
    make_capability_graph,
    make_dropped_output_plan,
    make_intent,
    make_lost_evidence_plan,
    make_satisfying_plan,
    make_unmet_check_plan,
)

# ── check_intent_conformance ─────────────────────────────────────────────


def test_intent_conformance_passes_for_satisfying_plan() -> None:
    intent = make_intent()
    plan = make_satisfying_plan()
    findings = check_intent_conformance(intent, plan)
    errors = [f for f in findings if f.severity == "error"]
    assert errors == []


def test_intent_conformance_flags_dropped_required_output() -> None:
    intent = make_intent()
    plan = make_dropped_output_plan()  # missing report.pdf
    findings = check_intent_conformance(intent, plan)
    errors = [f for f in findings if f.severity == "error"]
    assert errors, "a dropped required output must produce an error finding"
    assert all(isinstance(f, StepFinding) for f in errors)
    # The finding names the violated invariant.
    assert any("report.pdf" in f.detail or "report.pdf" in f.summary for f in errors)
    assert any(f.failed_invariant for f in errors)


def test_intent_conformance_returns_typed_findings() -> None:
    intent = make_intent()
    plan = make_satisfying_plan()
    findings = check_intent_conformance(intent, plan)
    assert isinstance(findings, tuple)
    for f in findings:
        assert isinstance(f, StepFinding)
        assert f.severity in {"info", "warning", "error"}


# ── check_capability_evidence ────────────────────────────────────────────


def test_capability_evidence_passes_when_all_evidenced() -> None:
    plan = make_satisfying_plan()
    caps = make_capability_graph(all_evidenced=True)
    findings = check_capability_evidence(plan, caps)
    errors = [f for f in findings if f.severity == "error"]
    assert errors == []


def test_capability_evidence_flags_lost_evidence() -> None:
    plan = make_lost_evidence_plan()
    caps = make_capability_graph(all_evidenced=False)  # cap_render is missing
    findings = check_capability_evidence(plan, caps)
    errors = [f for f in findings if f.severity == "error"]
    assert errors, "a lost-evidence capability must produce an error finding"
    # The error is per-step — it names the render step.
    assert any(f.step_id == "render" for f in errors)


def test_capability_evidence_flags_unknown_capability_id() -> None:
    """A capability_id that no longer maps to *any* node is an error."""
    plan = make_lost_evidence_plan()
    # A capability graph that knows nothing about cap_render at all.
    caps = make_capability_graph(all_evidenced=True)
    caps = caps.model_copy(update={"nodes": (caps.nodes[0],)})  # drop cap_render
    findings = check_capability_evidence(plan, caps)
    errors = [f for f in findings if f.severity == "error"]
    assert any(f.step_id == "render" for f in errors)


# ── check_lifecycle_consistency ──────────────────────────────────────────


def test_lifecycle_consistency_passes_for_satisfying_plan() -> None:
    plan = make_satisfying_plan()
    findings = check_lifecycle_consistency(plan)
    errors = [f for f in findings if f.severity == "error"]
    assert errors == []


def test_lifecycle_consistency_flags_ready_for_run_with_unmet_check() -> None:
    plan = make_unmet_check_plan()  # ready_for_run but compute has no checks
    findings = check_lifecycle_consistency(plan)
    assert findings, "a ready_for_run plan missing a verifiable check is inconsistent"
    assert any(f.severity in {"warning", "error"} for f in findings)
