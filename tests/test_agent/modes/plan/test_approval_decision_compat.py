"""ReviewDecision schema + PlanReviewView back-compat tests.

Locks the `target_steps / target_task_ids / cascade_downstream / feedback`
extension contract on :class:`~molexp.agent.review.ReviewDecision` so
the on-disk repair-history payloads stay deserialisable across the
rename from ``ApprovalDecision``.  Optional defaults mean a minimal
``ReviewDecision(approved=...)`` constructor still works.

The companion :class:`PlanReviewView` extension (``previous_validation_failures``
and ``repair_iteration``) is exercised here as a related back-compat
check; the runtime path through ``HumanReview._execute`` is covered
separately in ``test_repair_loop.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from molexp.agent.modes.plan.schemas import (
    PlanBrief,
    PlanReviewView,
    ReportDigest,
    ReviewDecision,
    WorkflowContract,
)


def _digest() -> ReportDigest:
    return ReportDigest(summary="s", experimental_goal="g")


def _plan_brief() -> PlanBrief:
    return PlanBrief(overview="ov", chosen_method="m", stages=("a",), rationale="r")


def _contract() -> WorkflowContract:
    return WorkflowContract(workflow_id="workflow_test01", task_io=())


# ── ReviewDecision repair fields ───────────────────────────────────────────


def test_review_decision_new_fields_default_to_empty_or_false() -> None:
    """The repair fields default to () / () / False / "" so existing
    constructors that only pass `approved` / `reason` / `override_validation`
    continue to work unchanged."""
    decision = ReviewDecision(approved=False)
    assert decision.target_steps == ()
    assert decision.target_task_ids == ()
    assert decision.cascade_downstream is False
    assert decision.feedback == ""


def test_review_decision_accepts_repair_targets() -> None:
    decision = ReviewDecision(
        approved=False,
        reason="needs replan",
        target_steps=("DraftImplementationPlan",),
        target_task_ids=("prepare", "couple"),
        cascade_downstream=True,
        feedback="rework the equilibration step",
    )
    assert decision.approved is False
    assert decision.target_steps == ("DraftImplementationPlan",)
    assert decision.target_task_ids == ("prepare", "couple")
    assert decision.cascade_downstream is True
    assert decision.feedback == "rework the equilibration step"


def test_review_decision_remains_frozen_with_new_fields() -> None:
    decision = ReviewDecision(approved=True)
    with pytest.raises(ValidationError):
        decision.feedback = "mutated"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        decision.target_steps = ("X",)  # type: ignore[misc]


def test_review_decision_minimal_payload() -> None:
    """A minimal payload (no repair fields) round-trips into the new schema
    with defaults filled in."""
    minimal_payload = {"approved": True, "reason": "ok", "override_validation": False}
    decision = ReviewDecision.model_validate(minimal_payload)
    assert decision.approved is True
    assert decision.reason == "ok"
    assert decision.override_validation is False
    # Repair fields fall back to defaults.
    assert decision.target_steps == ()
    assert decision.target_task_ids == ()
    assert decision.cascade_downstream is False
    assert decision.feedback == ""


def test_review_decision_extra_forbid_still_active() -> None:
    """`extra="forbid"` is preserved — unknown keys still error so typos
    do not silently no-op."""
    with pytest.raises(ValidationError):
        ReviewDecision.model_validate(
            {"approved": True, "no_such_field": "oops"},
        )


def test_review_decision_target_ids_coerce_to_tuples() -> None:
    """Pydantic should coerce list inputs into the declared tuple type so
    that JSON arrays (the on-disk form) round-trip without manual conversion."""
    decision = ReviewDecision.model_validate(
        {
            "approved": False,
            "target_steps": ["DraftImplementationPlan", "CompileWorkflowIR"],
            "target_task_ids": ["prepare"],
        },
    )
    assert decision.target_steps == ("DraftImplementationPlan", "CompileWorkflowIR")
    assert decision.target_task_ids == ("prepare",)


# ── PlanReviewView new fields ──────────────────────────────────────────────


def test_plan_review_view_new_fields_default_to_zero_and_empty() -> None:
    """`previous_validation_failures` / `repair_iteration` default to () / 0
    so first-iteration construction remains a positional / minimal kwargs
    call (pre-existing tests in test_pipeline_core.py do not need updates)."""
    view = PlanReviewView(
        plan_id="p",
        experiment_workspace_path=Path("/tmp/ws"),
        digest=_digest(),
        plan_brief=_plan_brief(),
        contract=_contract(),
        validation_passed=True,
        validation_summary="",
    )
    assert view.previous_validation_failures == ()
    assert view.repair_iteration == 0


def test_plan_review_view_carries_iteration_state() -> None:
    """When the repair loop is on iteration N, the review view surfaces the
    failures that triggered the rejection plus the iteration counter."""
    view = PlanReviewView(
        plan_id="p",
        experiment_workspace_path=Path("/tmp/ws"),
        digest=_digest(),
        plan_brief=_plan_brief(),
        contract=_contract(),
        validation_passed=False,
        validation_summary="2 checks failed",
        previous_validation_failures=(
            "task_implementation_module_present:couple",
            "task_test_module_present:isolate",
        ),
        repair_iteration=2,
    )
    assert view.repair_iteration == 2
    assert view.previous_validation_failures == (
        "task_implementation_module_present:couple",
        "task_test_module_present:isolate",
    )
