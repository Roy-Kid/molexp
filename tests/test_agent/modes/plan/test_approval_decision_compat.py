"""ApprovalDecision schema extension + back-compat tests.

Locks the `target_node_ids / target_task_ids / cascade_downstream / feedback`
extension contract from the ``planmode-review-repair-loop`` spec
(acceptance criterion ``ac-001``). The four new fields MUST be optional
with defaults so legacy JSON payloads (no new fields) deserialize cleanly.

The companion :class:`PlanReviewView` extension (``previous_validation_failures``
and ``repair_iteration``) is exercised here as a related back-compat check
(criterion ``ac-002``); the runtime path through ``HumanReview.execute`` is
covered separately in ``test_repair_loop.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from molexp.agent.modes.plan.schemas import (
    ApprovalDecision,
    PlanBrief,
    PlanReviewView,
    ReportDigest,
    WorkflowContract,
)


def _digest() -> ReportDigest:
    return ReportDigest(summary="s", experimental_goal="g")


def _plan_brief() -> PlanBrief:
    return PlanBrief(overview="ov", chosen_method="m", stages=("a",), rationale="r")


def _contract() -> WorkflowContract:
    return WorkflowContract(workflow_id="workflow_test01", task_io=())


# ── ApprovalDecision new fields ────────────────────────────────────────────


def test_approval_decision_new_fields_default_to_empty_or_false() -> None:
    """The four new repair fields default to () / () / False / "" so existing
    constructors that only pass `approved` / `reason` / `override_validation`
    continue to work unchanged."""
    decision = ApprovalDecision(approved=False)
    assert decision.target_node_ids == ()
    assert decision.target_task_ids == ()
    assert decision.cascade_downstream is False
    assert decision.feedback == ""


def test_approval_decision_accepts_repair_targets() -> None:
    decision = ApprovalDecision(
        approved=False,
        reason="needs replan",
        target_node_ids=("DraftImplementationPlan",),
        target_task_ids=("prepare", "couple"),
        cascade_downstream=True,
        feedback="rework the equilibration step",
    )
    assert decision.approved is False
    assert decision.target_node_ids == ("DraftImplementationPlan",)
    assert decision.target_task_ids == ("prepare", "couple")
    assert decision.cascade_downstream is True
    assert decision.feedback == "rework the equilibration step"


def test_approval_decision_remains_frozen_with_new_fields() -> None:
    decision = ApprovalDecision(approved=True)
    with pytest.raises(ValidationError):
        decision.feedback = "mutated"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        decision.target_node_ids = ("X",)  # type: ignore[misc]


def test_approval_decision_back_compat_legacy_json() -> None:
    """A legacy serialized payload without the four new keys must round-trip
    cleanly into the new schema with defaults filled in."""
    legacy_payload = {"approved": True, "reason": "ok", "override_validation": False}
    decision = ApprovalDecision.model_validate(legacy_payload)
    assert decision.approved is True
    assert decision.reason == "ok"
    assert decision.override_validation is False
    # New fields are filled with defaults — back-compat invariant.
    assert decision.target_node_ids == ()
    assert decision.target_task_ids == ()
    assert decision.cascade_downstream is False
    assert decision.feedback == ""


def test_approval_decision_extra_forbid_still_active() -> None:
    """`extra="forbid"` is preserved through the extension — unknown keys
    still error so typos do not silently no-op."""
    with pytest.raises(ValidationError):
        ApprovalDecision.model_validate(
            {"approved": True, "no_such_field": "oops"},
        )


def test_approval_decision_target_ids_coerce_to_tuples() -> None:
    """Pydantic should coerce list inputs into the declared tuple type so
    that JSON arrays (the on-disk form) round-trip without manual conversion."""
    decision = ApprovalDecision.model_validate(
        {
            "approved": False,
            "target_node_ids": ["DraftImplementationPlan", "CompileWorkflowIR"],
            "target_task_ids": ["prepare"],
        },
    )
    assert decision.target_node_ids == ("DraftImplementationPlan", "CompileWorkflowIR")
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
