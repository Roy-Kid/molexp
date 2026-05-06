"""Phase 1a: plan-mode state machine transitions (spec §6.3)."""

from __future__ import annotations

from molexp.agent.orchestration import (
    PlanState,
    PlanStateMachine,
    render_reject_feedback,
)


def test_initial_state_is_chat() -> None:
    machine = PlanStateMachine()
    assert machine.state is PlanState.CHAT
    assert not machine.is_parked()


def test_request_emit_approve_cycle() -> None:
    preview: dict = {"workflow_ir": {"nodes": []}}
    machine = PlanStateMachine().request_plan()
    assert machine.state is PlanState.PLAN_REQUESTED

    machine = machine.emit_plan(request_id="req-1", plan_markdown="step 1\nstep 2", preview=preview)
    assert machine.state is PlanState.PLAN_EMITTED
    assert machine.is_parked()

    approved = machine.approve(edited_plan="step 1 (edited)")
    assert approved.state is PlanState.PLAN_APPROVED
    assert approved.edited_plan == "step 1 (edited)"


def test_reject_carries_feedback() -> None:
    preview: dict = {"workflow_ir": {"nodes": []}}
    machine = (
        PlanStateMachine()
        .request_plan()
        .emit_plan("req-1", "step 1", preview)
        .reject("too many steps")
    )
    assert machine.state is PlanState.PLAN_REJECTED
    assert machine.last_feedback == "too many steps"


def test_reject_feedback_template_per_decision_o2() -> None:
    rendered = render_reject_feedback("trim to one step")
    assert "Plan rejected." in rendered
    assert "trim to one step" in rendered
    assert rendered.endswith("Revise the plan and emit it again.")


def test_render_reject_feedback_handles_empty_input() -> None:
    rendered = render_reject_feedback("")
    assert "(no feedback)" in rendered
