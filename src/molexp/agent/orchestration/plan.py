"""Plan-mode state machine.

States::

    CHAT -> PLAN_REQUESTED -> PLAN_EMITTED -> USER_DECISION
      approved: PLAN_APPROVED -> CHAT continuation
      rejected: PLAN_REJECTED -> PLAN_REQUESTED with feedback

The state machine is pure; the runner drives transitions and the
session inspects ``state`` to decide whether to park, replay, or
continue.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict

from molexp.workflow import PlanProposal, WorkflowPreviewView


class PlanState(str, Enum):
    CHAT = "chat"
    PLAN_REQUESTED = "plan_requested"
    PLAN_EMITTED = "plan_emitted"
    PLAN_APPROVED = "plan_approved"
    PLAN_REJECTED = "plan_rejected"


class PlanStateMachine(BaseModel):
    """Immutable snapshot of plan-mode progress.

    Transitions return a *new* machine; nothing mutates in place.
    """

    model_config = ConfigDict(frozen=True)

    state: PlanState = PlanState.CHAT
    last_request_id: str | None = None
    last_plan_markdown: str = ""
    last_preview: WorkflowPreviewView | None = None
    last_feedback: str = ""
    edited_plan: str | None = None
    edited_proposal: PlanProposal | None = None

    def request_plan(self) -> "PlanStateMachine":
        return self.model_copy(update={"state": PlanState.PLAN_REQUESTED})

    def emit_plan(
        self,
        request_id: str,
        plan_markdown: str,
        preview: WorkflowPreviewView,
    ) -> "PlanStateMachine":
        return self.model_copy(
            update={
                "state": PlanState.PLAN_EMITTED,
                "last_request_id": request_id,
                "last_plan_markdown": plan_markdown,
                "last_preview": preview,
            }
        )

    def approve(
        self,
        edited_plan: str | None = None,
        edited_proposal: PlanProposal | None = None,
    ) -> "PlanStateMachine":
        return self.model_copy(
            update={
                "state": PlanState.PLAN_APPROVED,
                "edited_plan": edited_plan,
                "edited_proposal": edited_proposal,
            }
        )

    def reject(self, feedback: str) -> "PlanStateMachine":
        return self.model_copy(
            update={
                "state": PlanState.PLAN_REJECTED,
                "last_feedback": feedback,
            }
        )

    def reset_to_chat(self) -> "PlanStateMachine":
        return PlanStateMachine()

    def is_parked(self) -> bool:
        return self.state == PlanState.PLAN_EMITTED


REJECT_FEEDBACK_TEMPLATE = "Plan rejected. Feedback: {feedback}. Revise the plan and emit it again."
"""Synthetic user-message wording the runner injects on plan reject."""


def render_reject_feedback(feedback: str) -> str:
    """Build the synthetic user message that follows ``PLAN_REJECTED``."""

    return REJECT_FEEDBACK_TEMPLATE.format(feedback=feedback.strip() or "(no feedback)")
