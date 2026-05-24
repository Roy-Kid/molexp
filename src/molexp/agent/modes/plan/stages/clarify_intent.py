"""Stage 2 — clarify or pass through; blocking missing-info halts the run."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, ClassVar

from molexp.agent.harness.events import AgentEvent, ClarificationRequiredEvent
from molexp.agent.harness.stage import Stage
from molexp.agent.modes._planning import PlanState
from molexp.agent.modes.plan.stages.thread_state import PlanThreadState
from molexp.agent.modes.plan.tasks_planning import clarify_intent
from molexp.agent.types import Message

if TYPE_CHECKING:
    from molexp.agent.harness.harness import AgentHarness
    from molexp.agent.modes.plan._mode import PlanMode

__all__ = ["ClarifyIntent"]


class ClarifyIntent(Stage[PlanThreadState, PlanThreadState]):
    """Inspect ``IntentSpec.missing_information`` for blocking items.

    On a blocking item: transition ``plan_folder`` to
    ``needs_clarification``, append an assistant message asking the
    questions, and yield :class:`ClarificationRequiredEvent` — a
    registered :class:`~molexp.agent.harness.repair.RepairPolicy`
    routes the pipeline to the ``needs_clarification`` terminal.
    Otherwise: transition to ``exploring`` and pass the state through.
    """

    name: ClassVar[str] = "ClarifyIntent"

    def __init__(self, *, plan_mode: PlanMode) -> None:
        self._plan_mode = plan_mode

    async def run(
        self,
        *,
        harness: AgentHarness,
        input: PlanThreadState,
    ) -> AsyncIterator[AgentEvent | PlanThreadState]:
        state = input
        assert state.intent is not None, "ClarifyIntent ran without SynthesizeIntent output"
        next_state, blocking = clarify_intent(intent=state.intent)
        if next_state is PlanState.needs_clarification:
            self._plan_mode._transition(PlanState.needs_clarification)
            questions = "; ".join(item.question for item in blocking)
            harness.session.append_message(
                Message(
                    role="assistant",
                    content=f"Need clarification before planning: {questions}",
                )
            )
            yield ClarificationRequiredEvent(questions=questions)
            yield state
            return
        self._plan_mode._transition(PlanState.exploring)
        yield state
