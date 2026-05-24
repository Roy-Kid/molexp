"""Stage 5 — pick one candidate and apply any pending repair diff."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, ClassVar

from molexp.agent.harness.events import AgentEvent, ArtifactWrittenEvent, PlanEmittedEvent
from molexp.agent.harness.stage import Stage
from molexp.agent.modes.plan.stages.thread_state import PlanThreadState
from molexp.agent.modes.plan.tasks_planning import select_plan

if TYPE_CHECKING:
    from molexp.agent.harness.harness import AgentHarness
    from molexp.agent.modes.plan._mode import PlanMode

__all__ = ["SelectPlan"]


class SelectPlan(Stage[PlanThreadState, PlanThreadState]):
    """Choose one candidate ``PlanGraph`` and apply any planted repair.

    After selection, consumes ``plan_mode._runtime.repair_signal``
    (planted by ``PreflightPlanGraph`` or ``EmitApprovedPlan``) and
    folds its ``PlanDiff`` into the selected plan via
    :meth:`PlanDiff.apply`. Writes ``selected_plan.json`` and emits
    :class:`PlanEmittedEvent`.
    """

    name: ClassVar[str] = "SelectPlan"

    def __init__(self, *, plan_mode: PlanMode) -> None:
        self._plan_mode = plan_mode

    async def run(
        self,
        *,
        harness: AgentHarness,
        input: PlanThreadState,
    ) -> AsyncIterator[AgentEvent | PlanThreadState]:
        state = input
        assert state.candidates is not None
        assert state.capabilities is not None
        selected = await select_plan(
            router=harness.router,
            candidates=state.candidates,
            capabilities=state.capabilities,
        )
        signal = self._plan_mode._runtime.consume()
        if signal is not None:
            selected = signal.plan_diff.apply(selected)
        path = self._plan_mode.plan_folder.write_selected_plan(selected)
        yield ArtifactWrittenEvent(path=str(path), description="selected PlanGraph")
        yield PlanEmittedEvent(plan_id=selected.plan_id, step_count=len(selected.steps))
        yield state.model_copy(update={"selected": selected})
