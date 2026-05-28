"""Stage 1 — synthesize a typed ``IntentSpec`` from free text."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, ClassVar

from molexp.agent.events import AgentEvent, ArtifactWrittenEvent
from molexp.agent.modes.plan.stages.thread_state import PlanThreadState
from molexp.agent.modes.plan.tasks_planning import synthesize_intent
from molexp.agent.stage import Stage

if TYPE_CHECKING:
    from molexp.agent.modes.plan._mode import PlanMode
    from molexp.agent.runtime import AgentHarness

__all__ = ["SynthesizeIntent"]


class SynthesizeIntent(Stage[str, PlanThreadState]):
    """Project the user's free-text input into a typed :class:`IntentSpec`.

    Consumes the entry ``user_input`` string; produces a
    :class:`PlanThreadState` carrying the new ``intent`` field and the
    original ``user_input``. Writes ``intent.json`` to the plan folder
    and emits an :class:`ArtifactWrittenEvent`.
    """

    name: ClassVar[str] = "SynthesizeIntent"

    def __init__(self, *, plan_mode: PlanMode) -> None:
        self._plan_mode = plan_mode

    async def run(
        self,
        *,
        harness: AgentHarness,
        input: str | PlanThreadState,
    ) -> AsyncIterator[AgentEvent | PlanThreadState]:
        user_input = input if isinstance(input, str) else input.user_input
        intent = await synthesize_intent(router=harness.router, user_input=user_input)
        path = self._plan_mode.plan_folder.write_intent(intent)
        yield ArtifactWrittenEvent(path=str(path), description="typed IntentSpec")
        yield PlanThreadState(user_input=user_input, intent=intent)
