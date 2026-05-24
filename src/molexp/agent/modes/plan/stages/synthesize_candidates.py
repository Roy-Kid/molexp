"""Stage 4 — synthesize 1-3 candidate :class:`PlanGraph`\\ s."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, ClassVar

from molexp.agent.harness.events import AgentEvent
from molexp.agent.harness.stage import Stage
from molexp.agent.modes.plan.stages.thread_state import PlanThreadState
from molexp.agent.modes.plan.tasks_planning import synthesize_candidates

if TYPE_CHECKING:
    from molexp.agent.harness.harness import AgentHarness
    from molexp.agent.modes.plan._mode import PlanMode

__all__ = ["SynthesizeCandidates"]


class SynthesizeCandidates(Stage[PlanThreadState, PlanThreadState]):
    """Ask the planner LLM for 1-3 candidate ``PlanGraph``\\ s.

    Each candidate is persisted under the plan folder via
    :meth:`PlanFolder.write_candidate`. Yields the new
    :class:`PlanThreadState` carrying the ``candidates`` field.

    The ``pre_state="draft_plan"`` tag drives the lifecycle validator:
    on first entry it transitions ``exploring → draft_plan``; on a
    repair rewind from ``preflight_failed`` it transitions
    ``preflight_failed → draft_plan``; on a rewind from
    ``awaiting_approval`` (rejected direction) it transitions
    ``awaiting_approval → draft_plan``. All three moves are legal per
    :data:`LEGAL_TRANSITIONS`.
    """

    name: ClassVar[str] = "SynthesizeCandidates"
    pre_state: ClassVar[str | None] = "draft_plan"

    def __init__(self, *, plan_mode: PlanMode) -> None:
        self._plan_mode = plan_mode

    async def run(
        self,
        *,
        harness: AgentHarness,
        input: PlanThreadState,
    ) -> AsyncIterator[AgentEvent | PlanThreadState]:
        state = input
        assert state.intent is not None
        assert state.capabilities is not None
        candidates = await synthesize_candidates(
            router=harness.router,
            intent=state.intent,
            capabilities=state.capabilities,
        )
        for candidate in candidates.candidates:
            self._plan_mode.plan_folder.write_candidate(candidate.label, candidate.plan_graph)
        yield state.model_copy(update={"candidates": candidates})
