"""Stage 3 — probe + project; emit a typed ``CapabilityGraph``."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, ClassVar

from molexp.agent.harness.events import AgentEvent, ArtifactWrittenEvent
from molexp.agent.harness.stage import Stage
from molexp.agent.modes.plan.capability_projection import capability_projection
from molexp.agent.modes.plan.stages.thread_state import PlanThreadState

if TYPE_CHECKING:
    from molexp.agent.harness.harness import AgentHarness
    from molexp.agent.modes.plan._mode import PlanMode

__all__ = ["ExploreCapabilities"]


class ExploreCapabilities(Stage[PlanThreadState, PlanThreadState]):
    """Probe the project's capability surface and project to a graph.

    Runs the mode's resolved :class:`CapabilityProbe` against the
    intent, then folds the :class:`ProbeResult` into the typed
    :class:`CapabilityGraph` via
    :func:`capability_projection`. Writes ``capability_graph.json``
    and yields an :class:`ArtifactWrittenEvent`.
    """

    name: ClassVar[str] = "ExploreCapabilities"
    pre_state: ClassVar[str | None] = "exploring"

    def __init__(self, *, plan_mode: PlanMode) -> None:
        self._plan_mode = plan_mode

    async def run(
        self,
        *,
        harness: AgentHarness,  # noqa: ARG002 — kept for substrate's Stage.run contract
        input: PlanThreadState,
    ) -> AsyncIterator[AgentEvent | PlanThreadState]:
        state = input
        assert state.intent is not None, "ExploreCapabilities ran without an intent"
        probe_result = await self._plan_mode._probe.probe(intent=state.intent)
        capabilities = capability_projection(probe_result)
        path = self._plan_mode.plan_folder.write_capability_graph(capabilities)
        yield ArtifactWrittenEvent(path=str(path), description="typed CapabilityGraph")
        yield state.model_copy(update={"capabilities": capabilities})
