"""``AgentLoop`` ABC + ``AgentRunResult`` value type.

Post spec ``harness-as-mode-substrate-03b`` the agent layer ships only
two loops — :class:`~molexp.agent.loops.ChatLoop` (one LLM round trip)
and :class:`~molexp.agent.loops.InteractiveLoop` (emergent tool loop).
Pipeline-style orchestration (Plan / Author / Run / Review) moved to
:mod:`molexp.harness`, so the substrate that ran them (``Stage``,
``ModePipeline``, ``PipelineEdge``, ``RepairPolicy``,
``run_pipeline``) is gone.

A loop is a plain async coroutine: ``async def run(*, runtime, sink,
user_input) -> None``. Events flow through the injected
:class:`~molexp.agent.events.AsyncIteratorEventSink`; the terminal
:class:`~molexp.agent.events.LoopCompletedEvent` carries the JSON dump
of the run's :class:`AgentRunResult` so the runner can rebuild it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from molexp.agent.types import Message, Usage, UsageBreakdown

if TYPE_CHECKING:
    from molexp.agent.events import AgentEvent, AsyncIteratorEventSink
    from molexp.agent.runtime import AgentRuntime


class AgentRunResult(BaseModel):
    """Outcome of one ``AgentRunner.run(...)`` call.

    Loops populate ``loop_state`` with loop-specific structured output;
    ChatLoop + InteractiveLoop leave it ``None``.

    ``usage`` is the aggregate token / request count for the run;
    ``usage_breakdown`` is the per-call list (one entry per LLM round
    trip). Both default empty when no LLM call is made.

    ``events`` holds the accumulated orchestration-level
    :data:`~molexp.agent.events.AgentEvent` stream the loop emitted
    while running — it defaults to ``()`` so callers that only want the
    terminal text are unaffected.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=False)

    text: str
    messages: tuple[Message, ...] = ()
    loop_state: dict[str, Any] | None = None
    usage: Usage = Field(default_factory=Usage)
    usage_breakdown: UsageBreakdown = Field(default_factory=UsageBreakdown)
    events: tuple[AgentEvent, ...] = ()


class AgentLoop(ABC):
    """Abstract strategy a loop must implement to be drivable by ``AgentRunner``.

    Subclasses set ``name`` to a stable identifier and implement
    :meth:`run` as a plain async coroutine that emits events through
    the injected ``sink`` and returns ``None``. ``runtime`` carries the
    four agent-layer services (session / router / execution_env /
    hooks); ``user_input`` is the end-user prompt.

    Resume contract
    ---------------
    :meth:`resume` is a classmethod that reconstructs a loop instance
    from persisted state. The default raises :exc:`NotImplementedError`;
    subclasses override it to read their own on-disk format.
    """

    name: str = ""

    @abstractmethod
    async def run(
        self,
        *,
        runtime: AgentRuntime,
        sink: AsyncIteratorEventSink,
        user_input: str,
    ) -> None:
        """Drive the loop, emitting orchestration events through ``sink``.

        The loop MUST emit a terminal
        :class:`~molexp.agent.events.LoopCompletedEvent` whose
        ``result`` carries the JSON dump of the run's
        :class:`AgentRunResult` so the runner can rebuild it.
        """
        ...

    @classmethod
    def resume(cls, **kwargs: Any) -> AgentLoop:  # noqa: ANN401
        """Reconstruct this loop from persisted state.

        Subclasses override this to read their own on-disk format.
        The default raises :exc:`NotImplementedError`.
        """
        raise NotImplementedError(f"{cls.__name__} does not support resume")


# Resolve the forward reference to AgentEvent so AgentRunResult can be
# validated/serialized at runtime (the field type is only TYPE_CHECKING
# imported above to keep the module's import graph shallow).
def _rebuild_models() -> None:
    """Inject ``AgentEvent`` and rebuild :class:`AgentRunResult`."""
    from molexp.agent.events import AgentEvent as _AgentEvent

    AgentRunResult.model_rebuild(_types_namespace={"AgentEvent": _AgentEvent})


_rebuild_models()


__all__ = ["AgentLoop", "AgentRunResult"]
