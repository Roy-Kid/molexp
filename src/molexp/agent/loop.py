"""``AgentRunResult`` — outcome of one :meth:`AgentRunner.run` call.

There is no molexp-owned conversation loop. Chat is one
``Router.complete_text``. Tool-using work is one ReAct
(``Router.stream_agentic``). Plan orchestration is a workflow.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from molexp.agent.types import Message, Usage, UsageBreakdown

if TYPE_CHECKING:
    from molexp.agent.events import AgentEvent


class AgentRunResult(BaseModel):
    """Outcome of one ``AgentRunner.run(...)`` call.

    ``usage`` is the aggregate token / request count for the run;
    ``usage_breakdown`` is the per-call list (one entry per LLM round
    trip). Both default empty when no LLM call is made.

    ``events`` holds the accumulated
    :data:`~molexp.agent.events.AgentEvent` stream emitted while
    running — it defaults to ``()`` so callers that only want the
    terminal text are unaffected.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=False)

    text: str
    messages: tuple[Message, ...] = ()
    loop_state: dict[str, Any] | None = None
    usage: Usage = Field(default_factory=Usage)
    usage_breakdown: UsageBreakdown = Field(default_factory=UsageBreakdown)
    events: tuple[AgentEvent, ...] = ()


def _rebuild_models() -> None:
    """Inject ``AgentEvent`` and rebuild :class:`AgentRunResult`."""
    from molexp.agent.events import AgentEvent as _AgentEvent

    AgentRunResult.model_rebuild(_types_namespace={"AgentEvent": _AgentEvent})


_rebuild_models()


__all__ = ["AgentRunResult"]
