"""``AgentMode`` ABC + ``AgentRunResult`` value type.

A mode encodes the strategy: PlanMode runs a multi-step planning
workflow; ChatMode does a single LLM round-trip; ReviewMode is reserved
for phase 2. The :class:`AgentRunner` injects a :class:`Router` into
the mode at run time — user code never constructs the underlying
pydantic-ai client directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from molexp.agent.types import Message, Usage, UsageBreakdown

if TYPE_CHECKING:
    from molexp.agent.router import Router
    from molexp.agent.session import AgentSession


class AgentRunResult(BaseModel):
    """Outcome of one ``AgentRunner.run(...)`` call.

    Modes populate ``mode_state`` with mode-specific structured output
    (a plan, a review verdict, …); ChatMode leaves it ``None``.

    ``usage`` is the aggregate token / request count for the run;
    ``usage_breakdown`` is the per-call list (one entry per LLM round
    trip) — useful for cost attribution across pipeline nodes. Both
    default empty when no LLM call is made (e.g. cached / stub modes).
    """

    model_config = ConfigDict(frozen=True)

    text: str
    messages: tuple[Message, ...] = ()
    mode_state: dict[str, Any] | None = None
    usage: Usage = Field(default_factory=Usage)
    usage_breakdown: UsageBreakdown = Field(default_factory=UsageBreakdown)


class AgentMode(ABC):
    """Abstract strategy a mode must implement to be drivable by ``AgentRunner``.

    Subclasses set ``name`` to a stable identifier and implement
    :meth:`run`. The ``router`` keyword is supplied by ``AgentRunner``;
    user code does not call ``run`` directly.

    Resume contract
    ---------------
    :meth:`resume` is a classmethod that reconstructs a mode instance
    from persisted state. The default raises :exc:`NotImplementedError`;
    subclasses override it to read their own on-disk format (e.g.
    :class:`~molexp.agent.modes.plan.PlanMode` restores from a
    :class:`~molexp.agent.modes.plan.plan_folder.PlanFolder`).
    """

    name: str = ""

    @abstractmethod
    async def run(
        self,
        *,
        router: Router,
        session: AgentSession,
        user_input: str,
    ) -> AgentRunResult: ...

    @classmethod
    def resume(cls, **kwargs: Any) -> AgentMode:  # noqa: ANN401
        """Reconstruct this mode from persisted state.

        Subclasses override this to read their own on-disk format.
        The default raises :exc:`NotImplementedError`.
        """
        raise NotImplementedError(f"{cls.__name__} does not support resume")


__all__ = ["AgentMode", "AgentRunResult"]
