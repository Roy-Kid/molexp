"""``AgentMode`` ABC + ``AgentRunResult`` value type.

A mode encodes the strategy: PlanMode runs a multi-step planning
workflow; ChatMode does a single LLM round-trip; ReviewMode is reserved
for phase 2. The ``AgentRunner`` injects a ``PydanticAIHarness`` (private)
into the mode at run time — user code never constructs a harness.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from molexp.agent.types import Message

if TYPE_CHECKING:
    from molexp.agent._pydanticai.harness import PydanticAIHarness
    from molexp.agent.session import AgentSession


class AgentRunResult(BaseModel):
    """Outcome of one ``AgentRunner.run(...)`` call.

    Modes populate ``mode_state`` with mode-specific structured output
    (a plan, a review verdict, …); ChatMode leaves it ``None``.
    """

    model_config = ConfigDict(frozen=True)

    text: str
    messages: tuple[Message, ...] = ()
    mode_state: dict[str, Any] | None = None


class AgentMode(ABC):
    """Abstract strategy a mode must implement to be drivable by ``AgentRunner``.

    Subclasses set ``name`` to a stable identifier and implement
    :meth:`run`. The ``harness`` keyword is supplied by ``AgentRunner``;
    user code does not call ``run`` directly.
    """

    name: str = ""

    @abstractmethod
    async def run(
        self,
        *,
        harness: PydanticAIHarness,
        session: AgentSession,
        user_input: str,
    ) -> AgentRunResult: ...


__all__ = ["AgentMode", "AgentRunResult"]
