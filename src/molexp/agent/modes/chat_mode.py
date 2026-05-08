"""``ChatMode`` — single-turn LLM round-trip mode.

Drives one prompt through :class:`PydanticAIHarness`, appends the
exchange to the session, and returns the assistant text. Multi-turn
support is implicit: the session history is the conversation log; each
``AgentRunner.run`` call is one assistant turn.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from molexp.agent.mode import AgentMode, AgentRunResult
from molexp.agent.types import Message

if TYPE_CHECKING:
    from molexp.agent._pydanticai.harness import PydanticAIHarness
    from molexp.agent.session import AgentSession


class ChatModeConfig(BaseModel):
    """Tunables for :class:`ChatMode`."""

    model_config = ConfigDict(frozen=True)

    system_prompt: str = ""
    temperature: float | None = None


class ChatMode(AgentMode):
    """One ``user_input`` → one LLM call → one ``AgentRunResult``."""

    name = "chat"

    def __init__(self, *, config: ChatModeConfig | None = None) -> None:
        self.config = config or ChatModeConfig()

    async def run(
        self,
        *,
        harness: PydanticAIHarness,
        session: AgentSession,
        user_input: str,
    ) -> AgentRunResult:
        session.append(Message(role="user", content=user_input))
        result = await harness.complete(user_input)
        session.append(Message(role="assistant", content=result.text))
        return AgentRunResult(text=result.text, messages=tuple(session.history))


__all__ = ["ChatMode", "ChatModeConfig"]
