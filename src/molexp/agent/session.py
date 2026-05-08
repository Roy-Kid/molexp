"""Session — value the user passes into ``AgentRunner.run``.

Carries ``session_id`` and the conversation history that survives across
turns. Plain Python class because it holds runtime references (asyncio
queues, history mutation methods) that pydantic ``BaseModel(frozen=True)``
forbids.
"""

from __future__ import annotations

import secrets
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from molexp.agent.types import Message


class AgentSession:
    """One in-flight conversation between a user and an ``AgentRunner``.

    The runner mutates ``history`` as it appends user / assistant /
    tool turns. Modes read it and may extend ``mode_state`` with
    mode-specific persisted state.
    """

    def __init__(
        self,
        *,
        session_id: str | None = None,
        history: list[Message] | None = None,
        mode_state: dict[str, Any] | None = None,
    ) -> None:
        self.session_id = session_id or secrets.token_hex(6)
        self.history: list[Message] = list(history or [])
        self.mode_state: dict[str, Any] = dict(mode_state or {})

    def append(self, message: Message) -> None:
        self.history.append(message)


__all__ = ["AgentSession"]
