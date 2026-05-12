"""Session — value the user passes into ``AgentRunner.run``.

Carries ``session_id``, the molexp-shaped conversation log used for
display, **and** the pydantic-ai-native message history that survives
across turns. Plain Python class because it holds runtime references
(asyncio queues, history mutation methods) that pydantic
``BaseModel(frozen=True)`` forbids.

The ``model_messages`` field is typed ``tuple[Any, ...]`` so this
module stays free of any ``pydantic_ai`` import — the agent layer's
import-boundary firewall confines that SDK to ``agent/_pydanticai/``.
The opaque type underneath is ``pydantic_ai.messages.ModelMessage``;
serialization to disk goes through
``molexp.agent._pydanticai.messages_codec``.
"""

from __future__ import annotations

import secrets
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from molexp.agent.types import Message


class AgentSession:
    """One in-flight conversation between a user and an ``AgentRunner``.

    The runner mutates ``history`` and ``model_messages`` as it appends
    user / assistant / tool turns. Modes read both and may extend
    ``mode_state`` with mode-specific persisted state.

    ``history`` is the molexp-shaped log (one
    :class:`~molexp.agent.types.Message` per turn) — flat strings,
    suitable for UI rendering and JSONL replay.

    ``model_messages`` is the pydantic-ai-native ``ModelMessage`` tuple
    forwarded back to ``Agent.run(message_history=...)`` so the LLM sees
    the full conversation context on every turn. Modes replace this
    tuple wholesale after each call (the ``result.all_messages()``
    return is the cumulative list, including the new turn).
    """

    def __init__(
        self,
        *,
        session_id: str | None = None,
        history: list[Message] | None = None,
        mode_state: dict[str, Any] | None = None,
        model_messages: tuple[Any, ...] = (),
    ) -> None:
        self.session_id = session_id or secrets.token_hex(6)
        self.history: list[Message] = list(history or [])
        self.mode_state: dict[str, Any] = dict(mode_state or {})
        self.model_messages: tuple[Any, ...] = tuple(model_messages)

    def append(self, message: Message) -> None:
        self.history.append(message)


__all__ = ["AgentSession"]
