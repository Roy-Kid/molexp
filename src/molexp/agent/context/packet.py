"""Context packet + budget metadata"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from molexp.agent.types import Message

_FROZEN = ConfigDict(frozen=True)


class ContextRef(BaseModel):
    """Pointer into a workspace artifact included in the context.

    The harness records what was included so debugging tools can show
    "this turn used these files / runs"; the model never sees the ref
    list, only the rendered system prompt + messages.
    """

    model_config = _FROZEN

    kind: str
    path: str
    excerpt: str = ""


class ContextBudget(BaseModel):
    """Soft caps tracked by the context manager.

    All values are character counts; the manager does not tokenize.
    Tokenization is provider-specific and lives behind the model
    plugin if needed.
    """

    model_config = _FROZEN

    max_chars: int = 200_000
    used_chars: int = 0
    history_chars: int = 0
    system_chars: int = 0


class ContextBuildRequest(BaseModel):
    """Inputs to :meth:`molexp.agent.context.ContextManager.build`."""

    model_config = _FROZEN

    session_id: str
    turn_id: str
    base_system: str
    workspace_addendum: str
    skill_addendum: str
    instructions_override: str | None
    history: tuple[Message, ...]
    extra_refs: tuple[ContextRef, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextPacket(BaseModel):
    """Final material handed to the model for one turn."""

    model_config = _FROZEN

    system: str
    messages: tuple[Message, ...]
    included_refs: tuple[ContextRef, ...]
    budget: ContextBudget
    diagnostics: tuple[str, ...] = ()
