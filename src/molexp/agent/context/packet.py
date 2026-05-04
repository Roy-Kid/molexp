"""Context packet + budget metadata per spec §6.1."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from molexp.agent.types import Message


@dataclass(frozen=True)
class ContextRef:
    """Pointer into a workspace artifact included in the context.

    The harness records what was included so debugging tools can show
    "this turn used these files / runs"; the model never sees the ref
    list, only the rendered system prompt + messages.
    """

    kind: str
    path: str
    excerpt: str = ""


@dataclass(frozen=True)
class ContextBudget:
    """Soft caps tracked by the context manager.

    All values are character counts; the manager does not tokenize.
    Tokenization is provider-specific and lives behind the model
    plugin if needed.
    """

    max_chars: int = 200_000
    used_chars: int = 0
    history_chars: int = 0
    system_chars: int = 0


@dataclass(frozen=True)
class ContextBuildRequest:
    """Inputs to :meth:`molexp.agent.context.ContextManager.build`."""

    session_id: str
    turn_id: str
    base_system: str
    workspace_addendum: str
    skill_addendum: str
    instructions_override: str | None
    history: tuple[Message, ...]
    extra_refs: tuple[ContextRef, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ContextPacket:
    """Final material handed to the model for one turn."""

    system: str
    messages: tuple[Message, ...]
    included_refs: tuple[ContextRef, ...]
    budget: ContextBudget
    diagnostics: tuple[str, ...] = ()
