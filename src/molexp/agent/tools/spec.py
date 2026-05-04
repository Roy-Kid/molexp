"""Tool boundary types per spec §5.3.

Tool execution lives entirely inside the harness; the model plugin
surfaces ``ModelToolCall`` objects (see :mod:`molexp.agent.model`) and
the dispatcher (see :mod:`molexp.agent.tools.dispatcher`) translates
them into ``ToolResult``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, TYPE_CHECKING

from molexp.agent.types import AgentFailure, ArtifactRef

if TYPE_CHECKING:  # pragma: no cover - import guard, used only for typing
    from molexp.agent.state.memory import MemoryStore


@dataclass(frozen=True)
class ToolSpec:
    """Static description of one tool.

    ``source`` is the canonical owner prefix. The model-facing
    ``ToolSchema`` may strip it if a provider has strict naming rules,
    but the harness retains the canonical name in state and events.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    source: str = "native"
    category: str = "workspace"
    mutates: bool = False
    requires_approval: bool = False
    tags: tuple[str, ...] = ()


@dataclass
class ToolContext:
    """Mutable per-call context handed to a tool callable.

    Not frozen because tools may attach short-lived bookkeeping
    (e.g. a per-call cache); persistent state must be stored through
    the workspace or through ``memory``.
    """

    workspace: Any
    session_id: str
    turn_id: str
    run: Any | None = None
    memory: "MemoryStore | None" = None


@dataclass(frozen=True)
class ToolResult:
    """Normalized return shape per spec §5.3.

    On failure ``ok`` is False and ``error`` is set; on success
    ``value`` carries the JSON-encodable payload (for the model) and
    ``artifacts`` carries any inline artifacts the UI should render.
    """

    ok: bool
    value: Any = None
    error: AgentFailure | None = None
    artifacts: tuple[ArtifactRef, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


ToolCallable = Callable[[dict[str, Any], ToolContext], Awaitable[ToolResult]]
"""Signature every tool implementation must satisfy."""


@dataclass(frozen=True)
class RegisteredTool:
    """Bookkeeping bundle stored inside :class:`ToolRegistry`."""

    spec: ToolSpec
    fn: ToolCallable
