"""Tool boundary types

Tool execution lives entirely inside the harness; the model plugin
surfaces ``ModelToolCall`` objects (see :mod:`molexp.agent.model`) and
the dispatcher (see :mod:`molexp.agent.tools.dispatcher`) translates
them into ``ToolResult``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel, ConfigDict, Field

from molexp._typing import JSONMapping, JSONValue, TaskOutput
from molexp.agent.memory.store import MemoryStore
from molexp.agent.types import AgentFailure

if TYPE_CHECKING:
    from molexp.workspace.run import RunContext
    from molexp.workspace.workspace import Workspace


class _ChatGatewayLike(Protocol):
    """Mid-turn user-prompt gateway used by ``native:ask_user``-style tools."""

    async def ask(self, prompt: str) -> str: ...


class ToolSpec(BaseModel):
    """Static description of one tool.

    ``source`` is the canonical owner prefix. The model-facing
    ``ToolSchema`` may strip it if a provider has strict naming rules,
    but the harness retains the canonical name in state and events.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    input_schema: JSONMapping
    source: str = "native"
    category: str = "workspace"
    mutates: bool = False
    requires_approval: bool = False
    tags: tuple[str, ...] = ()


class ToolResult(BaseModel):
    """Normalized return shape

    On failure ``ok`` is False and ``error`` is set; on success
    ``value`` carries the JSON-encodable payload (for the model) and
    ``artifacts`` carries asset URIs (``asset://...``) the UI should
    render. Tools that produce artifacts must write them through the
    workspace ``ArtifactAccessor`` and put the resulting ``asset.uri``
    into this field.
    """

    model_config = ConfigDict(frozen=True)

    ok: bool
    value: TaskOutput = None
    error: AgentFailure | None = None
    artifacts: tuple[str, ...] = ()
    metadata: dict[str, JSONValue] = Field(default_factory=dict)


ToolCallable = Callable[[JSONMapping, "ToolContext"], Awaitable[ToolResult]]
"""Signature every tool implementation must satisfy."""


class ToolContext:
    """Mutable per-call context handed to a tool callable.

    A plain Python class — not a BaseModel — because it carries live
    runtime references (workspace, run context, chat gateway, memory
    store) that are not pydantic data contracts. The "arbitrary types"
    pydantic escape hatch is banned in the agent layer; runtime
    carriers go here.

    Mutability is intentional: tools may attach short-lived bookkeeping
    (e.g. a per-call cache); persistent state must be stored through
    the workspace or through ``memory``.

    ``chat`` is the gateway tools use to talk to the user mid-turn
    (e.g. ``native:ask_user``). Set by the runner; tools that don't
    need it can ignore the field.
    """

    def __init__(
        self,
        workspace: Workspace | None,
        session_id: str,
        turn_id: str,
        run: RunContext | None = None,
        memory: MemoryStore | None = None,
        chat: _ChatGatewayLike | None = None,
    ) -> None:
        self.workspace = workspace
        self.session_id = session_id
        self.turn_id = turn_id
        self.run = run
        self.memory = memory
        self.chat = chat


class RegisteredTool:
    """Bookkeeping bundle stored inside :class:`ToolRegistry`.

    A plain Python class because it carries a live ``ToolCallable``,
    which pydantic cannot validate. Treated as logically immutable by
    convention — the registry never mutates after insert.
    """

    __slots__ = ("fn", "spec")

    def __init__(self, spec: ToolSpec, fn: ToolCallable) -> None:
        self.spec = spec
        self.fn = fn
