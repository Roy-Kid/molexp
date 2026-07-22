"""``Router`` — unified LLM dispatch protocol for ``molexp.agent``.

Every :class:`AgentLoop` reaches the LLM through a :class:`Router`.
ChatLoop wants a single text completion; PlanMode wants tier-routed
structured output. Both methods share one configuration surface
(:class:`AgentRunner` ``model=`` / ``models=`` kwargs) and one cache.

This module is the *protocol*, not the *implementation*. The concrete
class lives under the ``_pydanticai/`` firewall and is re-exported
**lazily** from the package surface — spell it
``from molexp.agent import PydanticAIRouter`` (never import from
``_pydanticai`` directly). This file imports nothing from pydantic-ai.
Stub routers used by tests implement the same protocol.

``RouterTextResult`` is a frozen :class:`dataclasses.dataclass` rather
than a pydantic model: it carries an opaque pydantic-ai
``AgentRunResult`` as ``raw``, which would otherwise force
``arbitrary_types_allowed=True`` (forbidden in :mod:`molexp.agent`
per the layer charter — see CLAUDE.md § "Pydantic vs plain class").
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, ConfigDict

from molexp.agent.types import UsageBreakdown

if TYPE_CHECKING:
    # Hook types live in ``molexp.agent.loops.hooks`` (SDK-free). Importing
    # them only under TYPE_CHECKING keeps this protocol module free of a
    # runtime import cycle with ``agent.loops`` — ``from __future__ import
    # annotations`` (above) defers every annotation to a string, so the
    # signatures below never trigger a runtime import.
    from molexp.agent.loops.hooks import (
        AfterToolHook,
        BeforeToolHook,
        ShouldStopGuard,
    )

__all__ = [
    "AgenticChunk",
    "FinalChunk",
    "McpToolSpec",
    "ModelTier",
    "Router",
    "RouterTextResult",
    "TextDeltaChunk",
    "ThinkingDeltaChunk",
    "TierModels",
    "ToolCallChunk",
    "ToolResultChunk",
]


# ── Tier vocabulary ────────────────────────────────────────────────────────


class ModelTier(StrEnum):
    """Semantic model tier — what a Task asks for, not which model it gets.

    The :class:`Router` resolves a tier to a concrete pydantic-ai
    model id via the ``models`` mapping the user supplied to
    :class:`~molexp.agent.AgentRunner`. Tasks declare what they need
    (cheap parsing vs. heavy reasoning); operators decide which model
    fulfils that need.
    """

    CHEAP = "cheap"
    DEFAULT = "default"
    HEAVY = "heavy"


# ── Tier → model map type ──────────────────────────────────────────────────


TierModels = Mapping[ModelTier, "str | object"]
"""Tier → model id mapping. Values are either pydantic-ai model strings
(``"deepseek:deepseek-v4-flash"``) or pydantic-ai ``models.Model``
instances (e.g. ``TestModel()`` for offline tests). The router treats
both shapes uniformly."""


# ── Result type for text completion ────────────────────────────────────────


@dataclass(frozen=True)
class RouterTextResult:
    """Normalized outcome of one :meth:`Router.complete_text` call.

    Attributes:
        text: The model's textual response.
        raw: The underlying pydantic-ai ``AgentRunResult`` if the
            router had one to expose; ``None`` for stub routers.
            Held opaquely (``Any``) so the agent layer can avoid
            importing pydantic-ai types here.
    """

    text: str
    raw: Any = field(default=None)


@dataclass(frozen=True)
class McpToolSpec:
    """SDK-free descriptor for a stdio MCP server to attach as an agent tool.

    The harness/services layers pass these to :meth:`Router.complete_structured`
    to make a codegen agent consult a live MCP server (e.g. molmcp's
    ``molcrafts_search`` / ``molcrafts_describe``) mid-generation instead of
    guessing an API from a static catalog. The concrete router turns each spec
    into a pydantic-ai MCP toolset — none of the callers touch pydantic-ai.

    Attributes:
        name: Toolset id; also the ``{name}_{tool}`` prefix on every tool.
        command: Executable launched over stdio (e.g. ``"molmcp"``).
        args: Command arguments.
        env: Extra environment as name/value pairs (tuple so the spec stays
            hashable/frozen); empty means inherit the parent environment.
    """

    name: str
    command: str
    args: tuple[str, ...] = ()
    env: tuple[tuple[str, str], ...] = ()


# ── Agentic-loop streaming chunks ──────────────────────────────────────────


class TextDeltaChunk(BaseModel):
    """One token-level assistant-text increment from the agentic loop."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["text_delta"] = "text_delta"
    text: str


class ThinkingDeltaChunk(BaseModel):
    """One reasoning-token increment from the agentic loop.

    Reasoning models (DeepSeek reasoner, Claude extended thinking) emit a
    private chain-of-thought *before* the answer. pydantic-ai surfaces it as
    a ``ThinkingPart`` / ``ThinkingPartDelta``, structurally a sibling of the
    answer's ``TextPart`` / ``TextPartDelta``. This chunk is its SDK-free
    projection — kept distinct from :class:`TextDeltaChunk` so a consumer can
    render reasoning in a collapsed / dimmed treatment rather than inline with
    the answer. A model that does not reason simply never yields one.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["thinking_delta"] = "thinking_delta"
    text: str


class ToolCallChunk(BaseModel):
    """A tool call the model dispatched inside the agentic loop.

    ``args_summary`` is a short human-readable rendering of the call
    arguments, never the full payload.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["tool_call"] = "tool_call"
    tool_name: str
    args_summary: str = ""


class ToolResultChunk(BaseModel):
    """The return of a dispatched tool call.

    ``ok`` is ``False`` when the tool raised / produced a retry prompt.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["tool_result"] = "tool_result"
    tool_name: str
    result_summary: str = ""
    ok: bool = True


class FinalChunk(BaseModel):
    """The terminal chunk — final assistant text + optional lossless history.

    ``model_messages_json`` is the pydantic-ai ``ModelMessage`` list as
    canonical JSON bytes (produced only inside ``_pydanticai``). Stubs omit
    it; production routers set it so :class:`~molexp.agent.loops.InteractiveLoop`
    can persist multiturn context without importing pydantic-ai.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["final"] = "final"
    text: str
    model_messages_json: bytes | None = None


AgenticChunk = TextDeltaChunk | ThinkingDeltaChunk | ToolCallChunk | ToolResultChunk | FinalChunk
"""SDK-free union of every chunk :meth:`Router.stream_agentic` yields.

Defined in this protocol module — *not* importing ``pydantic_ai`` — so
test fakes and the emergent
:class:`~molexp.agent.loops.interactive.InteractiveLoop` consume the
agentic loop without paying the SDK load cost. A :class:`ThinkingDeltaChunk`
(reasoning models only) precedes the answer's :class:`TextDeltaChunk`\\ s;
the terminal yield is always a :class:`FinalChunk`."""


# ── Router protocol ────────────────────────────────────────────────────────


SchemaT = TypeVar("SchemaT", bound=BaseModel)


@runtime_checkable
class Router(Protocol):
    """Unified LLM dispatch — text, structured, and agentic-loop paths.

    Three keyword-only dispatch methods:

    * :meth:`complete_text` — one free-form text round trip
      (``ChatLoop`` and any future single-shot mode).
    * :meth:`complete_structured` — schema-typed dispatch with retry
      and event hooks (``PlanMode`` per-task LLM calls).
    * :meth:`stream_agentic` — the emergent tool-using loop
      (``InteractiveLoop``): the model autonomously decides → calls a
      tool → observes → loops, streamed as an :data:`AgenticChunk` flow.

    All honor the ``tier`` axis: ``complete_text`` / ``stream_agentic``
    default to :attr:`ModelTier.DEFAULT`, ``complete_structured``
    requires it explicitly. The router's tier→model resolution is
    private; callers only know tiers.
    """

    async def complete_text(
        self,
        *,
        prompt: str,
        system: str = "",
        message_history: tuple[Any, ...] = (),
        tier: ModelTier = ModelTier.DEFAULT,
    ) -> RouterTextResult:
        """Drive one text round trip.

        Args:
            prompt: The user message.
            system: Optional system prompt; concrete routers may also
                read a fixed system prompt configured at construction.
            message_history: Opaque pydantic-ai ``ModelMessage`` tuple
                (or empty). Forwarded verbatim to the underlying
                ``Agent.run``.
            tier: Which tier's model to use. Defaults to ``DEFAULT``.

        Returns:
            :class:`RouterTextResult` with the model's text and the
            opaque raw run result.
        """
        ...

    async def complete_structured(
        self,
        *,
        tier: ModelTier,
        system: str,
        user: str,
        schema: type[SchemaT],
        node_id: str = "",
        mcp_tools: tuple[McpToolSpec, ...] = (),
    ) -> SchemaT:
        """Drive one schema-typed round trip with retry + event hooks.

        Args:
            tier: Which tier's model to use.
            system: System prompt.
            user: User prompt.
            schema: Pydantic model class; the router asks
                pydantic-ai's ``Agent`` to parse the response into
                this type.
            node_id: Caller-supplied identifier propagated into
                :class:`~molexp.agent._pydanticai.errors.ProviderError`
                and event records for traceability.
            mcp_tools: Stdio MCP servers to attach as agent tools for this
                call. When non-empty the agent may call those tools (e.g.
                molmcp code intelligence) before emitting its structured
                answer, so codegen consults the real API instead of guessing.

        Returns:
            One instance of ``schema``.

        Raises:
            ProviderError: On retry exhaustion or non-retryable
                failure (the
                :class:`~molexp.agent._pydanticai.errors.ProviderError`
                raised by the concrete router; the internal name
                remains ``ProviderError`` to minimize churn).
        """
        ...

    def stream_agentic(
        self,
        *,
        prompt: str,
        system: str = "",
        tools: tuple[Any, ...] = (),
        toolsets: tuple[Any, ...] = (),
        tier: ModelTier = ModelTier.DEFAULT,
        message_history: tuple[Any, ...] = (),
        before_tool: BeforeToolHook | None = None,
        after_tool: AfterToolHook | None = None,
        should_stop: ShouldStopGuard | None = None,
    ) -> AsyncIterator[AgenticChunk]:
        """Drive an emergent tool-using loop, streamed as :data:`AgenticChunk`\\ s.

        The model autonomously decides whether to answer or call a
        tool; the router runs the full agentic loop (tool dispatch,
        retries, message history) through pydantic-ai's native
        ``Agent.iter()`` — no hand-rolled loop. Each reasoning-token
        increment (reasoning models only) is a
        :class:`ThinkingDeltaChunk`, each model answer increment a
        :class:`TextDeltaChunk`, each dispatched call a
        :class:`ToolCallChunk`, each return a :class:`ToolResultChunk`.
        The terminal yield is always a :class:`FinalChunk`.

        Args:
            prompt: The user message.
            system: Optional system prompt for this loop.
            tools: Tools the model may call — opaque pydantic-ai
                ``Tool`` instances or bare callables, forwarded
                verbatim. The protocol stays SDK-free, hence ``Any``.
            toolsets: Opaque pydantic-ai toolset objects (typically from
                :func:`molexp.agent._pydanticai.mcp.build_mcp_server`).
                Empty means no MCP/toolset attachment. May be combined
                with ``tools``.
            tier: Which tier's model to use. Defaults to ``DEFAULT``.
            message_history: Opaque prior-turn history (or empty),
                forwarded verbatim to the underlying agent.
            before_tool: Optional :class:`~molexp.agent.loops.hooks.BeforeToolHook`
                consulted as each tool call is dispatched. ``None`` (the
                default) opts out — omitting it is byte-identical to today.
            after_tool: Optional :class:`~molexp.agent.loops.hooks.AfterToolHook`
                consulted as each tool result is observed. ``None`` (the
                default) opts out.
            should_stop: Optional
                :class:`~molexp.agent.loops.hooks.ShouldStopGuard` consulted
                before the terminal :class:`FinalChunk`. ``None`` (the default)
                opts out.

        Yields:
            :data:`AgenticChunk`\\ s in emission order; the last is a
            :class:`FinalChunk`.

        Note:
            Phase 01 honors only the after-tool boundary that passive
            ``Agent.iter()`` iteration permits — an ``after_tool`` returning
            :meth:`~molexp.agent.loops.hooks.HookOutcome.deny` flips the
            emitted :class:`ToolResultChunk` ``ok`` to ``False`` and folds the
            deny message into ``result_summary``. Before-tool veto,
            suspend-resume, and should-stop re-injection are *triggered and
            recorded* in phase 01 but their enforcement is realized in phase
            02. When all three hooks are ``None`` the chunk stream is
            byte-identical to omitting them entirely.
        """
        ...

    def clear_usage(self) -> None:
        """Forget any accumulated :class:`~molexp.agent.types.CallUsage`
        records. Modes call this at the start of each
        :meth:`AgentLoop.run` so the snapshot at the end reflects only
        that one run."""
        ...

    def snapshot_usage(self) -> UsageBreakdown:
        """Return an aggregate of every call recorded since the last
        :meth:`clear_usage`. Modes use this at end-of-run to render a
        per-call breakdown table and populate
        :attr:`AgentRunResult.usage`."""
        ...
