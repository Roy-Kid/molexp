"""``Router`` — unified LLM dispatch protocol for ``molexp.agent``.

Every :class:`AgentLoop` reaches the LLM through a :class:`Router`.
ChatLoop wants a single text completion; PlanMode wants tier-routed
structured output. Both methods share one configuration surface
(:class:`AgentRunner` ``model=`` / ``models=`` kwargs) and one cache.

This module is the *protocol*, not the *implementation*. The concrete
class :class:`~molexp.agent._pydanticai.router.PydanticAIRouter` lives
under the ``_pydanticai/`` firewall — this file imports nothing from
pydantic-ai. Stub routers used by tests implement the same protocol.

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
from typing import Any, Literal, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel, ConfigDict

from molexp.agent.types import UsageBreakdown

__all__ = [
    "AgenticChunk",
    "FinalChunk",
    "ModelTier",
    "Router",
    "RouterTextResult",
    "TextDeltaChunk",
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


# ── Agentic-loop streaming chunks ──────────────────────────────────────────


class TextDeltaChunk(BaseModel):
    """One token-level assistant-text increment from the agentic loop."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["text_delta"] = "text_delta"
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
    """The terminal chunk — carries the agentic loop's final assistant text."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["final"] = "final"
    text: str


AgenticChunk = TextDeltaChunk | ToolCallChunk | ToolResultChunk | FinalChunk
"""SDK-free union of every chunk :meth:`Router.stream_agentic` yields.

Defined in this protocol module — *not* importing ``pydantic_ai`` — so
test fakes and the emergent
:class:`~molexp.agent.loops.interactive.InteractiveLoop` consume the
agentic loop without paying the SDK load cost. The terminal yield is
always a :class:`FinalChunk`."""


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

        Returns:
            One instance of ``schema``.

        Raises:
            ProviderError: On retry exhaustion or non-retryable
                failure. (The internal name remains
                ``ProviderError`` to minimize churn — catch via
                ``from molexp.agent._pydanticai.errors import ProviderError``.)
        """
        ...

    def stream_agentic(
        self,
        *,
        prompt: str,
        system: str = "",
        tools: tuple[Any, ...] = (),
        tier: ModelTier = ModelTier.DEFAULT,
        message_history: tuple[Any, ...] = (),
    ) -> AsyncIterator[AgenticChunk]:
        """Drive an emergent tool-using loop, streamed as :data:`AgenticChunk`\\ s.

        The model autonomously decides whether to answer or call a
        tool; the router runs the full agentic loop (tool dispatch,
        retries, message history) through pydantic-ai's native
        ``Agent.iter()`` — no hand-rolled loop. Each model text
        increment is a :class:`TextDeltaChunk`, each dispatched call a
        :class:`ToolCallChunk`, each return a :class:`ToolResultChunk`.
        The terminal yield is always a :class:`FinalChunk`.

        Args:
            prompt: The user message.
            system: Optional system prompt for this loop.
            tools: Tools the model may call — opaque pydantic-ai
                ``Tool`` instances or bare callables, forwarded
                verbatim. The protocol stays SDK-free, hence ``Any``.
            tier: Which tier's model to use. Defaults to ``DEFAULT``.
            message_history: Opaque prior-turn history (or empty),
                forwarded verbatim to the underlying agent.

        Yields:
            :data:`AgenticChunk`\\ s in emission order; the last is a
            :class:`FinalChunk`.
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
