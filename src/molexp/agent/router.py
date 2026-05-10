"""``Router`` — unified LLM dispatch protocol for ``molexp.agent``.

Every :class:`AgentMode` reaches the LLM through a :class:`Router`.
ChatMode wants a single text completion; PlanMode wants tier-routed
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

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

from molexp.agent.types import UsageBreakdown

__all__ = [
    "ModelTier",
    "Router",
    "RouterTextResult",
    "TierModels",
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


# ── Router protocol ────────────────────────────────────────────────────────


SchemaT = TypeVar("SchemaT", bound=BaseModel)


@runtime_checkable
class Router(Protocol):
    """Unified LLM dispatch — both text and structured paths.

    Two methods, both keyword-only:

    * :meth:`complete_text` — one free-form text round trip
      (``ChatMode`` and any future single-shot mode).
    * :meth:`complete_structured` — schema-typed dispatch with retry
      and event hooks (``PlanMode`` per-task LLM calls).

    Both honor the ``tier`` axis: ``complete_text`` defaults to
    :attr:`ModelTier.DEFAULT`, ``complete_structured`` requires it
    explicitly. The router's tier→model resolution is private; callers
    only know tiers.
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

    def clear_usage(self) -> None:
        """Forget any accumulated :class:`~molexp.agent.types.CallUsage`
        records. Modes call this at the start of each
        :meth:`AgentMode.run` so the snapshot at the end reflects only
        that one run."""
        ...

    def snapshot_usage(self) -> UsageBreakdown:
        """Return an aggregate of every call recorded since the last
        :meth:`clear_usage`. Modes use this at end-of-run to render a
        per-call breakdown table and populate
        :attr:`AgentRunResult.usage`."""
        ...
