"""Runtime services consumed by the PlanMode workflow via ``ctx.deps``.

PlanMode's ``ctx.deps`` is a frozen :class:`PlanDeps` aggregate of
runtime services — provider routing, the tier policy, and the
on-disk experiment-workspace handle. ``ctx.config`` carries
JSON-only values (``user_input`` etc.); ``ctx.deps`` carries the
callables and stateful services that don't fit through a JSON
channel.

Sub-spec 06 will reintroduce gate / repair policy slots when the
human-review node lands; v1 of the rewrite (this sub-spec) drops
them — the materialize-to-workspace pipeline owns persistence
through :class:`~molexp.agent.modes.plan.workspace_layout.PlanWorkspaceHandle`
and there is no in-memory iteration counter to thread.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

if TYPE_CHECKING:
    from molexp.agent.modes.plan.policy import PlanModelPolicy
    from molexp.agent.modes.plan.workspace_layout import PlanWorkspaceHandle


__all__ = [
    "ModelTier",
    "PlanDeps",
    "Provider",
    "SchemaT",
]


# ── Tier vocabulary ────────────────────────────────────────────────────────


class ModelTier(StrEnum):
    """Semantic model tier — the only model identifier a Task may name.

    Concrete provider / model IDs live in :class:`Provider`
    configuration, never on Task classes. Tasks declare what they need
    (cheap parsing vs. heavy reasoning); operators decide what fulfils
    that need.
    """

    CHEAP = "cheap"
    DEFAULT = "default"
    HEAVY = "heavy"


# ── Provider ───────────────────────────────────────────────────────────────


SchemaT = TypeVar("SchemaT", bound=BaseModel)


@runtime_checkable
class Provider(Protocol):
    """LLM dispatch gateway — tier-routed, schema-typed.

    Tasks call this exactly once per LLM step. The provider owns model
    resolution, retries, tracing, and rate-limiting; tasks only know
    their tier and the schema they expect back.
    """

    async def invoke(
        self,
        *,
        tier: ModelTier,
        system: str,
        user: str,
        schema: type[SchemaT],
        node_id: str = "",
    ) -> SchemaT: ...

    async def invoke_with_template(
        self,
        *,
        tier: ModelTier,
        system: str,
        user_template: str,
        user_context: Mapping[str, Any],
        schema: type[SchemaT],
        node_id: str = "",
    ) -> SchemaT:
        """Render ``user_template`` against ``user_context`` then call :meth:`invoke`.

        Concrete providers MUST implement this; the abstract Protocol
        does not provide a default body. Template substitution failures
        surface as :class:`~molexp.agent._pydanticai.errors.ProviderError`
        with kind :attr:`ErrorKind.validation`.
        """
        ...


# ── PlanDeps aggregate ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class PlanDeps:
    """Runtime services bundle threaded through ``ctx.deps``.

    Three required fields, no defaults — the materialize-to-workspace
    pipeline is meaningless without all three:

    Attributes:
        provider: LLM dispatch gateway (concrete impl: ``PydanticAIProvider``).
        policy: Tier-aware model selection policy
            (:class:`~molexp.agent.modes.plan.policy.PlanModelPolicy`).
            Each task resolves its tier via
            ``policy.tier_for(type(self).__name__)``.
        workspace_handle: On-disk experiment-workspace handle
            (:class:`~molexp.agent.modes.plan.workspace_layout.PlanWorkspaceHandle`).
            All artifact writes route through this handle's API; tasks
            never touch ``Path.write_text`` directly.
    """

    provider: Provider
    policy: PlanModelPolicy
    workspace_handle: PlanWorkspaceHandle
