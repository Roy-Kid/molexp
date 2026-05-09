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
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

if TYPE_CHECKING:
    from molexp.agent.modes.plan.policy import PlanModelPolicy
    from molexp.agent.modes.plan.schemas import ApprovalDecision, PlanReviewView
    from molexp.agent.modes.plan.workspace_layout import PlanWorkspaceHandle


__all__ = [
    "AutoApproveGatePolicy",
    "GatePolicy",
    "InteractiveGatePolicy",
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


# ── Gate policy (sub-spec 06: terminal HumanReview node) ───────────────────


@runtime_checkable
class GatePolicy(Protocol):
    """Approval gate consulted by ``HumanReview`` at the pipeline tail.

    A single method — ``human_review`` — returning an
    :class:`ApprovalDecision`. PlanMode v1 ships only with the
    auto-approve and not-yet-implemented placeholders; production UIs
    will plug in their own implementation that prompts a human.
    """

    async def human_review(self, view: PlanReviewView) -> ApprovalDecision: ...


class AutoApproveGatePolicy:
    """Always-approve gate — the safe default for non-interactive use."""

    async def human_review(self, view: PlanReviewView) -> ApprovalDecision:
        del view  # protocol-pinned name; not consulted by auto-approve
        from molexp.agent.modes.plan.schemas import ApprovalDecision as _AD

        return _AD(approved=True)


class InteractiveGatePolicy:
    """Placeholder for a future UI / CLI-driven gate.

    Raises :class:`NotImplementedError` so callers who construct
    PlanMode without supplying their own gate are forced to either
    accept the auto-approve default or implement an interactive
    handler. The full interactive surface ships in a separate spec.
    """

    async def human_review(self, view: PlanReviewView) -> ApprovalDecision:
        del view
        raise NotImplementedError(
            "InteractiveGatePolicy is a placeholder — provide a concrete "
            "GatePolicy implementation or use AutoApproveGatePolicy."
        )


# ── PlanDeps aggregate ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class PlanDeps:
    """Runtime services bundle threaded through ``ctx.deps``.

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
        gate_policy: Approval gate consulted by ``HumanReview``.
            Defaults to :class:`AutoApproveGatePolicy` so non-interactive
            callers (tests, CLI happy paths) don't have to wire one in.
    """

    provider: Provider
    policy: PlanModelPolicy
    workspace_handle: PlanWorkspaceHandle
    gate_policy: GatePolicy = field(default_factory=AutoApproveGatePolicy)
