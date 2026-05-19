"""Runtime services consumed by the PlanMode workflow via ``ctx.deps``.

PlanMode's ``ctx.deps`` is a frozen :class:`PlanDeps` aggregate of
runtime services — the LLM router, the tier policy, the on-disk
experiment-workspace handle, two callable lookups for the
workflow-orthogonal :class:`~molexp.agent.review.ReviewPolicy` hooks
(per-step and plan-final), and one mutable
:class:`~molexp.agent.modes.plan.state.PlanRuntimeState` slot the loop
mutates between iterations. ``ctx.config`` carries JSON-only values
(``user_input`` etc.); ``ctx.deps`` carries the callables and stateful
services that don't fit through a JSON channel.

Cross-mode types — :class:`Router` / :class:`ModelTier` from
:mod:`molexp.agent.router` — are re-exported here so
``from molexp.agent.modes.plan.protocols import …`` style imports
keep working for plan-specific code without leaking the agent-layer
location. The :class:`~molexp.agent.review.ReviewPolicy` protocol and
its built-in policies live at :mod:`molexp.agent.review` parallel to
``mode.py`` because any mode with a multi-step workflow consumes them.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from molexp.agent.capability_discovery import CapabilityDiscoveryService
from molexp.agent.modes.plan.context import PlanRepairContext
from molexp.agent.modes.plan.state import PlanRuntimeState
from molexp.agent.router import ModelTier, Router

if TYPE_CHECKING:
    from molexp.agent.modes.plan.capability import (
        CapabilityEvidenceBatch,
        CapabilityNeedReport,
    )
    from molexp.agent.modes.plan.plan_folder import PlanFolder
    from molexp.agent.modes.plan.policy import PlanModelPolicy
    from molexp.agent.modes.plan.schemas import PlanBrief
    from molexp.agent.review import ReviewPolicy


__all__ = [
    "CapabilityDiscoveryService",
    "CapabilityProbe",
    "ModelTier",
    "PlanDeps",
    "PlanRepairContext",
    "Router",
]


# ── CapabilityProbe ────────────────────────────────────────────────────────


@runtime_checkable
class CapabilityProbe(Protocol):
    """Two-method abstraction over capability discovery.

    Compatibility abstraction under the newer
    :class:`CapabilityDiscoveryService` service. Discovery nodes should
    depend on the service; this probe remains for tests and older
    callers that inject a lower-level source directly.

    - :class:`~molexp.agent.modes.plan.tasks_capability.NullCapabilityProbe`
      — fallback when no source is configured; :meth:`draft_needs`
      returns ``CapabilityNeedReport(discovery_required=False, …)``,
      :meth:`discover` returns ``CapabilityEvidenceBatch(needs_repair=...)``
      whenever its input flips ``discovery_required=True`` (no exception).
    - :class:`molexp.agent._pydanticai.capability_probe.PydanticAICapabilityProbe`
      — concrete source-backed implementation.

    Tests inject ``StubCapabilityProbe`` implementations so the suite
    never reaches a real external source.
    """

    async def draft_needs(
        self,
        *,
        plan_brief: PlanBrief,
        repair_context: PlanRepairContext | None = None,
    ) -> CapabilityNeedReport:
        """Draft per-stage capability needs from the implementation plan."""
        ...

    async def discover(
        self,
        report: CapabilityNeedReport,
        repair_context: PlanRepairContext | None = None,
    ) -> CapabilityEvidenceBatch:
        """Resolve needs into concrete API evidence.

        Compatibility contract: implementations MAY still raise
        :class:`~molexp.agent.modes.plan.errors.CapabilityDiscoveryRequired`
        when discovery is impossible; the
        :class:`~molexp.agent.modes.plan.tasks_capability.DiscoverCapabilities`
        task catches it and plants a
        :class:`~molexp.agent.modes.plan.state.RepairSignal` on
        ``ctx.deps.runtime`` so the workflow loop can drive a repair.
        """
        ...


def _default_bypass_lookup() -> Callable[[], ReviewPolicy]:
    """Default factory: constant lookup over a
    :class:`~molexp.agent.review.BypassPolicy`."""
    from molexp.agent.review import BypassPolicy

    policy = BypassPolicy()

    def _lookup() -> ReviewPolicy:
        return policy

    return _lookup


# ── PlanDeps aggregate ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class PlanDeps:
    """Runtime services bundle threaded through ``ctx.deps``.

    The aggregate itself is frozen — its fields are services and
    callables, not data the tasks mutate. The mutable cross-task scratch
    pad lives on :attr:`runtime` (a
    :class:`~molexp.agent.modes.plan.state.PlanRuntimeState`), whose
    contents the workflow loop and individual tasks update freely.

    Attributes:
        router: LLM dispatch gateway. PlanMode tasks call
            ``ctx.deps.router.complete_structured(...)``.
        policy: Tier-aware model selection policy. Each task resolves
            its tier via ``policy.tier_for(type(self).__name__)``.
        plan_folder: On-disk plan workspace. All artifact writes route
            through this folder's API; tasks never touch
            ``Path.write_text`` directly.
        step_policy_lookup: Live lookup for the per-step review policy
            fired by :class:`PlanTask` after every non-terminal node's
            ``_execute`` completes. Callable so :class:`PlanMode` can
            hot-swap the policy mid-run while :class:`PlanDeps` stays
            frozen.
        final_policy_lookup: Live lookup for the plan-final review
            policy consulted by ``HumanReview``. Same hot-swap
            machinery as :attr:`step_policy_lookup`.
        capability_probe: :class:`CapabilityProbe` implementation
            retained as a compatibility escape hatch.
        capability_discovery: Service consumed by
            ``DraftCapabilityNeeds`` and ``DiscoverCapabilities``.
        runtime: Live mutable state shared across loop iterations of
            the plan workflow. Tasks read ``ctx.deps.runtime.iteration``
            etc. and plant repair signals on
            ``ctx.deps.runtime.repair_signal`` instead of raising.
    """

    router: Router
    policy: PlanModelPolicy
    plan_folder: PlanFolder
    step_policy_lookup: Callable[[], ReviewPolicy] = field(default_factory=_default_bypass_lookup)
    final_policy_lookup: Callable[[], ReviewPolicy] = field(default_factory=_default_bypass_lookup)
    capability_probe: CapabilityProbe | None = None
    capability_discovery: CapabilityDiscoveryService | None = None
    runtime: PlanRuntimeState = field(default_factory=PlanRuntimeState)

    @property
    def step_policy(self) -> ReviewPolicy:
        """Live per-step :class:`ReviewPolicy` — reads through the lookup each access."""
        return self.step_policy_lookup()

    @property
    def final_policy(self) -> ReviewPolicy:
        """Live plan-final :class:`ReviewPolicy` — reads through the lookup each access."""
        return self.final_policy_lookup()

    @property
    def repair_iteration(self) -> int:
        """Convenience: ``self.runtime.iteration``. Read-only view for tasks."""
        return self.runtime.iteration

    @property
    def repair_context(self) -> PlanRepairContext:
        """Convenience: structured repair feedback from the previous iteration.

        Always returns a usable :class:`PlanRepairContext` — empty on
        the first iteration, populated after :class:`RepairDecide` has
        consumed a rejection.
        """
        ctx = self.runtime.repair_context
        if ctx is None:
            return PlanRepairContext()
        return ctx
