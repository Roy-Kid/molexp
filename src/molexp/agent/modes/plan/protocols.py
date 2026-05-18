"""Runtime services consumed by the PlanMode workflow via ``ctx.deps``.

PlanMode's ``ctx.deps`` is a frozen :class:`PlanDeps` aggregate of
runtime services — the LLM router, the tier policy, the on-disk
experiment-workspace handle, and two callable lookups for the
workflow-orthogonal :class:`~molexp.agent.review.ReviewPolicy` hooks
(per-step and plan-final).  ``ctx.config`` carries JSON-only values
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
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from molexp.agent.capability_discovery import CapabilityDiscoveryService
from molexp.agent.modes.plan.context import PlanRepairContext
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
      — fallback when no source is configured;
      :meth:`draft_needs` returns
      ``CapabilityNeedReport(discovery_required=False, …)``,
      :meth:`discover` raises
      :class:`~molexp.agent.modes.plan.errors.CapabilityDiscoveryRequired`
      whenever its input flips ``discovery_required=True``.
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
        """Draft per-stage capability needs from the implementation plan.

        Runs before the workflow IR is compiled — the only upstream
        artefact is the natural-language plan brief. Discovery
        consumes the resulting report; ``CompileWorkflowIR`` /
        ``CompileTaskIR`` then write typed TaskIO from the evidence
        batch instead of guessing project-specific types.

        Args:
            plan_brief: Implementation-plan brief produced by
                ``DraftImplementationPlan``.

        Returns:
            Structured :class:`CapabilityNeedReport`. Setting
            ``discovery_required=False`` short-circuits the downstream
            ``DiscoverCapabilities`` node entirely (pure-stdlib paths).
        """
        ...

    async def discover(
        self,
        report: CapabilityNeedReport,
        repair_context: PlanRepairContext | None = None,
    ) -> CapabilityEvidenceBatch:
        """Resolve needs into concrete API evidence.

        Args:
            report: Output of :meth:`draft_needs`.

        Returns:
            :class:`CapabilityEvidenceBatch` populated with one
            :class:`CapabilityEvidence` per resolved need plus any
            :class:`MissingCapability` rows the source could not
            satisfy. ``discovery_skipped`` is propagated from
            ``report.discovery_required``.

        Raises:
            CapabilityDiscoveryRequired: When the report demands
                discovery but the probe is unable to perform it.
        """
        ...


def _default_bypass_lookup() -> Callable[[], ReviewPolicy]:
    """Default factory: constant lookup over a
    :class:`~molexp.agent.review.BypassPolicy`.

    Used by both the per-step hook and the plan-final hook when callers
    do not configure either explicitly; behaviour is "never block,
    always approve".
    """
    from molexp.agent.review import BypassPolicy

    policy = BypassPolicy()

    def _lookup() -> ReviewPolicy:
        return policy

    return _lookup


# ── PlanDeps aggregate ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class PlanDeps:
    """Runtime services bundle threaded through ``ctx.deps``.

    Attributes:
        router: LLM dispatch gateway implementing :class:`Router`.
            PlanMode tasks call ``ctx.deps.router.complete_structured(...)``.
            Concrete impl: :class:`~molexp.agent._pydanticai.router.PydanticAIRouter`.
        policy: Tier-aware model selection policy
            (:class:`~molexp.agent.modes.plan.policy.PlanModelPolicy`).
            Each task resolves its tier via
            ``policy.tier_for(type(self).__name__)``.
        plan_folder: On-disk plan workspace
            (:class:`~molexp.agent.modes.plan.plan_folder.PlanFolder`).
            All artifact writes route through this folder's API; tasks
            never touch ``Path.write_text`` directly.
        step_policy_lookup: Live lookup for the per-step review policy
            fired by :class:`PlanTask` after every node's ``_execute``
            completes (except terminal nodes that own their own review
            interaction).  Stored as a callable so the lifecycle owner
            — :class:`PlanMode` — can hot-swap the policy mid-run while
            :class:`PlanDeps` itself stays frozen.  Defaults to a
            :class:`~molexp.agent.review.BypassPolicy` lookup so
            existing pipelines run unattended.  Read sites should use
            the convenience :attr:`step_policy` property.
        final_policy_lookup: Live lookup for the plan-final review
            policy consulted by ``HumanReview``.  Same hot-swap
            machinery as :attr:`step_policy_lookup`; defaults to the
            same bypass policy.  Read sites should use
            :attr:`final_policy`.
        repair_target_tasks: Optional subset of experiment-task ids that
            ``GenerateTaskTests`` / ``GenerateTaskImplementations`` are
            permitted to regenerate this round. ``None`` means "regenerate
            every task brief" (the default fresh-pass behavior); a
            non-None tuple restricts the LLM call to the listed task ids
            so untouched tasks reuse last-iteration outputs from disk.
            Set by :func:`drive_with_repair` between iterations; tasks
            never read this directly outside the two codegen nodes.
        repair_iteration: Zero on the first round; incremented before
            each repair iteration. Surfaced into the
            :class:`~molexp.agent.modes.plan.schemas.PlanReviewView`
            constructed by ``HumanReview`` so reviewers see which round
            they are in.
        step_outputs_log: Mutable dict mapping plan-node name → most
            recent approved output.  Populated by
            :meth:`~molexp.agent.modes.plan.tasks.PlanTask.execute`
            immediately after the per-step
            :class:`~molexp.agent.review.ReviewPolicy` approves a step.
            Lets :func:`drive_with_repair` seed boundary stubs when a
            :class:`StepRejected` exception interrupts the run — so
            rejecting one step replays only that step (plus cascade),
            not the whole pipeline from scratch.  The same dict is
            threaded through every ``dataclass.replace`` the repair
            loop performs so the log accumulates across iterations.
        capability_probe: :class:`CapabilityProbe` implementation
            retained as a compatibility escape hatch. New wiring should
            prefer :attr:`capability_discovery`, which owns hint
            extraction plus the underlying probe.
        capability_discovery: service consumed by
            ``DraftCapabilityNeeds`` and ``DiscoverCapabilities``.
            PlanMode only calls this abstract service; transport,
            policy, and source-specific lookup stay behind the service.
        repair_context: Structured feedback from the previous rejection
            when this is a repair iteration. First pass uses the empty
            context. LLM nodes render this centrally into their prompts
            so reviewer feedback is not lost between iterations.
    """

    router: Router
    policy: PlanModelPolicy
    plan_folder: PlanFolder
    step_policy_lookup: Callable[[], ReviewPolicy] = field(default_factory=_default_bypass_lookup)
    final_policy_lookup: Callable[[], ReviewPolicy] = field(default_factory=_default_bypass_lookup)
    repair_target_tasks: tuple[str, ...] | None = None
    repair_iteration: int = 0
    step_outputs_log: dict[str, Any] = field(default_factory=dict)
    capability_probe: CapabilityProbe | None = None
    capability_discovery: CapabilityDiscoveryService | None = None
    repair_context: PlanRepairContext = field(default_factory=PlanRepairContext)

    @property
    def step_policy(self) -> ReviewPolicy:
        """Live per-step :class:`ReviewPolicy` — reads through the lookup each access."""
        return self.step_policy_lookup()

    @property
    def final_policy(self) -> ReviewPolicy:
        """Live plan-final :class:`ReviewPolicy` — reads through the lookup each access."""
        return self.final_policy_lookup()
