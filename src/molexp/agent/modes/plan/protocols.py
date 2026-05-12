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

from molexp.agent.router import ModelTier, Router

if TYPE_CHECKING:
    from molexp.agent.modes.plan.capability import (
        CapabilityEvidenceBatch,
        CapabilityNeedReport,
    )
    from molexp.agent.modes.plan.policy import PlanModelPolicy
    from molexp.agent.modes.plan.schemas import PlanBrief
    from molexp.agent.modes.plan.workspace_layout import PlanWorkspaceHandle
    from molexp.agent.review import ReviewPolicy


__all__ = [
    "CapabilityProbe",
    "ModelTier",
    "PlanDeps",
    "Router",
]


# ── CapabilityProbe ────────────────────────────────────────────────────────


@runtime_checkable
class CapabilityProbe(Protocol):
    """Two-method abstraction over capability discovery.

    PlanMode's ``DraftCapabilityNeeds`` and ``DiscoverCapabilities``
    nodes delegate the LLM call + MCP plumbing to this Protocol so the
    nodes themselves stay free of ``pydantic_ai`` imports. Phase 4
    provides two concrete implementations:

    - :class:`~molexp.agent.modes.plan.tasks_capability.NullCapabilityProbe`
      — fallback when no MCP server is configured;
      :meth:`draft_needs` returns
      ``CapabilityNeedReport(discovery_required=False, …)``,
      :meth:`discover` raises
      :class:`~molexp.agent.modes.plan.errors.CapabilityDiscoveryRequired`
      whenever its input flips ``discovery_required=True``.
    - :class:`molexp.agent._pydanticai.capability_probe.PydanticAICapabilityProbe`
      — wraps two ``pydantic_ai.Agent`` instances (a no-tool structured
      agent for needs drafting, an MCP-attached agent for evidence
      collection).

    Tests inject ``StubCapabilityProbe`` implementations so the suite
    never reaches a real LLM or MCP server.
    """

    async def draft_needs(
        self,
        *,
        plan_brief: PlanBrief,
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
    ) -> CapabilityEvidenceBatch:
        """Resolve needs into concrete API evidence.

        Args:
            report: Output of :meth:`draft_needs`.

        Returns:
            :class:`CapabilityEvidenceBatch` populated with one
            :class:`CapabilityEvidence` per resolved need plus any
            :class:`MissingCapability` rows the MCP probe could not
            satisfy. ``discovery_skipped`` is propagated from
            ``report.discovery_required``.

        Raises:
            CapabilityDiscoveryRequired: When the report demands
                discovery but the probe is unable to perform it (no MCP
                server configured, etc.).
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
        workspace_handle: On-disk experiment-workspace handle
            (:class:`~molexp.agent.modes.plan.workspace_layout.PlanWorkspaceHandle`).
            All artifact writes route through this handle's API; tasks
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
            consumed by Phase 4's ``DraftCapabilityNeeds`` and
            ``DiscoverCapabilities`` nodes. Defaults to ``None`` so
            existing pipelines (Phases 0-3) keep constructing
            :class:`PlanDeps` unchanged; ``AgentRunner.run`` (updated
            in Phase 4) lazily wires in either a
            ``PydanticAICapabilityProbe`` (when molmcp is configured)
            or a ``NullCapabilityProbe`` (fallback).
    """

    router: Router
    policy: PlanModelPolicy
    workspace_handle: PlanWorkspaceHandle
    step_policy_lookup: Callable[[], ReviewPolicy] = field(default_factory=_default_bypass_lookup)
    final_policy_lookup: Callable[[], ReviewPolicy] = field(default_factory=_default_bypass_lookup)
    repair_target_tasks: tuple[str, ...] | None = None
    repair_iteration: int = 0
    step_outputs_log: dict[str, Any] = field(default_factory=dict)
    capability_probe: CapabilityProbe | None = None

    @property
    def step_policy(self) -> ReviewPolicy:
        """Live per-step :class:`ReviewPolicy` — reads through the lookup each access."""
        return self.step_policy_lookup()

    @property
    def final_policy(self) -> ReviewPolicy:
        """Live plan-final :class:`ReviewPolicy` — reads through the lookup each access."""
        return self.final_policy_lookup()
