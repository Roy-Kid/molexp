"""Runtime services consumed by the PlanMode workflow via ``ctx.deps``.

PlanMode's ``ctx.deps`` is a frozen :class:`PlanDeps` aggregate of
runtime services — the LLM router, the tier policy, the on-disk
experiment-workspace handle, and a callable lookup for the
workflow-orthogonal :class:`~molexp.agent.policy.GatePolicy`.
``ctx.config`` carries JSON-only values (``user_input`` etc.);
``ctx.deps`` carries the callables and stateful services that don't
fit through a JSON channel.

Cross-mode types — :class:`Router` / :class:`ModelTier` from
:mod:`molexp.agent.router` — are re-exported here so
``from molexp.agent.modes.plan.protocols import …`` style imports
keep working for plan-specific code without leaking the agent-layer
location. Approval-gate types (:class:`GatePolicy`,
:class:`AutoApproveGatePolicy`, :func:`static_gate_policy_lookup`)
are deliberately NOT re-exported — they live at
:mod:`molexp.agent.policy` parallel to ``mode.py`` because any mode
with a multi-step workflow consumes them.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from molexp.agent.router import ModelTier, Router

if TYPE_CHECKING:
    from molexp.agent.modes.plan.capability import (
        CapabilityEvidenceBatch,
        CapabilityNeedReport,
    )
    from molexp.agent.modes.plan.policy import PlanModelPolicy
    from molexp.agent.modes.plan.schemas import (
        ApprovalDecision,
        PlanBrief,
        PlanReviewView,
        TaskIRBrief,
    )
    from molexp.agent.modes.plan.workspace_layout import PlanWorkspaceHandle
    from molexp.agent.policy import GatePolicy
    from molexp.workflow import WorkflowContract

    type PlanGatePolicy = GatePolicy[PlanReviewView, ApprovalDecision]
else:
    # At runtime ``PlanGatePolicy`` is just a forward-ref string. It is
    # only consumed in annotations (which are strings under
    # ``from __future__ import annotations``), so the runtime value
    # never needs to be a real class — keeping it as a string lets us
    # avoid a runtime import of ``GatePolicy`` here, which in turn
    # keeps the legacy-protocol-name guard test simple.
    PlanGatePolicy = "GatePolicy[PlanReviewView, ApprovalDecision]"

__all__ = [
    "CapabilityProbe",
    "ModelTier",
    "PlanDeps",
    "PlanGatePolicy",
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
        contract: WorkflowContract,
        briefs: tuple[TaskIRBrief, ...],
    ) -> CapabilityNeedReport:
        """Draft per-task capability needs.

        Args:
            plan_brief: Implementation-plan brief produced by
                ``DraftImplementationPlan``.
            contract: Typed workflow contract from ``CompileWorkflowIR``.
            briefs: Per-task IR briefs from ``CompileTaskIR``.

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


def _default_gate_policy_lookup() -> Callable[[], PlanGatePolicy]:
    """Default factory: constant lookup over an :class:`AutoApproveGatePolicy`
    bound to ``ApprovalDecision(approved=True)``."""
    from molexp.agent.modes.plan.schemas import ApprovalDecision
    from molexp.agent.policy import AutoApproveGatePolicy, static_gate_policy_lookup

    return static_gate_policy_lookup(AutoApproveGatePolicy(ApprovalDecision(approved=True)))


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
        gate_policy_lookup: Live lookup for the approval gate consulted
            by ``HumanReview``. Stored as a callable (not the policy
            itself) so the lifecycle owner — :class:`PlanMode` — can
            hot-swap the policy mid-run while ``PlanDeps`` itself stays
            frozen. Defaults to a constant lookup over an
            :class:`~molexp.agent.policy.AutoApproveGatePolicy` bound to
            ``ApprovalDecision(approved=True)``. Read sites should use
            the convenience :attr:`gate_policy` property below.
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
    gate_policy_lookup: Callable[[], PlanGatePolicy] = field(
        default_factory=_default_gate_policy_lookup
    )
    repair_target_tasks: tuple[str, ...] | None = None
    repair_iteration: int = 0
    capability_probe: CapabilityProbe | None = None

    @property
    def gate_policy(self) -> PlanGatePolicy:
        """Live :class:`GatePolicy` — calls :attr:`gate_policy_lookup` each access.

        Read-side ergonomics: existing call sites such as
        ``await ctx.deps.gate_policy.human_review(view)`` keep working
        unchanged, but each access reads through the lookup so a
        :class:`PlanMode` setter swap propagates immediately.
        """
        return self.gate_policy_lookup()
