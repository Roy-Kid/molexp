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
from typing import TYPE_CHECKING, TypeAlias

from molexp.agent.router import ModelTier, Router

if TYPE_CHECKING:
    from molexp.agent.modes.plan.policy import PlanModelPolicy
    from molexp.agent.modes.plan.schemas import ApprovalDecision, PlanReviewView
    from molexp.agent.modes.plan.workspace_layout import PlanWorkspaceHandle
    from molexp.agent.policy import GatePolicy

    PlanGatePolicy: TypeAlias = GatePolicy[PlanReviewView, ApprovalDecision]
else:
    # At runtime ``PlanGatePolicy`` is just a forward-ref string. It is
    # only consumed in annotations (which are strings under
    # ``from __future__ import annotations``), so the runtime value
    # never needs to be a real class — keeping it as a string lets us
    # avoid a runtime import of ``GatePolicy`` here, which in turn
    # keeps the legacy-protocol-name guard test simple.
    PlanGatePolicy = "GatePolicy[PlanReviewView, ApprovalDecision]"

__all__ = [
    "ModelTier",
    "PlanDeps",
    "PlanGatePolicy",
    "Router",
]


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
    """

    router: Router
    policy: PlanModelPolicy
    workspace_handle: PlanWorkspaceHandle
    gate_policy_lookup: Callable[[], PlanGatePolicy] = field(default_factory=_default_gate_policy_lookup)
    repair_target_tasks: tuple[str, ...] | None = None
    repair_iteration: int = 0

    @property
    def gate_policy(self) -> PlanGatePolicy:
        """Live :class:`GatePolicy` — calls :attr:`gate_policy_lookup` each access.

        Read-side ergonomics: existing call sites such as
        ``await ctx.deps.gate_policy.human_review(view)`` keep working
        unchanged, but each access reads through the lookup so a
        :class:`PlanMode` setter swap propagates immediately.
        """
        return self.gate_policy_lookup()
