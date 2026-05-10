"""Tier-aware model-selection policy for PlanMode.

PlanMode tasks declare *what* tier they need (cheap parsing vs. heavy
reasoning); operators decide *what model* fulfils that need. The
:class:`PlanModelPolicy` keeps that mapping in one place and is
threaded through ``ctx.deps.model_policy`` so individual task classes
no longer hard-code their tier.

The policy class is consumed by both today's pipeline (``IntakeTask``
… ``HandoffTask``) and the renamed sub-spec 05 pipeline
(``IngestReport`` … ``RepairOnValidationFailure``); a single union of
allowed node names lives in :data:`PLAN_NODE_NAMES` so a typo in
``node_tiers`` surfaces at construction.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from pydantic import BaseModel, ConfigDict, model_validator

from molexp.agent.modes.plan.protocols import ModelTier

__all__ = [
    "PLAN_NODE_NAMES",
    "STANDARD_PLAN_POLICY",
    "PlanModelPolicy",
]


# ── Node-name vocabulary ───────────────────────────────────────────────────


_CURRENT_NODE_NAMES: frozenset[str] = frozenset(
    {
        "IntakeTask",
        "GoalTask",
        "ContextTask",
        "MethodTask",
        "DecompositionTask",
        "ProtocolTask",
        "PreviewTask",
        "GateATask",
        "CodegenTask",
        "CompileTask",
        "DryRunTask",
        "GateBTask",
        "RepairTask",
        "HandoffTask",
    }
)
"""Class names of the present-day 14-task PlanMode pipeline."""


_FUTURE_NODE_NAMES: frozenset[str] = frozenset(
    {
        "IngestReport",
        "DraftReportDigest",
        "ValidateWorkspace",
        "DraftImplementationPlan",
        "CompileWorkflowIR",
        "CompileTaskIR",
        "DraftCapabilityNeeds",
        "DiscoverCapabilities",
        "GenerateWorkflowSkeleton",
        "GenerateTaskTests",
        "GenerateTaskImplementations",
        "HumanReview",
        "FinalHandoffCheck",
        "RepairOnValidationFailure",
    }
)
"""Node names introduced by sub-spec 05's pipeline rewrite."""


PLAN_NODE_NAMES: Final[frozenset[str]] = _CURRENT_NODE_NAMES | _FUTURE_NODE_NAMES
"""Union of every node id any :class:`PlanModelPolicy` may reference.

Bridging set — keeps the policy class valid against today's pipeline
*and* sub-spec 05's renamed pipeline without churn at the rename
boundary."""


# ── Policy class ───────────────────────────────────────────────────────────


class PlanModelPolicy(BaseModel):
    """Frozen mapping from PlanMode node id → :class:`ModelTier`.

    Lookup falls back to :attr:`default_tier` when ``node_id`` is not
    in :attr:`node_tiers`. Construction-time validation rejects keys
    not in :data:`PLAN_NODE_NAMES` so a typo trips immediately rather
    than silently degrading to default.

    Attributes:
        default_tier: Tier returned when no per-node override applies.
            Defaults to :attr:`ModelTier.DEFAULT`.
        node_tiers: Explicit node-id → tier overrides. Empty by
            default. Keys must be members of :data:`PLAN_NODE_NAMES`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    default_tier: ModelTier = ModelTier.DEFAULT
    node_tiers: Mapping[str, ModelTier] = {}

    @model_validator(mode="after")
    def _check_known_node_ids(self) -> PlanModelPolicy:
        unknown = sorted(set(self.node_tiers) - PLAN_NODE_NAMES)
        if unknown:
            raise ValueError(
                f"PlanModelPolicy.node_tiers references unknown node id(s): "
                f"{unknown}. Allowed names live in PLAN_NODE_NAMES."
            )
        return self

    def tier_for(self, node_id: str) -> ModelTier:
        """Resolve the tier for ``node_id``.

        Args:
            node_id: PlanMode workflow node identifier (typically the
                Task class ``__name__``).

        Returns:
            The override from :attr:`node_tiers` if present,
            otherwise :attr:`default_tier`.
        """
        return self.node_tiers.get(node_id, self.default_tier)


# ── Standard policy ────────────────────────────────────────────────────────


STANDARD_PLAN_POLICY: Final[PlanModelPolicy] = PlanModelPolicy(
    default_tier=ModelTier.DEFAULT,
    node_tiers={
        # Sub-spec 05 node names — operator-friendly cheap / default /
        # heavy split per the user's tier table.
        "IngestReport": ModelTier.CHEAP,
        "DraftReportDigest": ModelTier.CHEAP,
        "ValidateWorkspace": ModelTier.CHEAP,
        "DraftImplementationPlan": ModelTier.DEFAULT,
        "CompileWorkflowIR": ModelTier.DEFAULT,
        "CompileTaskIR": ModelTier.DEFAULT,
        "DraftCapabilityNeeds": ModelTier.HEAVY,
        "DiscoverCapabilities": ModelTier.HEAVY,
        "GenerateWorkflowSkeleton": ModelTier.DEFAULT,
        "GenerateTaskTests": ModelTier.DEFAULT,
        "GenerateTaskImplementations": ModelTier.HEAVY,
        "HumanReview": ModelTier.CHEAP,
        "FinalHandoffCheck": ModelTier.CHEAP,
        "RepairOnValidationFailure": ModelTier.HEAVY,
        # Current pipeline node names — preserve each task's
        # pre-existing TIER ClassVar value so migration is
        # behavior-equivalent.
        "IntakeTask": ModelTier.CHEAP,
        "GoalTask": ModelTier.CHEAP,
        "ContextTask": ModelTier.CHEAP,
        "MethodTask": ModelTier.DEFAULT,
        "DecompositionTask": ModelTier.HEAVY,
        "ProtocolTask": ModelTier.HEAVY,
        "CodegenTask": ModelTier.HEAVY,
    },
)
"""Default policy applied when ``PlanDeps.model_policy`` is unset.

Two halves:

- The sub-spec 05 entries set the tier table the user requested:
  cheap for cleanup / digest / validation; default for design /
  IR / skeleton / tests; heavy for full impl + repair.
- The current-pipeline entries mirror each ``*Task`` class's prior
  ``TIER`` ClassVar so the migration to policy-driven dispatch is
  bit-equivalent for callers running today's 14-task workflow.
"""
