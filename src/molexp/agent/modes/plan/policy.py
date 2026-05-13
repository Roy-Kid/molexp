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


PLAN_NODE_NAMES: Final[frozenset[str]] = frozenset(
    {
        "IngestReport",
        "DraftReportDigest",
        "ClarifyMissingInformation",
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
    }
)
"""Union of every node id any :class:`PlanModelPolicy` may reference.

Construction-time validation rejects any key not in this set."""


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
        "IngestReport": ModelTier.CHEAP,
        "DraftReportDigest": ModelTier.CHEAP,
        "ClarifyMissingInformation": ModelTier.CHEAP,
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
    },
)
"""Default policy applied when ``PlanDeps.model_policy`` is unset.

Cheap tier for cleanup / digest / validation; default for design /
IR / skeleton / tests; heavy for full impl + discovery."""
