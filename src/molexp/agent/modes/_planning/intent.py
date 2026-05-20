"""Intent cluster — the formal user-intent contract (``IntentSpec``).

Pure frozen-pydantic data models; no LLM, no I/O. Part of the shared
``molexp.agent.modes._planning`` planning-contracts package — the
substrate every agent mode (Plan / Author / Run / Review) reads.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class RiskLevel(StrEnum):
    """Coarse risk classification.

    Shared by :class:`IntentSpec`, ``PlanStep``, and ``CapabilityNode``.
    """

    low = "low"
    medium = "medium"
    high = "high"


class IntentConstraint(BaseModel):
    """One constraint the user places on the work.

    Attributes:
        kind: The constraint category (e.g. ``"time"``, ``"data"``).
        detail: A human-readable description of the constraint.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: str
    detail: str


class MissingInfoItem(BaseModel):
    """A question whose answer the planner still needs.

    Attributes:
        question: The open question to put to the user.
        blocking: Whether planning cannot proceed until it is answered.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    question: str
    blocking: bool


class SuccessCriterion(BaseModel):
    """One condition that defines a successful outcome.

    Attributes:
        summary: A human-readable statement of the criterion.
        verifiable: Whether the criterion can be checked mechanically.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    summary: str
    verifiable: bool


class ResourceBudget(BaseModel):
    """Cost / time / model-tier ceilings for the work.

    Attributes:
        max_cost_usd: Spend ceiling in USD, or ``None`` for unbounded.
        max_wall_seconds: Wall-clock ceiling in seconds, or ``None``.
        model_tier: Preferred model tier, or ``None`` for the default.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    max_cost_usd: float | None
    max_wall_seconds: float | None
    model_tier: str | None


class IntentSpec(BaseModel):
    """The formal user-intent contract captured at intake.

    Every later plan / repair / review step is judged against this
    contract so the user's original goal is never lost.

    Attributes:
        objective: The one-sentence goal of the work.
        non_goals: Things explicitly out of scope.
        required_outputs: Artefacts the user must receive.
        constraints: Hard limits the work must respect.
        assumptions: Conditions assumed true without verification.
        missing_information: Open questions for the user.
        success_criteria: Conditions defining a successful outcome.
        allowed_side_effects: Side effects the user has sanctioned.
        budget: Cost / time / model-tier ceilings.
        risk_level: Overall risk classification of the work.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    objective: str
    non_goals: tuple[str, ...]
    required_outputs: tuple[str, ...]
    constraints: tuple[IntentConstraint, ...]
    assumptions: tuple[str, ...]
    missing_information: tuple[MissingInfoItem, ...]
    success_criteria: tuple[SuccessCriterion, ...]
    allowed_side_effects: tuple[str, ...]
    budget: ResourceBudget
    risk_level: RiskLevel
