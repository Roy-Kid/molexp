"""RED-phase tests for the intent cluster of ``molexp.agent.modes._planning``.

The package does not exist yet; these tests are the RED phase of TDD and
fail at collection until the implementation lands.

Covers, per the testing rules:

- Basics  — full valid construction of every intent model + a
  ``model_dump(mode="json")`` round-trip.
- Edge cases — ``extra="forbid"`` rejects an unknown field.
- Immutability — ``frozen=True`` rejects attribute assignment.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from molexp.agent.modes._planning import (
    IntentConstraint,
    IntentSpec,
    MissingInfoItem,
    ResourceBudget,
    RiskLevel,
    SuccessCriterion,
)

# --------------------------------------------------------------------------
# fixtures (hand-built; no LLM, no I/O)
# --------------------------------------------------------------------------


def _budget() -> ResourceBudget:
    return ResourceBudget(
        max_cost_usd=12.5,
        max_wall_seconds=600.0,
        model_tier="standard",
    )


def _intent() -> IntentSpec:
    return IntentSpec(
        objective="Train and benchmark a QM9 energy regressor.",
        non_goals=("hyperparameter sweep", "production deployment"),
        required_outputs=("model.pt", "metrics.json"),
        constraints=(
            IntentConstraint(kind="time", detail="finish within one hour"),
            IntentConstraint(kind="data", detail="QM9 subset only"),
        ),
        assumptions=("GPU available", "dataset already downloaded"),
        missing_information=(
            MissingInfoItem(question="Which target property?", blocking=True),
            MissingInfoItem(question="Preferred random seed?", blocking=False),
        ),
        success_criteria=(
            SuccessCriterion(summary="MAE below 0.05 eV", verifiable=True),
            SuccessCriterion(summary="readable report", verifiable=False),
        ),
        allowed_side_effects=("write artifacts to run dir",),
        budget=_budget(),
        risk_level=RiskLevel.medium,
    )


# --------------------------------------------------------------------------
# basics — RiskLevel enum
# --------------------------------------------------------------------------


def test_risk_level_has_three_members() -> None:
    assert {m.value for m in RiskLevel} == {"low", "medium", "high"}


def test_risk_level_string_value() -> None:
    assert RiskLevel.high == "high"


# --------------------------------------------------------------------------
# basics — leaf models
# --------------------------------------------------------------------------


def test_intent_constraint_construction() -> None:
    constraint = IntentConstraint(kind="memory", detail="under 8 GB")
    assert constraint.kind == "memory"
    assert constraint.detail == "under 8 GB"


def test_missing_info_item_construction() -> None:
    item = MissingInfoItem(question="Which optimizer?", blocking=True)
    assert item.question == "Which optimizer?"
    assert item.blocking is True


def test_success_criterion_construction() -> None:
    criterion = SuccessCriterion(summary="tests pass", verifiable=True)
    assert criterion.summary == "tests pass"
    assert criterion.verifiable is True


def test_resource_budget_accepts_none_fields() -> None:
    budget = ResourceBudget(max_cost_usd=None, max_wall_seconds=None, model_tier=None)
    assert budget.max_cost_usd is None
    assert budget.max_wall_seconds is None
    assert budget.model_tier is None


# --------------------------------------------------------------------------
# basics — IntentSpec full construction
# --------------------------------------------------------------------------


def test_intent_spec_full_construction() -> None:
    intent = _intent()
    assert intent.objective.startswith("Train")
    assert intent.non_goals == ("hyperparameter sweep", "production deployment")
    assert len(intent.constraints) == 2
    assert intent.constraints[0].kind == "time"
    assert intent.budget.model_tier == "standard"
    assert intent.risk_level is RiskLevel.medium


# --------------------------------------------------------------------------
# basics — JSON round-trip
# --------------------------------------------------------------------------


def test_intent_spec_json_round_trip() -> None:
    intent = _intent()
    dumped = intent.model_dump(mode="json")
    restored = IntentSpec.model_validate(dumped)
    assert restored == intent


def test_intent_spec_json_dump_is_jsonable() -> None:
    dumped = _intent().model_dump(mode="json")
    assert dumped["risk_level"] == "medium"
    assert isinstance(dumped["constraints"], list)
    assert dumped["budget"]["max_cost_usd"] == 12.5


# --------------------------------------------------------------------------
# edge cases — extra="forbid"
# --------------------------------------------------------------------------


def test_intent_constraint_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        IntentConstraint(kind="time", detail="fast", weight=3)  # type: ignore[call-arg]


def test_missing_info_item_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        MissingInfoItem(question="q", blocking=False, urgent=True)  # type: ignore[call-arg]


def test_resource_budget_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        ResourceBudget(
            max_cost_usd=1.0,
            max_wall_seconds=1.0,
            model_tier="t",
            max_tokens=100,  # type: ignore[call-arg]
        )


def test_intent_spec_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        IntentSpec(
            objective="o",
            non_goals=(),
            required_outputs=(),
            constraints=(),
            assumptions=(),
            missing_information=(),
            success_criteria=(),
            allowed_side_effects=(),
            budget=_budget(),
            risk_level=RiskLevel.low,
            owner="bob",  # type: ignore[call-arg]
        )


# --------------------------------------------------------------------------
# immutability — frozen=True
# --------------------------------------------------------------------------


def test_intent_constraint_is_frozen() -> None:
    constraint = IntentConstraint(kind="time", detail="fast")
    with pytest.raises(ValidationError):
        constraint.detail = "slow"  # type: ignore[misc]


def test_resource_budget_is_frozen() -> None:
    budget = _budget()
    with pytest.raises(ValidationError):
        budget.max_cost_usd = 99.0  # type: ignore[misc]


def test_intent_spec_is_frozen() -> None:
    intent = _intent()
    with pytest.raises(ValidationError):
        intent.objective = "new objective"  # type: ignore[misc]
