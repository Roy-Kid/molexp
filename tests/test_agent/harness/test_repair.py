"""Tests for ``RepairPolicy`` introduced by agent-mode-stage-pipeline-01.

Covers ac-002 of the substrate spec: ``RepairPolicy`` is a frozen
pydantic BaseModel with four required fields
(``trigger_event_kind`` / ``rewind_to`` / ``max_iterations`` /
``on_exhausted``) and no ``arbitrary_types_allowed=True`` escape hatch.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from molexp.agent.repair import RepairPolicy


def test_repair_policy_is_a_pydantic_basemodel() -> None:
    assert issubclass(RepairPolicy, BaseModel)


def test_repair_policy_is_frozen() -> None:
    policy = RepairPolicy(
        trigger_event_kind="preflight_failed",
        rewind_to="SynthesizeCandidates",
        max_iterations=3,
        on_exhausted="preflight_failed",
    )
    with pytest.raises(ValidationError):
        policy.max_iterations = 5  # type: ignore[misc]


def test_repair_policy_required_fields() -> None:
    """All four fields are required — omitting any one raises."""
    with pytest.raises(ValidationError):
        RepairPolicy(  # type: ignore[call-arg]
            rewind_to="X",
            max_iterations=1,
            on_exhausted="terminal",
        )
    with pytest.raises(ValidationError):
        RepairPolicy(  # type: ignore[call-arg]
            trigger_event_kind="kind",
            max_iterations=1,
            on_exhausted="terminal",
        )
    with pytest.raises(ValidationError):
        RepairPolicy(  # type: ignore[call-arg]
            trigger_event_kind="kind",
            rewind_to="X",
            on_exhausted="terminal",
        )
    with pytest.raises(ValidationError):
        RepairPolicy(  # type: ignore[call-arg]
            trigger_event_kind="kind",
            rewind_to="X",
            max_iterations=1,
        )


def test_repair_policy_field_types() -> None:
    """``max_iterations`` is int; the other three are str."""
    fields = RepairPolicy.model_fields
    assert fields["trigger_event_kind"].annotation is str
    assert fields["rewind_to"].annotation is str
    assert fields["max_iterations"].annotation is int
    assert fields["on_exhausted"].annotation is str


def test_repair_policy_does_not_allow_arbitrary_types() -> None:
    """``arbitrary_types_allowed`` stays False per agent-layer charter."""
    assert RepairPolicy.model_config.get("arbitrary_types_allowed") is not True


def test_repair_policy_json_round_trip() -> None:
    policy = RepairPolicy(
        trigger_event_kind="preflight_failed",
        rewind_to="SynthesizeCandidates",
        max_iterations=3,
        on_exhausted="preflight_failed",
    )
    assert RepairPolicy.model_validate(policy.model_dump()) == policy
