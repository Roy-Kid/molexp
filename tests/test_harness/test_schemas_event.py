"""Tests for HarnessEvent + EventType (Phase 1 schema layer).

Locks the wire format per spec §4.2:
- flat single-class frozen pydantic
- typing.Literal[...] discriminator
- seq is non-negative int
- unknown type raises ValidationError
- payload is a free-form dict
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import get_args, get_origin

import pytest
from pydantic import ValidationError


def test_harness_event_round_trip() -> None:
    from molexp.harness.schemas.event import HarnessEvent

    event = HarnessEvent(
        id="evt-001",
        run_id="run-xyz",
        seq=1,
        type="stage_started",
        actor="harness",
        created_at=datetime(2026, 5, 26, tzinfo=UTC),
        payload={"stage": "SaveUserPlan"},
        artifact_ids=[],
    )
    dumped = event.model_dump_json()
    rehydrated = HarnessEvent.model_validate_json(dumped)
    assert rehydrated == event


def test_harness_event_is_frozen() -> None:
    from molexp.harness.schemas.event import HarnessEvent

    event = HarnessEvent(
        id="evt-001",
        run_id="run-xyz",
        seq=1,
        type="run_created",
        actor="harness",
        created_at=datetime(2026, 5, 26, tzinfo=UTC),
    )
    with pytest.raises(ValidationError):
        event.seq = 2  # type: ignore[misc]


def test_harness_event_defaults_payload_and_artifact_ids() -> None:
    from molexp.harness.schemas.event import HarnessEvent

    event = HarnessEvent(
        id="evt-001",
        run_id="run-xyz",
        seq=1,
        type="run_created",
        actor="harness",
        created_at=datetime(2026, 5, 26, tzinfo=UTC),
    )
    assert event.payload == {}
    assert event.artifact_ids == []


def test_harness_event_rejects_unknown_type() -> None:
    from molexp.harness.schemas.event import HarnessEvent

    with pytest.raises(ValidationError):
        HarnessEvent(
            id="evt-001",
            run_id="run-xyz",
            seq=1,
            type="not_a_real_event",  # type: ignore[arg-type]
            actor="harness",
            created_at=datetime(2026, 5, 26, tzinfo=UTC),
        )


def test_harness_event_seq_must_be_non_negative() -> None:
    from molexp.harness.schemas.event import HarnessEvent

    HarnessEvent(
        id="evt-001",
        run_id="run-xyz",
        seq=0,
        type="run_created",
        actor="harness",
        created_at=datetime(2026, 5, 26, tzinfo=UTC),
    )
    with pytest.raises(ValidationError):
        HarnessEvent(
            id="evt-001",
            run_id="run-xyz",
            seq=-1,
            type="run_created",
            actor="harness",
            created_at=datetime(2026, 5, 26, tzinfo=UTC),
        )


def test_event_type_is_literal_not_enum() -> None:
    from typing import Literal

    from molexp.harness.schemas import event as event_mod

    assert get_origin(event_mod.EventType) is Literal
    import enum

    for name in dir(event_mod):
        obj = getattr(event_mod, name)
        if isinstance(obj, type) and issubclass(obj, enum.Enum) and obj is not enum.Enum:
            pytest.fail(f"schemas/event.py must not define enum: {name}")


def test_event_type_covers_stage_artifact_validation_approval_lifecycle() -> None:
    """EventType ships the 30-value set from harness-goal.md §4.2."""
    from molexp.harness.schemas.event import EventType

    expected = {
        "run_created",
        "run_completed",
        "run_failed",
        "stage_started",
        "stage_completed",
        "stage_failed",
        "artifact_created",
        "artifact_validated",
        "validation_passed",
        "validation_failed",
        "agent_called",
        "agent_completed",
        "agent_failed",
        "tool_called",
        "tool_completed",
        "tool_failed",
        "task_started",
        "task_completed",
        "task_failed",
        "test_started",
        "test_completed",
        "test_failed",
        "approval_requested",
        "approval_granted",
        "approval_rejected",
        "policy_checked",
        "policy_passed",
        "policy_failed",
        "artifact_edge_created",
    }
    actual = set(get_args(EventType))
    missing = expected - actual
    assert not missing, f"EventType missing values: {missing}"
