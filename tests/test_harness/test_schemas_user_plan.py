"""Tests for UserPlan (Phase 2 §4.4 schema).

Locks the wire format:
- frozen pydantic round-trip
- attachments default to empty list, metadata default to empty dict
- user_id is optional
- attachments accept ArtifactRef instances (typed)
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError


def test_user_plan_round_trip() -> None:
    from molexp.harness.schemas.artifact import ArtifactRef
    from molexp.harness.schemas.user_plan import UserPlan

    attachment = ArtifactRef(
        id="abc12345",
        kind="input_file",
        uri="file:///tmp/x.dcd",
        sha256="0" * 64,
        created_at=datetime(2026, 5, 26, tzinfo=UTC),
        created_by="user",
    )
    plan = UserPlan(
        raw_text="I want to simulate water at 300K.",
        user_id="alice",
        submitted_at=datetime(2026, 5, 26, tzinfo=UTC),
        attachments=[attachment],
        metadata={"source": "cli"},
    )
    dumped = plan.model_dump_json()
    rehydrated = UserPlan.model_validate_json(dumped)
    assert rehydrated == plan


def test_user_plan_is_frozen() -> None:
    from molexp.harness.schemas.user_plan import UserPlan

    plan = UserPlan(
        raw_text="hi",
        submitted_at=datetime(2026, 5, 26, tzinfo=UTC),
    )
    with pytest.raises(ValidationError):
        plan.raw_text = "mutated"  # type: ignore[misc]


def test_user_plan_defaults() -> None:
    from molexp.harness.schemas.user_plan import UserPlan

    plan = UserPlan(
        raw_text="hi",
        submitted_at=datetime(2026, 5, 26, tzinfo=UTC),
    )
    assert plan.user_id is None
    assert plan.attachments == []
    assert plan.metadata == {}


def test_user_plan_attachments_must_be_artifact_refs() -> None:
    from molexp.harness.schemas.user_plan import UserPlan

    with pytest.raises(ValidationError):
        UserPlan(
            raw_text="hi",
            submitted_at=datetime(2026, 5, 26, tzinfo=UTC),
            attachments=[{"not": "an artifact ref"}],  # type: ignore[list-item]
        )
