"""Tests for ArtifactRef + ArtifactKind (Phase 1 schema layer).

Locks the wire format and validation contract per spec §4.1:
- frozen pydantic round-trip
- ArtifactKind is a typing.Literal alias (not Enum)
- unknown kind raises ValidationError
- sha256 is bare hex (no "sha256:" prefix)
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import get_args, get_origin

import pytest
from pydantic import ValidationError


def test_artifact_ref_round_trip() -> None:
    from molexp.harness.schemas.artifact import ArtifactRef

    ref = ArtifactRef(
        id="a1b2c3d4",
        kind="workflow_ir",
        uri="file:///tmp/wf.json",
        sha256="0" * 64,
        created_at=datetime(2026, 5, 26, tzinfo=UTC),
        created_by="harness",
        parent_ids=["parent-1"],
        metadata={"package": "molpy", "version": "0.1.0"},
    )
    dumped = ref.model_dump_json()
    rehydrated = ArtifactRef.model_validate_json(dumped)
    assert rehydrated == ref


def test_artifact_ref_is_frozen() -> None:
    from molexp.harness.schemas.artifact import ArtifactRef

    ref = ArtifactRef(
        id="a1b2c3d4",
        kind="log",
        uri="file:///x",
        sha256="0" * 64,
        created_at=datetime(2026, 5, 26, tzinfo=UTC),
        created_by="harness",
    )
    with pytest.raises(ValidationError):
        ref.id = "mutated"  # type: ignore[misc]


def test_artifact_ref_defaults_parent_ids_and_metadata() -> None:
    from molexp.harness.schemas.artifact import ArtifactRef

    ref = ArtifactRef(
        id="a1b2c3d4",
        kind="log",
        uri="file:///x",
        sha256="0" * 64,
        created_at=datetime(2026, 5, 26, tzinfo=UTC),
        created_by="harness",
    )
    assert ref.parent_ids == []
    assert ref.metadata == {}


def test_artifact_ref_rejects_unknown_kind() -> None:
    from molexp.harness.schemas.artifact import ArtifactRef

    with pytest.raises(ValidationError):
        ArtifactRef(
            id="a1b2c3d4",
            kind="not_a_real_kind",  # type: ignore[arg-type]
            uri="file:///x",
            sha256="0" * 64,
            created_at=datetime(2026, 5, 26, tzinfo=UTC),
            created_by="harness",
        )


def test_artifact_ref_sha256_must_be_bare_hex() -> None:
    """ArtifactRef.sha256 stores bare hex, not the 'sha256:<hex>' prefixed form."""
    from molexp.harness.schemas.artifact import ArtifactRef

    # Bare hex of length 64 is accepted.
    ArtifactRef(
        id="a1b2c3d4",
        kind="log",
        uri="file:///x",
        sha256="a" * 64,
        created_at=datetime(2026, 5, 26, tzinfo=UTC),
        created_by="harness",
    )
    # Prefixed form is rejected — caller (FileArtifactStore) must strip it.
    with pytest.raises(ValidationError):
        ArtifactRef(
            id="a1b2c3d4",
            kind="log",
            uri="file:///x",
            sha256="sha256:" + "a" * 64,
            created_at=datetime(2026, 5, 26, tzinfo=UTC),
            created_by="harness",
        )


def test_artifact_kind_is_literal_not_enum() -> None:
    """ArtifactKind discriminator MUST be typing.Literal[...], not enum.Enum."""
    from typing import Literal

    from molexp.harness.schemas import artifact as artifact_mod

    assert get_origin(artifact_mod.ArtifactKind) is Literal
    # No enum.Enum classes defined in the module.
    import enum

    for name in dir(artifact_mod):
        obj = getattr(artifact_mod, name)
        if isinstance(obj, type) and issubclass(obj, enum.Enum) and obj is not enum.Enum:
            pytest.fail(f"schemas/artifact.py must not define enum: {name}")


def test_artifact_kind_full_19_value_set() -> None:
    """ArtifactKind ships the full 19-kind enum from harness-goal.md §4.1.

    Phase 7 additively widens to add "validation_report" — assert the original
    19 are a subset, not equality (additive widening preserves backward compat).
    """
    from molexp.harness.schemas.artifact import ArtifactKind

    expected_baseline = {
        "user_plan",
        "experiment_report",
        "workflow_ir",
        "bound_workflow",
        "test_spec",
        "execution_plan",
        "execution_result",
        "test_result",
        "analysis_result",
        "final_report",
        "audit_report",
        "stdout",
        "stderr",
        "log",
        "input_file",
        "output_file",
        "plot",
        "dataset",
        "checkpoint",
    }
    assert expected_baseline <= set(get_args(ArtifactKind))
