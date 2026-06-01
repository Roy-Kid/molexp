"""Tests for ArtifactRef + ArtifactKind (Phase 1 schema layer).

Locks the wire format and validation contract per spec §4.1:
- frozen pydantic round-trip
- ArtifactKind is an open `str` alias; well-known values listed in `WELL_KNOWN_ARTIFACT_KINDS`
- arbitrary string kinds accepted; empty string still rejected
- sha256 is bare hex (no "sha256:" prefix)
"""

from __future__ import annotations

from datetime import UTC, datetime

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


def test_artifact_ref_accepts_arbitrary_string_kind() -> None:
    """ArtifactRef accepts arbitrary string kinds under the open `str` contract.

    Spec ac-004: agent-layer modes register kinds like "intent_spec",
    "plan_graph", "preflight_report", … without round-tripping through the
    harness schema module.
    """
    from molexp.harness.schemas.artifact import ArtifactRef

    ref = ArtifactRef(
        id="a1b2c3d4",
        kind="intent_spec",
        uri="file:///x",
        sha256="0" * 64,
        created_at=datetime(2026, 5, 26, tzinfo=UTC),
        created_by="harness",
    )
    assert ref.kind == "intent_spec"
    # JSON round-trip preserves the custom kind value.
    rehydrated = ArtifactRef.model_validate_json(ref.model_dump_json())
    assert rehydrated.kind == "intent_spec"
    assert rehydrated == ref


def test_artifact_ref_rejects_empty_kind() -> None:
    """ArtifactRef rejects kind="" — pydantic min_length=1 constraint pins the
    edge case from the spec's Testing strategy.
    """
    from molexp.harness.schemas.artifact import ArtifactRef

    with pytest.raises(ValidationError):
        ArtifactRef(
            id="a1b2c3d4",
            kind="",
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


def test_artifact_kind_is_str_alias() -> None:
    """ArtifactKind MUST be the `str` builtin alias, not Literal/Enum.

    Spec ac-001: `ArtifactKind is str` in the artifact schema module.
    """
    from molexp.harness.schemas import artifact as artifact_mod

    assert artifact_mod.ArtifactKind is str
    # No enum.Enum classes defined in the module.
    import enum

    for name in dir(artifact_mod):
        obj = getattr(artifact_mod, name)
        if isinstance(obj, type) and issubclass(obj, enum.Enum) and obj is not enum.Enum:
            pytest.fail(f"schemas/artifact.py must not define enum: {name}")


def test_well_known_artifact_kinds_contents() -> None:
    """`WELL_KNOWN_ARTIFACT_KINDS` ships the 20 baseline kinds as a string tuple.

    Spec ac-002: tuple[str, ...] of length >= 20 containing every prior Literal
    member plus the 20th value "validation_report".
    """
    from molexp.harness.schemas.artifact import WELL_KNOWN_ARTIFACT_KINDS

    assert isinstance(WELL_KNOWN_ARTIFACT_KINDS, tuple)
    assert all(isinstance(k, str) for k in WELL_KNOWN_ARTIFACT_KINDS)
    assert len(WELL_KNOWN_ARTIFACT_KINDS) >= 20

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
    well_known_set = set(WELL_KNOWN_ARTIFACT_KINDS)
    assert expected_baseline <= well_known_set
    assert "validation_report" in well_known_set


def test_well_known_artifact_kinds_reexported_through_top_level() -> None:
    """`WELL_KNOWN_ARTIFACT_KINDS` re-exports through schemas and harness top-level.

    Spec ac-003: `from molexp.harness import WELL_KNOWN_ARTIFACT_KINDS` succeeds;
    same tuple object as in artifact.py; listed in both `__all__`s.
    """
    import molexp.harness as harness_top
    import molexp.harness.schemas as schemas_top
    from molexp.harness.schemas import artifact as artifact_mod

    assert harness_top.WELL_KNOWN_ARTIFACT_KINDS is artifact_mod.WELL_KNOWN_ARTIFACT_KINDS
    assert schemas_top.WELL_KNOWN_ARTIFACT_KINDS is artifact_mod.WELL_KNOWN_ARTIFACT_KINDS
    assert "WELL_KNOWN_ARTIFACT_KINDS" in harness_top.__all__
    assert "WELL_KNOWN_ARTIFACT_KINDS" in schemas_top.__all__
