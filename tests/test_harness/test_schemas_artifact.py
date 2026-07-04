"""Tests for PlanArtifactRef + ArtifactKind (Phase 1 schema layer).

Locks the two non-obvious wire contracts:
- ArtifactKind is an open `str` alias; arbitrary (agent-registered) kinds accepted
- sha256 is bare hex (no "sha256:" prefix) — callers must strip the workspace prefix
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError


def test_artifact_ref_accepts_arbitrary_string_kind() -> None:
    """PlanArtifactRef accepts arbitrary string kinds under the open `str` contract.

    Spec ac-004: agent-layer modes register kinds like "intent_spec",
    "plan_graph", "preflight_report", … without round-tripping through the
    harness schema module.
    """
    from molexp.harness.schemas.artifact import PlanArtifactRef

    ref = PlanArtifactRef(
        id="a1b2c3d4",
        kind="intent_spec",
        uri="file:///x",
        sha256="0" * 64,
        created_at=datetime(2026, 5, 26, tzinfo=UTC),
        created_by="harness",
    )
    assert ref.kind == "intent_spec"
    # JSON round-trip preserves the custom kind value.
    rehydrated = PlanArtifactRef.model_validate_json(ref.model_dump_json())
    assert rehydrated.kind == "intent_spec"
    assert rehydrated == ref


def test_artifact_ref_sha256_must_be_bare_hex() -> None:
    """PlanArtifactRef.sha256 stores bare hex, not the 'sha256:<hex>' prefixed form."""
    from molexp.harness.schemas.artifact import PlanArtifactRef

    # Bare hex of length 64 is accepted.
    PlanArtifactRef(
        id="a1b2c3d4",
        kind="log",
        uri="file:///x",
        sha256="a" * 64,
        created_at=datetime(2026, 5, 26, tzinfo=UTC),
        created_by="harness",
    )
    # Prefixed form is rejected — caller (FileArtifactStore) must strip it.
    with pytest.raises(ValidationError):
        PlanArtifactRef(
            id="a1b2c3d4",
            kind="log",
            uri="file:///x",
            sha256="sha256:" + "a" * 64,
            created_at=datetime(2026, 5, 26, tzinfo=UTC),
            created_by="harness",
        )
