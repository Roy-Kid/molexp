"""Tests for validate_provenance (Phase 5).

Three codes + two clean baselines.

Codes:
- artifact_not_found (error): get_ref raises ArtifactNotFoundError
- unreachable_root (error): trace_backward non-empty but no root_kind ancestor
- orphan_artifact (warning): empty trace_backward AND artifact.kind != root_kind

Clean baselines:
- artifact_is_root_clean: empty trace + kind == root_kind → passed=True
- chain_reaches_root_clean: multi-edge chain ending at root_kind → passed=True
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture()
def stores(tmp_path: Path):
    from molexp.harness.store.file_artifact_store import FileArtifactStore
    from molexp.harness.store.sqlite_lineage_store import SQLiteArtifactLineageStore

    artifacts = FileArtifactStore(root=tmp_path / "artifacts")
    provenance = SQLiteArtifactLineageStore(
        path=tmp_path / "events.sqlite", artifact_store=artifacts
    )
    return artifacts, provenance


def _codes(report) -> list[str]:
    return [v.code for v in report.violations]


# ---------------------------------------------------------- error codes


def test_artifact_not_found(stores) -> None:
    """Validator must catch ArtifactNotFoundError; never let it bubble."""
    from molexp.harness.validators.provenance import ProvenanceValidator

    artifact_store, lineage_store = stores
    # No artifact registered.
    report = ProvenanceValidator.validate(
        "ghost-id",
        artifact_store=artifact_store,
        lineage_store=lineage_store,
    )
    assert "artifact_not_found" in _codes(report)
    assert report.passed is False
    assert report.target_kind == "provenance"
    assert report.target_id == "ghost-id"


def test_unreachable_root(stores) -> None:
    """Chain B→C with no user_plan ancestor."""
    from molexp.harness.validators.provenance import ProvenanceValidator

    artifact_store, lineage_store = stores
    b = artifact_store.put_json(
        kind="experiment_report", obj={"b": 1}, created_by="x", parent_ids=[]
    )
    c = artifact_store.put_json(kind="workflow_ir", obj={"c": 1}, created_by="x", parent_ids=[b.id])
    lineage_store.add_edge(parent_id=b.id, child_id=c.id)
    report = ProvenanceValidator.validate(
        c.id,
        artifact_store=artifact_store,
        lineage_store=lineage_store,
    )
    assert "unreachable_root" in _codes(report)


def test_orphan_artifact_warning(stores) -> None:
    """Singleton non-root artifact → warning (could be partial run)."""
    from molexp.harness.validators.provenance import ProvenanceValidator

    artifact_store, lineage_store = stores
    ref = artifact_store.put_json(
        kind="experiment_report", obj={"x": 1}, created_by="x", parent_ids=[]
    )
    report = ProvenanceValidator.validate(
        ref.id,
        artifact_store=artifact_store,
        lineage_store=lineage_store,
    )
    matches = [v for v in report.violations if v.code == "orphan_artifact"]
    assert matches, "expected orphan_artifact warning"
    assert matches[0].severity == "warning"
    if all(v.severity == "warning" for v in report.violations):
        assert report.passed is True


# ---------------------------------------------------------- clean baselines


def test_artifact_is_root_clean(stores) -> None:
    """Artifact whose own kind == root_kind with empty backward trace → clean."""
    from molexp.harness.validators.provenance import ProvenanceValidator

    artifact_store, lineage_store = stores
    ref = artifact_store.put_text(
        kind="user_plan", text="simulate water", created_by="user", parent_ids=[]
    )
    report = ProvenanceValidator.validate(
        ref.id,
        artifact_store=artifact_store,
        lineage_store=lineage_store,
    )
    assert report.passed is True
    assert report.violations == []


def test_chain_reaches_root_clean(stores) -> None:
    """Multi-edge chain root_kind → ... → leaf returns passed=True."""
    from molexp.harness.validators.provenance import ProvenanceValidator

    artifact_store, lineage_store = stores
    user_plan = artifact_store.put_text(
        kind="user_plan", text="simulate water", created_by="user", parent_ids=[]
    )
    report = artifact_store.put_json(
        kind="experiment_report",
        obj={"x": 1},
        created_by="harness",
        parent_ids=[user_plan.id],
    )
    ir = artifact_store.put_json(
        kind="workflow_ir", obj={"y": 1}, created_by="harness", parent_ids=[report.id]
    )
    lineage_store.add_edge(parent_id=user_plan.id, child_id=report.id)
    lineage_store.add_edge(parent_id=report.id, child_id=ir.id)
    validation = ProvenanceValidator.validate(
        ir.id,
        artifact_store=artifact_store,
        lineage_store=lineage_store,
    )
    assert validation.passed is True
    assert validation.violations == []


def test_root_kind_override(stores) -> None:
    """root_kind kwarg lets caller assert lineage to an intermediate kind."""
    from molexp.harness.validators.provenance import ProvenanceValidator

    artifact_store, lineage_store = stores
    user_plan = artifact_store.put_text(
        kind="user_plan", text="x", created_by="user", parent_ids=[]
    )
    report = artifact_store.put_json(
        kind="experiment_report",
        obj={"x": 1},
        created_by="harness",
        parent_ids=[user_plan.id],
    )
    ir = artifact_store.put_json(
        kind="workflow_ir", obj={"y": 1}, created_by="harness", parent_ids=[report.id]
    )
    lineage_store.add_edge(parent_id=user_plan.id, child_id=report.id)
    lineage_store.add_edge(parent_id=report.id, child_id=ir.id)
    # Assert ir traces back to some experiment_report (not necessarily user_plan).
    result = ProvenanceValidator.validate(
        ir.id,
        artifact_store=artifact_store,
        lineage_store=lineage_store,
        root_kind="experiment_report",
    )
    assert result.passed is True


# ---------------------------------------------------------- signature


def test_validate_provenance_signature_and_import(stores) -> None:
    from molexp.harness import ProvenanceValidator as top
    from molexp.harness.schemas.validation import ValidationReport
    from molexp.harness.validators import ProvenanceValidator as via_pkg
    from molexp.harness.validators.provenance import ProvenanceValidator as via_mod

    assert top is via_pkg is via_mod
    artifact_store, lineage_store = stores
    ref = artifact_store.put_text(kind="user_plan", text="x", created_by="user", parent_ids=[])
    report = top.validate(ref.id, artifact_store=artifact_store, lineage_store=lineage_store)
    assert isinstance(report, ValidationReport)
