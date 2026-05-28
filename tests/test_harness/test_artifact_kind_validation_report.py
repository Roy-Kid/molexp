"""Tests for ``WELL_KNOWN_ARTIFACT_KINDS`` membership + open-string contract.

Phase 7 added ``validation_report``; the harness-as-mode-substrate-01 spec
opened ``ArtifactKind`` to ``str``, moving the closed enumeration into the
:data:`WELL_KNOWN_ARTIFACT_KINDS` documentation constant. This file pins
the well-known membership (existing baseline + ``validation_report``) and
the new "arbitrary string accepted" contract.
"""

from __future__ import annotations

from datetime import UTC, datetime


def _build_ref(*, kind: str):
    from molexp.harness.schemas.artifact import ArtifactRef

    return ArtifactRef(
        id="art01234",
        kind=kind,
        uri="file:///tmp/x",
        sha256="0" * 64,
        created_at=datetime(2026, 5, 26, tzinfo=UTC),
        created_by="harness",
    )


def test_validation_report_kind_accepted() -> None:
    ref = _build_ref(kind="validation_report")
    assert ref.kind == "validation_report"


def test_well_known_artifact_kinds_has_at_least_twenty_values() -> None:
    from molexp.harness.schemas.artifact import WELL_KNOWN_ARTIFACT_KINDS

    assert len(set(WELL_KNOWN_ARTIFACT_KINDS)) >= 20


def test_existing_nineteen_kinds_still_validate() -> None:
    """Regression: every pre-Phase-7 baseline kind still constructs."""
    pre_phase7 = [
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
    ]
    for kind in pre_phase7:
        _build_ref(kind=kind)


def test_unknown_kind_now_accepted() -> None:
    """Under the open-``str`` contract, agent-layer kinds construct fine."""
    ref = _build_ref(kind="not_a_real_kind")
    assert ref.kind == "not_a_real_kind"
