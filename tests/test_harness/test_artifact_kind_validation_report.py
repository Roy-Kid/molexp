"""Tests for ArtifactKind widening to include "validation_report" (Phase 7).

Additive Literal extension — 19 existing kinds keep validating, +1 new kind.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import get_args

import pytest
from pydantic import ValidationError


def _build_ref(*, kind: str):
    from molexp.harness.schemas.artifact import ArtifactRef

    return ArtifactRef(
        id="art01234",
        kind=kind,  # type: ignore[arg-type]
        uri="file:///tmp/x",
        sha256="0" * 64,
        created_at=datetime(2026, 5, 26, tzinfo=UTC),
        created_by="harness",
    )


def test_validation_report_kind_accepted() -> None:
    ref = _build_ref(kind="validation_report")
    assert ref.kind == "validation_report"


def test_artifact_kind_has_exactly_twenty_values() -> None:
    from molexp.harness.schemas.artifact import ArtifactKind

    assert len(set(get_args(ArtifactKind))) == 20


def test_existing_nineteen_kinds_still_validate() -> None:
    """Regression: every pre-Phase-7 ArtifactKind value still constructs."""
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


def test_unknown_kind_still_rejected() -> None:
    with pytest.raises(ValidationError):
        _build_ref(kind="not_a_real_kind")
