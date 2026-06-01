"""``ModeResult`` schema tests (spec plan-mode-revival-01, ac-006).

``ModeResult`` is the frozen pydantic result returned by ``Mode.run``:
``mode_name``, ``run_id``, ``execution_id``, ``stage_artifacts``,
``final_artifact``, ``validation_reports``. It must be exported from both
``molexp.harness.schemas`` and ``molexp.harness``.

RED until ``src/molexp/harness/schemas/mode_result.py`` exists and is wired
into the two ``__init__`` export lists.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from molexp.harness.schemas import ArtifactRef, ValidationReport


def _ref(artifact_id: str, kind: str) -> ArtifactRef:
    return ArtifactRef(
        id=artifact_id,
        kind=kind,
        uri=f"mem://{artifact_id}",
        sha256="ab" * 32,
        created_at=datetime.now(UTC),
        created_by="test",
        parent_ids=[],
    )


def test_mode_result_importable_from_schemas() -> None:
    """ac-006: exported from molexp.harness.schemas."""
    from molexp.harness.schemas import ModeResult  # noqa: F401


def test_mode_result_importable_from_harness_root() -> None:
    """ac-006: exported from molexp.harness."""
    from molexp.harness import ModeResult  # noqa: F401


def test_mode_result_is_frozen() -> None:
    """ac-006: ModeResult is an immutable (frozen) pydantic model."""
    from pydantic import ValidationError

    from molexp.harness.schemas import ModeResult

    result = ModeResult(
        mode_name="demo",
        run_id="run-1",
        execution_id="exec-1",
        stage_artifacts=(),
        final_artifact=None,
        validation_reports=(),
    )
    with pytest.raises(ValidationError):
        result.mode_name = "mutated"  # type: ignore[misc]


def test_mode_result_carries_per_stage_artifacts_and_final() -> None:
    """ac-006: stage_artifacts + final_artifact + validation_reports fields populate."""
    from molexp.harness.schemas import ModeResult

    a = _ref("art-a", "user_plan")
    b = _ref("art-b", "experiment_report")
    report = ValidationReport(passed=True, violations=[], target_kind="workflow_ir", target_id="x")

    result = ModeResult(
        mode_name="demo",
        run_id="run-1",
        execution_id="exec-1",
        stage_artifacts=(a, b),
        final_artifact=b,
        validation_reports=(report,),
    )

    assert result.mode_name == "demo"
    assert result.run_id == "run-1"
    assert result.execution_id == "exec-1"
    assert result.stage_artifacts == (a, b)
    assert result.final_artifact == b
    assert result.validation_reports == (report,)
