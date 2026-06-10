"""Tests for the ``ExecutionResult`` schema (spec ``harness-run-mode-01-substrate``, T01).

RED before implementation: ``ExecutionResult`` does not exist yet, so the
module-level import fails at collection. After GREEN these assert the
frozen wire shape: required execution identity + timing fields, a
two-value ``status`` Literal, and the ``TestResult``-aligned
``ArtifactRef | None`` stdout/stderr channel.

Timestamps are fixed constants — no wall clock in assertions.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from molexp.harness import ArtifactRef, ExecutionResult

_STARTED = datetime(2026, 6, 10, 12, 0, 0, tzinfo=UTC)
_ENDED = datetime(2026, 6, 10, 12, 5, 0, tzinfo=UTC)


def _artifact_ref(kind: str) -> ArtifactRef:
    return ArtifactRef(
        id=f"art-{kind}",
        kind=kind,
        uri=f"file:///artifacts/{kind}",
        sha256="ab" * 32,
        created_at=_STARTED,
        created_by="seed",
    )


def test_execution_result_minimal_fields_and_defaults() -> None:
    result = ExecutionResult(
        id="exec-result-1",
        bound_workflow_id="bw-x",
        status="succeeded",
        exit_code=0,
        started_at=_STARTED,
        ended_at=_ENDED,
    )
    assert result.id == "exec-result-1"
    assert result.bound_workflow_id == "bw-x"
    assert result.status == "succeeded"
    assert result.exit_code == 0
    assert result.started_at == _STARTED
    assert result.ended_at == _ENDED
    assert result.outputs == {}
    assert result.output_artifacts == []
    assert result.stdout is None
    assert result.stderr is None
    assert result.metadata == {}


def test_execution_result_full_round_trip() -> None:
    out_ref = _artifact_ref("output_file")
    stdout_ref = _artifact_ref("stdout")
    stderr_ref = _artifact_ref("stderr")
    result = ExecutionResult(
        id="exec-result-2",
        bound_workflow_id="bw-x",
        status="succeeded",
        exit_code=0,
        started_at=_STARTED,
        ended_at=_ENDED,
        outputs={"n_frames": 100, "label": "ok"},
        output_artifacts=[out_ref],
        stdout=stdout_ref,
        stderr=stderr_ref,
        metadata={"missing_outputs": ""},
    )
    assert result.outputs == {"n_frames": 100, "label": "ok"}
    assert result.output_artifacts == [out_ref]
    assert result.stdout == stdout_ref
    assert result.stderr == stderr_ref
    assert result.metadata == {"missing_outputs": ""}


def test_execution_result_accepts_failed_status() -> None:
    result = ExecutionResult(
        id="exec-result-3",
        bound_workflow_id="bw-x",
        status="failed",
        exit_code=1,
        started_at=_STARTED,
        ended_at=_ENDED,
    )
    assert result.status == "failed"
    assert result.exit_code == 1


def test_execution_result_rejects_unknown_status() -> None:
    with pytest.raises(ValidationError):
        ExecutionResult(
            id="exec-result-4",
            bound_workflow_id="bw-x",
            status="exploded",  # type: ignore[arg-type]
            exit_code=2,
            started_at=_STARTED,
            ended_at=_ENDED,
        )


def test_execution_result_is_frozen() -> None:
    result = ExecutionResult(
        id="exec-result-5",
        bound_workflow_id="bw-x",
        status="succeeded",
        exit_code=0,
        started_at=_STARTED,
        ended_at=_ENDED,
    )
    with pytest.raises(ValidationError):
        result.exit_code = 1  # type: ignore[misc]


def test_execution_result_reexported_from_harness() -> None:
    import molexp.harness as h
    from molexp.harness.schemas import ExecutionResult as FromSchemas
    from molexp.harness.schemas.execution_result import ExecutionResult as Canonical

    assert h.ExecutionResult is Canonical
    assert FromSchemas is Canonical
    assert ExecutionResult is Canonical
