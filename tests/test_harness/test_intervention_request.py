"""RED tests for the plan-emergent-06 intervention data contract.

Pins the frozen intervention payload + the blocked-task error the realization
phase raises when a task cannot be self-repaired to green:

- ``BlockedTask`` / ``InterventionRequest`` are frozen pydantic (assignment
  raises), carry the declared fields (no scope tag — that is 07's concern),
  and both import from ``molexp.harness.schemas``.
- ``TaskRealizationBlockedError`` subclasses ``StageExecutionError`` (so
  existing ``except StageExecutionError`` sites still stop the pipeline) and
  carries ``request_ref`` + ``blocked_task_ids``.
- ``"intervention_request"`` is a well-known artifact kind.

The production modules do not exist yet: an ``ImportError`` at collection time
is the valid RED signal.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

import molexp.harness.errors as errors_mod
import molexp.harness.schemas as schemas_mod
from molexp.harness.errors import StageExecutionError, TaskRealizationBlockedError
from molexp.harness.schemas import (
    WELL_KNOWN_ARTIFACT_KINDS,
    BlockedTask,
    InterventionRequest,
    PlanArtifactRef,
)


def _ref(kind: str = "intervention_request") -> PlanArtifactRef:
    """A structurally valid ``PlanArtifactRef`` (bare-hex sha256)."""
    return PlanArtifactRef(
        id="art-intervention-1",
        kind=kind,
        uri="mem://art-intervention-1",
        sha256="a" * 64,
        created_at=datetime(2026, 7, 22, tzinfo=UTC),
        created_by="test",
    )


def _blocked(task_id: str = "b-beta") -> BlockedTask:
    return BlockedTask(
        task_id=task_id,
        ir_task_id="beta",
        slug="beta",
        attempts=3,
        error="AssertionError: boom",
        question="task beta failed after 3 attempts; how should it be written?",
    )


class TestBlockedTask:
    def test_fields_and_error_artifact_id_defaults_none(self) -> None:
        bt = BlockedTask(
            task_id="b-beta",
            ir_task_id="beta",
            slug="beta",
            attempts=3,
            error="AssertionError: boom",
            question="how should task beta be written?",
        )
        assert bt.task_id == "b-beta"
        assert bt.ir_task_id == "beta"
        assert bt.slug == "beta"
        assert bt.attempts == 3
        assert bt.error == "AssertionError: boom"
        assert bt.question == "how should task beta be written?"
        # error_artifact_id is optional and defaults to None.
        assert bt.error_artifact_id is None

    def test_error_artifact_id_accepts_a_ref_id(self) -> None:
        bt = BlockedTask(
            task_id="b-beta",
            ir_task_id="beta",
            slug="beta",
            attempts=2,
            error="boom",
            error_artifact_id="art-failure-9",
            question="q",
        )
        assert bt.error_artifact_id == "art-failure-9"

    def test_rejects_assignment(self) -> None:
        bt = _blocked()
        with pytest.raises(ValidationError):
            bt.attempts = 9  # type: ignore[misc]


class TestInterventionRequest:
    def test_fields_and_blocked_tasks_is_a_tuple(self) -> None:
        created = datetime(2026, 7, 22, 12, 0, tzinfo=UTC)
        req = InterventionRequest(
            id="ivr-1",
            bound_workflow_id="bw-1",
            blocked_tasks=(_blocked("b-beta"),),
            reason="1 task could not be realized",
            created_at=created,
        )
        assert req.id == "ivr-1"
        assert req.bound_workflow_id == "bw-1"
        assert isinstance(req.blocked_tasks, tuple)
        assert req.blocked_tasks[0].task_id == "b-beta"
        assert req.reason == "1 task could not be realized"
        assert req.created_at == created

    def test_rejects_assignment(self) -> None:
        req = InterventionRequest(
            id="ivr-1",
            bound_workflow_id="bw-1",
            blocked_tasks=(_blocked(),),
            reason="blocked",
            created_at=datetime(2026, 7, 22, tzinfo=UTC),
        )
        with pytest.raises(ValidationError):
            req.reason = "changed"  # type: ignore[misc]

    def test_both_schemas_exported_from_harness_schemas(self) -> None:
        # ac-001: both schema names import from molexp.harness.schemas.
        assert "BlockedTask" in schemas_mod.__all__
        assert "InterventionRequest" in schemas_mod.__all__
        assert schemas_mod.BlockedTask is BlockedTask
        assert schemas_mod.InterventionRequest is InterventionRequest


class TestTaskRealizationBlockedError:
    def test_is_stage_execution_error_subclass(self) -> None:
        assert issubclass(TaskRealizationBlockedError, StageExecutionError)

    def test_carries_request_ref_and_blocked_task_ids(self) -> None:
        ref = _ref()
        err = TaskRealizationBlockedError(ref, ["b-beta"])
        assert isinstance(err, StageExecutionError)
        assert err.request_ref is ref
        assert err.blocked_task_ids == ["b-beta"]

    def test_exported_in_errors_all(self) -> None:
        assert "TaskRealizationBlockedError" in errors_mod.__all__


class TestWellKnownArtifactKinds:
    def test_intervention_request_kind_present(self) -> None:
        assert "intervention_request" in WELL_KNOWN_ARTIFACT_KINDS
