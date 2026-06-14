"""``ExecutionResult`` — outcome of one harness-controlled workflow execution.

The product of :class:`~molexp.harness.stages.execute_workflow.ExecuteWorkflow`:
the materialized driver ran in an executor subprocess; this record captures
the exit status, the parsed ``outputs.json`` payload, and the artifact refs
the executor collected. Field naming aligns with
:class:`~molexp.harness.schemas.command.CommandResult` and
:class:`~molexp.harness.schemas.test_spec.TestResult`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from molexp.harness.schemas.artifact import ArtifactRef

__all__ = ["ExecutionResult"]


class ExecutionResult(BaseModel):
    """Outcome of executing a generated workflow through an :class:`Executor`.

    Attributes:
        id: Unique id for this execution record.
        bound_workflow_id: The ``BoundWorkflow`` this execution realizes.
        status: ``"succeeded"`` iff the driver subprocess exited 0.
        exit_code: Raw subprocess exit code.
        started_at: Subprocess start time (from ``CommandResult``).
        ended_at: Subprocess end time (from ``CommandResult``).
        outputs: Task-name → output mapping parsed from ``outputs.json``;
            empty when the driver produced none or it was unparseable.
        output_artifacts: Files the executor collected post-run.
        stdout: Captured stdout artifact, if any.
        stderr: Captured stderr artifact, if any.
        metadata: Executor metadata (timeout flags, missing outputs, …).
    """

    model_config = ConfigDict(frozen=True)

    id: str
    bound_workflow_id: str
    status: Literal["succeeded", "failed"]
    exit_code: int
    started_at: datetime
    ended_at: datetime
    outputs: dict[str, Any] = Field(default_factory=dict)
    output_artifacts: list[ArtifactRef] = Field(default_factory=list)
    stdout: ArtifactRef | None = None
    stderr: ArtifactRef | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
