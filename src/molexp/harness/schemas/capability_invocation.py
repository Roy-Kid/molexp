"""``CapabilityInvocationResult`` — outcome of one direct capability invocation.

The product of
:class:`~molexp.harness.stages.invoke_capability.InvokeCapability`: a single
:class:`~molexp.harness.schemas.capability.ToolCapability` ran through an
:class:`~molexp.harness.executors.Executor` subprocess; this record captures the
exit status, the parsed ``result.json`` payload, and the artifact refs the
executor collected. Field naming mirrors
:class:`~molexp.harness.schemas.execution_result.ExecutionResult` (which records
a workflow execution) and
:class:`~molexp.harness.schemas.command.CommandResult`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from molexp.harness.schemas.artifact import ArtifactRef

__all__ = ["CapabilityInvocationResult"]


class CapabilityInvocationResult(BaseModel):
    """Outcome of invoking one capability through an :class:`Executor`.

    Attributes:
        id: Unique id for this invocation record.
        capability_id: The ``ToolCapability`` id that was invoked.
        status: ``"succeeded"`` iff the runner subprocess exited 0.
        exit_code: Raw subprocess exit code.
        started_at: Subprocess start time (from ``CommandResult``).
        ended_at: Subprocess end time (from ``CommandResult``).
        outputs: The callable's return, parsed from ``result.json`` (a
            ``{"return": <value>}`` mapping); empty when the runner produced
            none (e.g. a ``DryRunExecutor`` no-op) or it was unparseable.
        output_artifacts: Files the executor collected post-run.
        stdout: Captured stdout artifact, if any.
        stderr: Captured stderr artifact, if any.
        metadata: Executor metadata (``{"executor": …}``, timeout flags, …).
    """

    model_config = ConfigDict(frozen=True)

    id: str
    capability_id: str
    status: Literal["succeeded", "failed"]
    exit_code: int
    started_at: datetime
    ended_at: datetime
    outputs: dict[str, Any] = Field(default_factory=dict)
    output_artifacts: list[ArtifactRef] = Field(default_factory=list)
    stdout: ArtifactRef | None = None
    stderr: ArtifactRef | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
