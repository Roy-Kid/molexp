"""Core result types for molexp workflow API.

Context types live in ``workflow.context``; this module holds only
result / execution-handle types and shared type variables.
"""

from __future__ import annotations

from typing import Any, TypeVar

# ── Shared type variables (re-exported by __init__) ─────────────────────────

StateT = TypeVar("StateT")
DepsT = TypeVar("DepsT")
InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


# ── Workflow execution results ──────────────────────────────────────────────


class WorkflowResult:
    """Result of a completed workflow execution.

    Attributes:
        status: ``"completed"`` | ``"failed"`` | ``"cancelled"``
        outputs: Mapping of task name to task output.
        run_id: Associated workspace Run ID, if any.
        execution_id: Opaque ID for resumption support.
    """

    def __init__(
        self,
        status: str,
        outputs: dict[str, Any],
        run_id: str | None = None,
        execution_id: str | None = None,
    ) -> None:
        self.status = status
        self.outputs = outputs
        self.run_id = run_id
        self.execution_id = execution_id

    def __repr__(self) -> str:
        return (
            f"WorkflowResult(status={self.status!r}, "
            f"tasks={list(self.outputs.keys())})"
        )


class WorkflowExecution:
    """Handle for a running workflow.

    Returned by ``WorkflowSpec.start()`` for async control.
    """

    def __init__(
        self,
        execution_id: str,
        workflow_id: str,
        run_id: str | None = None,
    ) -> None:
        self.execution_id = execution_id
        self.workflow_id = workflow_id
        self.run_id = run_id

    async def wait(self) -> WorkflowResult:
        """Block until the workflow completes."""
        raise NotImplementedError

    async def cancel(self) -> None:
        """Cancel the running workflow."""
        raise NotImplementedError
