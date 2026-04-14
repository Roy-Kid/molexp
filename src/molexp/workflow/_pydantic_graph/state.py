"""Internal pydantic-graph state and deps types.

Users never import these directly — they use the public StepContext/WorkflowResult API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class WorkflowState:
    """Shared mutable state threaded through all workflow steps via pydantic-graph.

    step_outputs maps step_name → output value as steps complete.
    The final entry in step_outputs is the workflow result when done.
    """

    step_outputs: dict[str, Any] = field(default_factory=dict)
    failed: bool = False
    error: str | None = None

    def record(self, step_name: str, output: Any) -> "WorkflowState":
        """Return a new state with the given step output recorded."""
        return WorkflowState(
            step_outputs={**self.step_outputs, step_name: output},
            failed=self.failed,
            error=self.error,
        )

    def fail(self, step_name: str, exc: Exception) -> "WorkflowState":
        """Return a new state marked as failed."""
        return WorkflowState(
            step_outputs=self.step_outputs,
            failed=True,
            error=f"Step '{step_name}' failed: {exc}",
        )


    def _sync_from(self, other: "WorkflowState") -> None:
        """Update this state in-place from *other*.

        pydantic-graph holds a reference to the state object and snapshots it
        after each node.  We MUST mutate the original reference so the
        snapshot reflects the latest outputs.  This method centralises that
        necessary mutation in one place.
        """
        self.step_outputs = other.step_outputs
        self.failed = other.failed
        self.error = other.error


@dataclass
class WorkflowDeps:
    """Dependencies injected into every pydantic-graph node.

    run: The molexp Run associated with this execution (may be None).
    run_context: The active RunContext associated with this execution (may be None).
    config: The active :class:`~molexp.config.ProfileConfig` (may be None).
    user_deps: Application-level deps forwarded from the caller.
    """

    run: Any = None
    run_context: Any = None
    config: Any = None  # molexp.config.ProfileConfig | None
    user_deps: Any = None
