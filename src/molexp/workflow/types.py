"""Core result types for molexp workflow API.

Context types live in ``workflow.context``; this module holds only
result / execution-handle types and shared type variables.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeVar

# ── Shared type variables (re-exported by __init__) ─────────────────────────

StateT = TypeVar("StateT")
DepsT = TypeVar("DepsT")
InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


# ── Route selection sentinels ───────────────────────────────────────────────


@dataclass(frozen=True)
class Next:
    """Runtime route selector.

    Returned by a task body to pick a declared outgoing route by label. Picks
    one of the ``routes={label: target}`` entries declared on the task; does
    NOT jump to a task named ``label``.

    Spec: §6 ``Next`` is a route selector, not a node jumper.
    """

    label: str


@dataclass(frozen=True)
class End:
    """Workflow terminator (frame-scoped).

    Returned by a task body to terminate the workflow after the current frame
    finishes. Bypasses ``routes`` entirely. Same-frontier sibling tasks still
    run to completion and their outputs are recorded (§8 frame-scoped End).

    This is molexp's own sentinel — distinct from ``pydantic_graph.End``.
    """


# ── Edge sum types (spec §3, §7) ────────────────────────────────────────────


@dataclass(frozen=True)
class UnconditionalEdges:
    """A node's outgoing control edge set, all unconditional.

    Empty ``targets`` = terminal node (0 out edges). Non-empty = either a
    single forward edge or a fan-out to multiple successors.
    """

    targets: tuple[str, ...]


@dataclass(frozen=True)
class BranchEdges:
    """A node's outgoing control edge set, label-routed.

    The task body MUST return ``Next(label)`` selecting one of the declared
    labels. Mixing branch + unconditional on the same node is rejected at
    compile time (``EdgeShapeError``).
    """

    routes: Mapping[str, str]  # label → target task name


# Sum type for a task's compiled out-edge set.
OutEdges = UnconditionalEdges | BranchEdges


# ── Workflow-level errors ───────────────────────────────────────────────────


class WorkflowError(Exception):
    """Base class for all molexp workflow errors."""


class CycleError(WorkflowError):
    """`depends_on` graph contains a cycle. Use a control edge instead."""


class EdgeShapeError(WorkflowError):
    """A node mixes unconditional + branch out-edges (illegal — pick one form)."""


class EntryAmbiguousError(WorkflowError):
    """A workflow with explicit control edges did not declare ``wf.entry(...)``."""


class UnknownTaskError(WorkflowError):
    """An entry / control / branch declaration references an unregistered task."""


class UnreachableTaskError(WorkflowError):
    """A registered task is not reachable from any entry through control edges."""


class UnknownRouteError(WorkflowError):
    """``Next("label")`` returned a label that's not in the task's declared routes."""


class MissingRouteError(WorkflowError):
    """A branch-shaped node returned a plain Output without a ``Next`` or ``End``."""


class WorkflowDeadlockError(WorkflowError):
    """Frontier exhausted but pending targets remain with unsatisfied data deps."""


class SanityCheckFailed(Exception):
    """A ``Workflow.sanity_check(..., on_fail='halt')`` predicate returned false.

    Subclasses :class:`Exception` rather than :class:`WorkflowError` because
    sanity failure is a *runtime data condition*, not a workflow-graph
    structural error.  The runtime catches it as a failed-task exception
    and surfaces a ``WorkflowResult(status='failed')``, while bona-fide
    :class:`WorkflowError` subclasses (cycle / edge / route bugs) keep
    propagating to the caller.

    ``sanity_events`` carries a snapshot of the sanity event log up to and
    including the offending event so the runtime can preserve it on the
    failed :class:`WorkflowResult` even when the WorkflowState is unwound.
    """

    def __init__(
        self,
        task: str,
        message: str | None = None,
        sanity_events: list[dict[str, Any]] | None = None,
    ) -> None:
        self.task = task
        self.sanity_events = list(sanity_events or [])
        super().__init__(message or f"Sanity check failed after task {task!r}")


class ParallelExecutionError(WorkflowError):
    """One or more elements in a ``wf.parallel`` body raised.

    Spec 05 §4 D3 — runtime captures per-element exceptions instead of
    cancelling siblings; once ``asyncio.gather`` finishes, the runtime
    raises this with a ``failures`` map (element index → exception) so
    callers can introspect which elements failed without losing the
    siblings' outcomes (which are already recorded in
    ``state.results[body]`` by index, with ``None`` placeholders for
    failed indices).

    Attributes:
        body: Name of the parallel body task.
        failures: Mapping ``element_index → original_exception``.
    """

    def __init__(self, body: str, failures: dict[int, Exception]) -> None:
        self.body = body
        self.failures = failures
        indices = sorted(failures.keys())
        super().__init__(
            f"Parallel body {body!r} had {len(failures)} element failure(s) "
            f"at indices {indices}: "
            + ", ".join(f"[{i}] {type(failures[i]).__name__}: {failures[i]}" for i in indices)
        )


# ── Workflow-level warnings (non-fatal) ─────────────────────────────────────


class LoopMaxItersExceeded(UserWarning):
    """Emitted when ``wf.loop(..., max_iters=N)`` reaches the cap.

    The runtime forces ``Next("exit")`` once the loop's ``until`` task has
    dispatched ``Next("continue")`` ``max_iters`` times. The workflow itself
    completes successfully — the warning lets callers detect runaway loops
    without having to fail the run. Use ``pytest.warns(LoopMaxItersExceeded)``
    or the standard :mod:`warnings` filters to catch it.
    """


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
        sanity_events: list[dict[str, Any]] | None = None,
    ) -> None:
        self.status = status
        self.outputs = outputs
        self.run_id = run_id
        self.execution_id = execution_id
        self.sanity_events: list[dict[str, Any]] = list(sanity_events or [])

    def __repr__(self) -> str:
        return f"WorkflowResult(status={self.status!r}, tasks={list(self.outputs.keys())})"


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
