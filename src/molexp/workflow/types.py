"""Core result types for molexp workflow API.

Context types live in ``workflow.context``; this module holds only
result / execution-handle types and shared type variables.

The workflow public surface is uniformly pydantic — value types are
:class:`pydantic.BaseModel` (frozen). Runtime containers that hold
live ``asyncio`` objects (e.g. :class:`WorkflowExecution`) remain plain
Python classes per the project's typing rule. The only dataclass-based
code in the workflow layer lives in the private ``_pydantic_graph/``
shim, where the upstream :class:`pydantic_graph.BaseNode` protocol
requires dataclass nodes — that constraint is imposed by the
third-party library, not by molexp.
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict

from .._typing import JSONValue, TaskOutput
from ._pydantic_graph import End as End

# ── Route routing token (public) ─────────────────────────────────────────────


class Next(BaseModel):
    """Routing token — the public return value of branch / loop-``until`` tasks.

    ``Next("label")`` picks one of the ``routes={label: target}`` entries
    declared on the task (via ``wf.branch`` or ``@wf.task(routes=...)``);
    it does NOT jump to a task named ``label``. A ``wf.loop`` ``until``
    task returns ``Next("continue")`` to repeat the body or
    ``Next("exit")`` to proceed to ``on_exit``.

    Return ``(value, Next("label"))`` to carry a value on the routed edge:
    the target task receives ``value`` as its ``ctx.inputs``
    (values-on-edges delivery; a declared ``depends_on`` interface always
    wins). Part of ``molexp.workflow.__all__`` — import it as
    ``from molexp.workflow import Next``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str

    def __init__(self, label: str | None = None, **data: JSONValue) -> None:
        # Accept ``Next("continue")`` positionally as well as ``Next(label=...)``.
        if label is not None and "label" not in data:
            data["label"] = label
        super().__init__(**data)


# ── Workflow terminator ─────────────────────────────────────────────────────
# ``End`` is re-exported from ``pydantic_graph`` so molexp does not maintain
# a duplicate sentinel class. ``molexp.workflow.End is pydantic_graph.End``
# is a runtime invariant; see § Workflow ↔ pydantic-graph boundary in
# CLAUDE.md and ``test_pydantic_graph_boundary.py``.


# ── Edge sum types (spec §3, §7) ────────────────────────────────────────────


class UnconditionalEdges(BaseModel):
    """A node's outgoing control edge set, all unconditional.

    Empty ``targets`` = terminal node (0 out edges). Non-empty = either a
    single forward edge or a fan-out to multiple successors.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    targets: tuple[str, ...]


class BranchEdges(BaseModel):
    """A node's outgoing control edge set, label-routed.

    The task body MUST return ``Next(label)`` selecting one of the declared
    labels. Mixing branch + unconditional on the same node is rejected at
    compile time (``EdgeShapeError``).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

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


class ParallelExecutionError(WorkflowError):
    """One or more elements in a ``wf.parallel`` body raised.

    Spec 05 §4 D3 — runtime captures per-element exceptions instead of
    cancelling siblings; once ``asyncio.gather`` finishes, the runtime
    raises this with a ``failures`` map (element index → exception) so
    callers can introspect which elements failed without losing the
    siblings' outcomes (which are already recorded in
    ``state.results[body]`` by index, with ``None`` placeholders for
    failed indices).
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


class CommandError(WorkflowError):
    """An external command run by a :class:`~molexp.workflow.CommandTask` exited non-zero.

    Carries the command's ``returncode``, ``stdout``, and ``stderr`` for caller
    introspection; the message surfaces ``stderr`` (falling back to ``stdout``
    when ``stderr`` is empty). Under ``wf.parallel`` it is captured per element
    like any other :class:`WorkflowError`.
    """

    def __init__(self, returncode: int, stdout: str, stderr: str) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        detail = (stderr or stdout or "").strip()
        super().__init__(
            f"command exited with returncode {returncode}" + (f": {detail}" if detail else "")
        )


class MissingUpstreamResultError(WorkflowError):
    """A consumer's declared dependency has no recorded result.

    Raised by ``_collect_upstream_outputs`` when a multi-dependency consumer
    asks for a declared dependency name that never landed in
    ``WorkflowState.results`` — turning the old silent ``dict.get`` ``None``
    coalescing into a loud, named failure (the dependency barrier guarantees
    presence on the happy path, so this is a contract assertion). The message
    names the consumer task, the missing dependency, and the recorded names.
    """

    def __init__(self, consumer: str, missing: list[str], recorded: list[str]) -> None:
        self.consumer = consumer
        self.missing = missing
        self.recorded = recorded
        super().__init__(
            f"task {consumer!r} expected upstream result(s) {missing} but none were "
            f"recorded; recorded results: {recorded}"
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


class WorkflowResult(BaseModel):
    """Result of a completed workflow execution.

    Attributes:
        status: ``"completed"`` | ``"failed"`` | ``"cancelled"``
        outputs: Mapping of task name to task output.
        run_id: Associated workspace Run ID, if any.
        execution_id: Opaque ID for resumption support.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: str
    outputs: dict[str, TaskOutput]
    run_id: str | None = None
    execution_id: str | None = None

    def __repr__(self) -> str:
        return f"WorkflowResult(status={self.status!r}, tasks={list(self.outputs.keys())})"


class WorkflowExecution:
    """Handle for a running workflow.

    Returned by ``Workflow.start()`` for async control. This is a
    runtime container — it carries live ``asyncio`` state in concrete
    subclasses — and is therefore a plain Python class rather than a
    pydantic model.
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
