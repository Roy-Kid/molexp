"""Internal pydantic-graph state and deps types.

Users never import these directly — they touch them only through the
public ``WorkflowResult`` API.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from ..protocols import (
    JSONMapping,
    RunContextLike,
    RunLike,
    TaskBody,
    TaskOutput,
    UserDeps,
)

if TYPE_CHECKING:
    from .._graph_decl import ParallelDecl, TaskRegistration
    from ..types import OutEdges


@dataclass
class WorkflowState:
    """Shared mutable state threaded through workflow tasks.

    Spec 04 frontier-based scheduling state:

    * ``results`` — task_name → output as tasks finish; loops overwrite
      prior values ("曾经完成过一次" semantics).
    * ``completed`` — names of tasks that finished at least once during
      this run; never shrinks across loop iterations (used by the data-
      dep satisfaction check).
    * ``pending_targets`` — frontier candidates waiting on unmet data
      dependencies; carried across frames for join semantics.
    * ``loop_counters`` — per-loop ``until``-task → iteration count;
      WorkflowStep increments and consults this to enforce
      ``wf.loop(..., max_iters=N)``.
    """

    results: dict[str, TaskOutput] = field(default_factory=dict)
    completed: frozenset[str] = field(default_factory=frozenset)
    pending_targets: tuple[str, ...] = field(default_factory=tuple)
    loop_counters: dict[str, int] = field(default_factory=dict)
    parallel_runs: dict[str, int] = field(default_factory=dict)
    failed: bool = False
    error: str | None = None
    # Names that arrived already-completed via ``Workflow.execute(seed_outputs=...)``.
    # The frontier scheduler filters these out of every frame so they are
    # consumed via ``state.results`` but never re-executed. Distinct from
    # ``completed`` because loops legitimately rerun tasks already in
    # ``completed`` whereas seeded entries should never run.
    seeded: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def from_seed(cls, seed: Mapping[str, TaskOutput]) -> WorkflowState:
        """Construct an initial state seeded with already-known task outputs.

        Used by the PlanMode review→repair loop (``Workflow.execute(
        seed_outputs=...)``): each seeded entry is treated as a task
        that already finished successfully, so its name lands in both
        ``completed`` and ``seeded``. The data-dep satisfaction check
        sees the names in ``completed`` (so downstream is ready); the
        frontier filter sees them in ``seeded`` (so the body never
        runs).
        """
        names = frozenset(seed)
        return cls(
            results=dict(seed),
            completed=names,
            seeded=names,
        )

    def record(self, step_name: str, output: TaskOutput) -> WorkflowState:
        """Return a new state with the given task's output recorded.

        Adds *step_name* to ``completed``; loops re-running the same task
        leave ``completed`` unchanged but overwrite ``results[step_name]``.
        """
        return WorkflowState(
            results={**self.results, step_name: output},
            completed=self.completed | {step_name},
            pending_targets=self.pending_targets,
            loop_counters=dict(self.loop_counters),
            parallel_runs=dict(self.parallel_runs),
            failed=self.failed,
            error=self.error,
            seeded=self.seeded,
        )

    def mark_completed(self, names: Iterable[str]) -> WorkflowState:
        """Return a new state extending ``completed`` with *names*."""
        return WorkflowState(
            results=self.results,
            completed=self.completed | frozenset(names),
            pending_targets=self.pending_targets,
            loop_counters=dict(self.loop_counters),
            parallel_runs=dict(self.parallel_runs),
            failed=self.failed,
            error=self.error,
            seeded=self.seeded,
        )

    def set_pending(self, targets: Iterable[str]) -> WorkflowState:
        """Return a new state with ``pending_targets`` replaced."""
        return WorkflowState(
            results=self.results,
            completed=self.completed,
            pending_targets=tuple(targets),
            loop_counters=dict(self.loop_counters),
            parallel_runs=dict(self.parallel_runs),
            failed=self.failed,
            error=self.error,
            seeded=self.seeded,
        )

    def with_loop_counter(self, until_name: str, count: int) -> WorkflowState:
        """Return a new state with ``loop_counters[until_name]`` set to *count*."""
        return WorkflowState(
            results=self.results,
            completed=self.completed,
            pending_targets=self.pending_targets,
            loop_counters={**self.loop_counters, until_name: count},
            parallel_runs=dict(self.parallel_runs),
            failed=self.failed,
            error=self.error,
            seeded=self.seeded,
        )

    def with_parallel_run(self, body_name: str, count: int) -> WorkflowState:
        """Return a new state with ``parallel_runs[body_name]`` set to *count*."""
        return WorkflowState(
            results=self.results,
            completed=self.completed,
            pending_targets=self.pending_targets,
            loop_counters=dict(self.loop_counters),
            parallel_runs={**self.parallel_runs, body_name: count},
            failed=self.failed,
            error=self.error,
            seeded=self.seeded,
        )

    def fail(self, step_name: str, exc: Exception) -> WorkflowState:
        """Return a new state marked as failed."""
        return WorkflowState(
            results=self.results,
            completed=self.completed,
            pending_targets=self.pending_targets,
            loop_counters=dict(self.loop_counters),
            parallel_runs=dict(self.parallel_runs),
            failed=True,
            error=f"Step '{step_name}' failed: {exc}",
            seeded=self.seeded,
        )

    def _sync_from(self, other: WorkflowState) -> None:
        """Update this state in-place from *other*.

        pydantic-graph holds a reference to the state object and snapshots
        it after each node. We MUST mutate the original reference so the
        snapshot reflects the latest outputs. This method centralises
        that necessary mutation in one place.
        """
        self.results = other.results
        self.completed = other.completed
        self.pending_targets = other.pending_targets
        self.loop_counters = other.loop_counters
        self.parallel_runs = other.parallel_runs
        self.failed = other.failed
        self.error = other.error
        self.seeded = other.seeded


@dataclass
class WorkflowDeps:
    """Dependencies injected into every pydantic-graph node.

    Attributes:
        run: The molexp Run associated with this execution (may be None).
        run_context: The active RunContext (may be None).
        config: The active :class:`~molexp.config.ProfileConfig` (may be None).
        user_deps: Application-level deps forwarded from the caller.
        remote_executor: Optional remote-execution gateway (set by molq).
        run_dir: Path to the run's directory on disk (may be None).
        task_by_name: name → registered task body (Task / Actor / Runnable /
            Streamable / async callable). Populated by the compiler.
        out_edges: name → :data:`~molexp.workflow.types.OutEdges` (sum of
            :class:`UnconditionalEdges` / :class:`BranchEdges`).
            Populated by the compiler.
        entry_frontier: task names making up the initial frontier.
        loop_max_iters: ``wf.loop`` runtime guard. ``until_task_name →
            max_iters``; WorkflowStep increments
            ``state.loop_counters[until_name]`` each time the until task
            dispatches ``Next("continue")``, and forces ``Next("exit") +
            emits :class:`~molexp.workflow.LoopMaxItersExceeded` once the
            cap is reached.
    """

    run: RunLike | None = None
    run_context: RunContextLike | None = None
    config: JSONMapping | None = None
    user_deps: UserDeps = None
    # ``remote_executor`` is a duck-typed callable from molq when present.
    # It is reached only by molq-aware tasks and is opaque to the scheduler.
    remote_executor: UserDeps = None
    run_dir: Path | None = None
    task_by_name: Mapping[str, TaskBody] = field(default_factory=dict)
    registration_by_name: Mapping[str, TaskRegistration] = field(default_factory=dict)
    out_edges: Mapping[str, OutEdges] = field(default_factory=dict)
    entry_frontier: tuple[str, ...] = field(default_factory=tuple)
    loop_max_iters: Mapping[str, int] = field(default_factory=dict)
    # Spec 05 — wf.parallel runtime fan-out. ``body_task_name → ParallelDecl``;
    # WorkflowStep recognises body tasks present here and dispatches them
    # via a fan-out scheduler instead of a singleton invocation.
    parallel_decls: Mapping[str, ParallelDecl] = field(default_factory=dict)
