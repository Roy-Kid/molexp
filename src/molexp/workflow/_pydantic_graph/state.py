"""Internal pydantic-graph state and deps types.

Users never import these directly — they touch them only through the
public ``WorkflowResult`` API.

This module MUST NOT import ``pydantic_graph`` — it carries only plain
data containers threaded through the per-task ``Step`` nodes built in
:mod:`.compiler`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from ..protocols import (
    JSONMapping,
    RunContextLike,
    RunLike,
    TaskOutput,
    UserDeps,
)

if TYPE_CHECKING:
    import anyio

    from .._graph_decl import ParallelDecl, TaskRegistration
    from ..cache import Caching
    from ..snapshot import TaskSnapshot


@dataclass
class WorkflowState:
    """Shared, **mutated-in-place** state threaded through workflow tasks.

    pydantic-graph holds a single reference to this object for the whole
    run and snapshots it after each node. molexp tasks do not read their
    inputs from edge tokens — each task reads upstream outputs from this
    shared ``results`` dict (via ``_collect_upstream_outputs``). Each
    per-task ``Step`` therefore mutates ``results`` in place:
    ``ctx.state.results[name] = output``.

    Fields:

    * ``results`` — task_name → output as tasks finish; loops overwrite
      prior values ("曾经完成过一次" semantics).
    * ``completed`` — names of tasks that finished at least once.
    * ``loop_counters`` — per-loop ``until``-task → iteration count; the
      ``until`` step increments and consults this to enforce
      ``wf.loop(..., max_iters=N)``.
    * ``parallel_runs`` — ``wf.parallel`` body fan-out width, recorded by
      the collector step for observability.
    * ``failed`` / ``error`` — terminal failure flags.
    * ``seeded`` — names that arrived already-completed via
      ``Workflow.execute(seed_outputs=...)``; their step skips the body
      but still routes normally.
    * ``running`` — number of task bodies currently executing. The
      per-Step dependency barrier reads this as a frontier-liveness
      signal: ``running == 0`` while a dependency is still missing means
      no body is in flight to ever satisfy it (deadlock detection).
    * ``progress`` — monotonic counter bumped on every ``record`` and on
      every parallel-collector publish; lets the barrier distinguish a
      quiescent (no progress) window from one where results are landing.
    """

    results: dict[str, TaskOutput] = field(default_factory=dict)
    completed: set[str] = field(default_factory=set)
    loop_counters: dict[str, int] = field(default_factory=dict)
    parallel_runs: dict[str, int] = field(default_factory=dict)
    failed: bool = False
    error: str | None = None
    seeded: set[str] = field(default_factory=set)
    running: int = 0
    progress: int = 0

    @classmethod
    def from_seed(cls, seed: Mapping[str, TaskOutput]) -> WorkflowState:
        """Construct an initial state seeded with already-known task outputs.

        Used by the PlanMode review→repair loop (``Workflow.execute(
        seed_outputs=...)``): each seeded entry is treated as a task that
        already finished successfully, so its name lands in both
        ``completed`` and ``seeded``. Downstream tasks find the values in
        ``results``; the seeded task's own step skips its body.
        """
        names = set(seed)
        return cls(
            results=dict(seed),
            completed=set(names),
            seeded=set(names),
        )

    def record(self, step_name: str, output: TaskOutput) -> None:
        """Record *step_name*'s output in place and mark it completed."""
        self.results[step_name] = output
        self.completed.add(step_name)
        self.progress += 1


@dataclass
class WorkflowDeps:
    """Dependencies injected into every per-task ``Step`` node.

    Attributes:
        run: The molexp Run associated with this execution (may be None).
        run_context: The active RunContext (may be None).
        config: The active :class:`~molexp.profile.ProfileConfig` (may be None).
        user_deps: Application-level deps forwarded from the caller.
        remote_executor: Optional remote-execution gateway (set by molq).
        run_dir: Path to the run's directory on disk (may be None).
        registration_by_name: name → :class:`TaskRegistration`. Built fresh
            per execution by the runtime from ``compiled._tasks``.
        parallel_decls: ``body_task_name → ParallelDecl``.
        loop_max_iters: ``until_task_name → max_iters`` (``wf.loop`` guard).
        parallel_limiters: ``body_task_name → anyio.CapacityLimiter`` —
            one fresh limiter per parallel body, sized to
            ``decl.max_concurrency``, bounding the map fan-out.
        cache: Optional content-addressed :class:`~molexp.workflow.cache.Caching`.
            ``None`` (default) disables caching — the per-task Step hook
            behaves exactly as before. The runtime resolves the effective
            cache and populates this field per execution.
        snapshots: ``task_name → TaskSnapshot`` (the compiled artifact's
            per-task static identity). The cache hook keys on
            ``snapshots[name].key | input_hash``.
    """

    run: RunLike | None = None
    run_context: RunContextLike | None = None
    config: JSONMapping | None = None
    user_deps: UserDeps = None
    # ``remote_executor`` is a duck-typed callable from molq when present.
    # It is reached only by molq-aware tasks and is opaque to the runtime.
    remote_executor: UserDeps = None
    run_dir: Path | None = None
    registration_by_name: Mapping[str, TaskRegistration] = field(default_factory=dict)
    parallel_decls: Mapping[str, ParallelDecl] = field(default_factory=dict)
    loop_max_iters: Mapping[str, int] = field(default_factory=dict)
    parallel_limiters: Mapping[str, anyio.CapacityLimiter] = field(default_factory=dict)
    cache: Caching | None = None
    snapshots: Mapping[str, TaskSnapshot] = field(default_factory=dict)
