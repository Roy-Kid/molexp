"""User-facing experiment specification.

An ``Experiment`` is a lightweight spec that binds a workflow to a
parameter space.  It carries no filesystem state — the CLI materializes
it into workspace entities at execution time.

Example::

    experiment = project.experiment(
        "lr-sweep",
        params=me.GridSpace({"lr": [1e-4, 3e-4, 1e-3]}),
        n_replicas=3,
    )

    def train(ctx: me.RunContext) -> None: ...

    experiment.set_workflow(train)
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from molexp.workflow.context import TaskContext
from molexp.workflow.spec import WorkflowBuilder, WorkflowSpec
from molexp.workflow.task import Task

if TYPE_CHECKING:
    from molexp.project import Project

# Default replica seeds — deterministic, well-separated
_DEFAULT_SEEDS = [42, 123, 456, 789, 1234]


class _EntryTask(Task):
    """Wraps a bare ``fn(RunContext) -> None`` into a workflow Task.

    When the user passes a plain function to ``set_workflow()``, it is
    promoted to a single-Task ``WorkflowSpec`` via this wrapper.  The
    wrapper calls the function with the ``RunContext`` extracted from the
    ``TaskContext``.
    """

    def __init__(self, fn: Callable) -> None:
        self._fn = fn

    async def execute(self, ctx: TaskContext) -> None:
        run_ctx = ctx.run_context
        if run_ctx is None:
            raise RuntimeError(
                f"{self._fn.__name__}() requires a RunContext, but the "
                "workflow was executed without a workspace run."
            )
        result = self._fn(run_ctx)
        if asyncio.iscoroutine(result) or inspect.isawaitable(result):
            await result


def _promote_to_workflow(fn: Callable, name: str) -> WorkflowSpec:
    """Promote a bare ``fn(RunContext)`` to a single-Task WorkflowSpec."""
    task = _EntryTask(fn)
    return WorkflowBuilder(name=name).add(task, name=fn.__name__).build()


class Experiment:
    """User-facing experiment specification (no filesystem side effects).

    Created via :meth:`Project.experiment` — do not instantiate directly.

    Attributes:
        name: Experiment name.
        project: Parent project spec.
        params: Parameter space for sweeps (``None`` = single run).
        n_replicas: Number of replica runs per parameter combination.
        description: Human-readable description.
        tags: Metadata tags.
    """

    def __init__(
        self,
        name: str,
        project: Project,
        *,
        params: Any | None = None,
        n_replicas: int = 1,
        description: str = "",
        tags: list[str] | None = None,
        seeds: list[int] | None = None,
    ) -> None:
        self.name = name
        self.project = project
        self.params = params
        self.n_replicas = n_replicas
        self.description = description
        self.tags = tags or []
        self.seeds = seeds
        self._workflow: WorkflowSpec | None = None

    def set_workflow(self, workflow: WorkflowSpec | Callable) -> None:
        """Bind a workflow to this experiment.

        If *workflow* is a callable ``(RunContext) -> None``, it is
        auto-promoted to a single-Task :class:`WorkflowSpec`.  The internal
        type is **always** ``WorkflowSpec`` — no multiple types.

        Args:
            workflow: A compiled ``WorkflowSpec``, or a bare callable that
                accepts ``RunContext``.

        Raises:
            TypeError: If *workflow* is not a ``WorkflowSpec`` or callable.
            ValueError: If a workflow is already bound.
        """
        if self._workflow is not None:
            raise ValueError(
                f"Experiment {self.name!r} already has a workflow bound. "
                "Call set_workflow() only once per experiment."
            )
        if isinstance(workflow, WorkflowSpec):
            self._workflow = workflow
        elif callable(workflow):
            self._workflow = _promote_to_workflow(workflow, self.name)
        else:
            raise TypeError(
                f"Expected WorkflowSpec or callable, got {type(workflow).__name__}"
            )

    @property
    def workflow(self) -> WorkflowSpec | None:
        """The bound workflow (always ``WorkflowSpec`` or ``None``)."""
        return self._workflow

    def get_seeds(self) -> list[int]:
        """Return replica seeds (length == ``n_replicas``)."""
        if self.seeds is not None:
            return list(self.seeds[: self.n_replicas])
        seeds = list(_DEFAULT_SEEDS)
        while len(seeds) < self.n_replicas:
            seeds.append(seeds[-1] + 111)
        return seeds[: self.n_replicas]
