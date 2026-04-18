"""Workflow specification: functional DSL and OOP builder.

Two equivalent ways to define a workflow:

Functional style::

    wf = workflow(name="pipeline")

    @wf.task
    async def fetch(ctx: TaskContext) -> FetchResult: ...

    @wf.task(depends_on=["fetch"])
    async def validate(ctx: TaskContext) -> ValidateResult: ...

    result = await wf.build().execute(run=run)

OOP builder style::

    wf = (
        WorkflowBuilder(name="pipeline")
        .add(FetchTask())
        .add(ValidateTask(), depends_on=["fetch"])
        .build()
    )
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from typing import Any

from molexp.config import ProfileConfig

from .types import WorkflowExecution, WorkflowResult


class TaskRegistration:
    """Internal record of a registered task or actor."""

    __slots__ = ("name", "fn_or_class", "depends_on", "is_actor", "remote")

    def __init__(
        self,
        name: str,
        fn_or_class: Any,
        depends_on: list[str],
        is_actor: bool = False,
        remote: Any = None,
    ) -> None:
        self.name = name
        self.fn_or_class = fn_or_class
        self.depends_on = depends_on
        self.is_actor = is_actor
        self.remote = remote


# ── WorkflowSpec ────────────────────────────────────────────────────────────


class WorkflowSpec:
    """Compiled, executable workflow specification.

    Produced by ``WorkflowBuilder.build()`` or ``WorkflowDSL.build()``.
    """

    def __init__(
        self,
        name: str,
        workflow_id: str,
        tasks: list[TaskRegistration],
        mode: str = "batch",
    ) -> None:
        self.name = name
        self.workflow_id = workflow_id
        self._tasks = tasks
        self._mode = mode
        self._runtime: Any = None  # WorkflowRuntime, lazy

    def _get_runtime(self) -> Any:
        if self._runtime is None:
            from .runtime import create_default_runtime

            self._runtime = create_default_runtime()
        return self._runtime

    async def execute(
        self,
        run: Any = None,
        run_context: Any = None,
        *,
        profile_config: ProfileConfig | None = None,
        **kwargs: Any,
    ) -> WorkflowResult:
        """Run the workflow to completion and return the result.

        Args:
            run: A workspace ``Run`` object (runtime creates RunContext).
            run_context: An existing ``RunContext`` (used directly).
                Mutually exclusive with *run*.
            profile_config: Active :class:`~molexp.config.ProfileConfig`
                for this execution.  When *run_context* is passed, the
                context's own config takes precedence.
        """
        return await self._get_runtime().execute(
            self,
            run=run,
            run_context=run_context,
            profile_config=profile_config,
            **kwargs,
        )

    async def start(
        self,
        run: Any = None,
        run_context: Any = None,
        *,
        profile_config: ProfileConfig | None = None,
        **kwargs: Any,
    ) -> WorkflowExecution:
        """Start the workflow asynchronously and return a handle."""
        return await self._get_runtime().start(
            self,
            run=run,
            run_context=run_context,
            profile_config=profile_config,
            **kwargs,
        )


# ── Functional DSL ──────────────────────────────────────────────────────────


class WorkflowDSL:
    """Builder returned by :func:`workflow`.

    Provides ``@wf.task`` and ``@wf.actor`` decorators.
    Call ``.build()`` to produce a :class:`WorkflowSpec`.
    """

    def __init__(self, name: str, mode: str = "batch") -> None:
        self._name = name
        self._mode = mode
        self._tasks: list[TaskRegistration] = []

    def task(
        self,
        fn: Callable | None = None,
        *,
        depends_on: list[str] | None = None,
        name: str | None = None,
        remote: Any = None,
    ) -> Callable:
        """Register a function as a workflow task.

        Usage::

            @wf.task
            async def fetch(ctx): ...

            @wf.task(depends_on=["fetch"])
            async def validate(ctx): ...
        """

        def decorator(f: Callable) -> Callable:
            task_name = name or f.__name__
            self._tasks.append(
                TaskRegistration(
                    name=task_name,
                    fn_or_class=f,
                    depends_on=depends_on or [],
                    is_actor=False,
                    remote=remote,
                )
            )
            return f

        if fn is not None:
            return decorator(fn)
        return decorator

    def actor(
        self,
        fn: Callable | None = None,
        *,
        depends_on: list[str] | None = None,
        name: str | None = None,
    ) -> Callable:
        """Register an async generator as a streaming actor."""

        def decorator(f: Callable) -> Callable:
            actor_name = name or f.__name__
            self._tasks.append(
                TaskRegistration(
                    name=actor_name,
                    fn_or_class=f,
                    depends_on=depends_on or [],
                    is_actor=True,
                )
            )
            return f

        if fn is not None:
            return decorator(fn)
        return decorator

    def build(self) -> WorkflowSpec:
        """Compile registered tasks into a :class:`WorkflowSpec`."""
        tasks = list(self._tasks)
        return WorkflowSpec(
            name=self._name,
            workflow_id=_stable_workflow_id(self._name, tasks),
            tasks=tasks,
            mode=self._mode,
        )


def workflow(name: str, mode: str = "batch") -> WorkflowDSL:
    """Create a workflow DSL builder.

    Example::

        wf = workflow(name="data-pipeline")

        @wf.task
        async def fetch(ctx: TaskContext) -> FetchResult: ...

        result = await wf.build().execute(run=run)
    """
    return WorkflowDSL(name=name, mode=mode)


# ── OOP builder ─────────────────────────────────────────────────────────────


class WorkflowBuilder:
    """OOP-style builder for composing Task / Actor instances.

    Example::

        wf = (
            WorkflowBuilder(name="data-pipeline")
            .add(FetchTask())
            .add(ValidateTask(), depends_on=["fetch"])
            .build()
        )
    """

    def __init__(self, name: str, mode: str = "batch") -> None:
        self._name = name
        self._mode = mode
        self._tasks: list[TaskRegistration] = []

    def add(
        self,
        task: Any,
        *,
        depends_on: list[str] | None = None,
        name: str | None = None,
        remote: Any = None,
    ) -> WorkflowBuilder:
        """Add a task or actor instance.

        Accepts:
        - A :class:`~Task` / :class:`~Actor` subclass instance
        - Any object matching the :class:`~Runnable` or :class:`~Streamable` protocol
        - A bare callable (treated as a batch task)
        """
        from .protocols import Streamable

        task_name = name or _to_snake_case(type(task).__name__)
        for suffix in ("_task", "_actor"):
            if task_name.endswith(suffix):
                task_name = task_name[: -len(suffix)]
                break

        self._tasks.append(
            TaskRegistration(
                name=task_name,
                fn_or_class=task,
                depends_on=depends_on or [],
                is_actor=isinstance(task, Streamable),
                remote=remote,
            )
        )
        return self

    def build(self) -> WorkflowSpec:
        """Build the final :class:`WorkflowSpec`."""
        tasks = list(self._tasks)
        return WorkflowSpec(
            name=self._name,
            workflow_id=_stable_workflow_id(self._name, tasks),
            tasks=tasks,
            mode=self._mode,
        )


# ── Control-flow helpers ────────────────────────────────────────────────────


def parallel_map(
    wf: WorkflowDSL,
    *,
    fan_out_over: str,
    depends_on: list[str] | None = None,
    name: str | None = None,
) -> Callable:
    """Decorator for fan-out parallel tasks."""

    def decorator(fn: Callable) -> Callable:
        task_name = name or fn.__name__
        wf._tasks.append(
            TaskRegistration(
                name=task_name,
                fn_or_class=fn,
                depends_on=depends_on or [],
                is_actor=False,
            )
        )
        fn._parallel_map_config = {"fan_out_over": fan_out_over}  # type: ignore[attr-defined]
        return fn

    return decorator


def join(
    wf: WorkflowDSL,
    *,
    reducer: str | Callable | None = None,
    depends_on: list[str] | None = None,
    name: str | None = None,
) -> Callable:
    """Decorator for collecting and reducing parallel outputs."""

    def decorator(fn: Callable) -> Callable:
        task_name = name or fn.__name__
        wf._tasks.append(
            TaskRegistration(
                name=task_name,
                fn_or_class=fn,
                depends_on=depends_on or [],
                is_actor=False,
            )
        )
        fn._join_config = {"reducer": reducer}  # type: ignore[attr-defined]
        return fn

    return decorator


# ── Helpers ─────────────────────────────────────────────────────────────────


def _to_snake_case(name: str) -> str:
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    name = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", name)
    return name.lower()


def _stable_workflow_id(name: str, tasks: list[TaskRegistration]) -> str:
    """Deterministic workflow ID from name + task topology."""
    parts = [name]
    for t in tasks:
        dep_str = ",".join(sorted(t.depends_on))
        parts.append(f"{t.name}:{type(t.fn_or_class).__qualname__}:[{dep_str}]")
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]
