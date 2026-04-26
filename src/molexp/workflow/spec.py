"""Workflow specification — unified OOP API.

Define a workflow by instantiating :class:`Workflow` and registering tasks
through its methods. Decorator and builder styles share the same class::

    wf = Workflow(name="pipeline")

    @wf.task
    async def fetch(ctx: TaskContext) -> FetchResult: ...

    @wf.task(depends_on=["fetch"])
    async def validate(ctx: TaskContext) -> ValidateResult: ...

    # OOP style — add Task / Actor instances
    wf.add(ProcessTask(), depends_on=["validate"])

    # Control flow
    @wf.parallel_map(fan_out_over="items", depends_on=["fetch"])
    async def process_item(ctx): ...

    @wf.join(reducer="sum", depends_on=["process_item"])
    async def collect(ctx): ...

    spec = wf.build()
    result = await spec.execute(run=run)
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from typing import Any

from molexp.config import ProfileConfig

from .types import WorkflowExecution, WorkflowResult


def _callable_name(f: Callable, fallback: str = "anonymous") -> str:
    """Return a Python function's ``__name__`` if present, else ``fallback``.

    Type-checked codepaths annotate decorator targets as ``Callable``, which
    static checkers cannot prove has ``__name__``. In practice every decorated
    target is a function, so ``getattr`` is sufficient and keeps the annotation
    free of more specific protocols.
    """
    return getattr(f, "__name__", None) or fallback


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

    Produced by :meth:`Workflow.build`.
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


# ── Workflow (unified OOP API) ──────────────────────────────────────────────


class Workflow:
    """OOP workflow definition. Supports decorator and builder styles.

    Instantiate once, then register tasks via the decorators
    (:meth:`task`, :meth:`actor`, :meth:`parallel_map`, :meth:`join`)
    or the OOP method :meth:`add`. Call :meth:`build` to produce a
    :class:`WorkflowSpec`.

    Example (decorator)::

        wf = Workflow(name="pipeline")

        @wf.task
        async def fetch(ctx: TaskContext) -> dict: ...

    Example (OOP)::

        wf = Workflow(name="pipeline")
        wf.add(FetchTask())
        wf.add(ProcessTask(), depends_on=["fetch"])
    """

    def __init__(self, name: str, mode: str = "batch") -> None:
        self._name = name
        self._mode = mode
        self._tasks: list[TaskRegistration] = []

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return self._name

    @property
    def mode(self) -> str:
        return self._mode

    # ── Decorator: function-as-task ─────────────────────────────────────

    def task(
        self,
        fn: Callable | None = None,
        *,
        depends_on: list[str] | None = None,
        name: str | None = None,
        remote: Any = None,
    ) -> Callable:
        """Register a function as a batch workflow task.

        Usage::

            @wf.task
            async def fetch(ctx): ...

            @wf.task(depends_on=["fetch"])
            async def validate(ctx): ...
        """

        def decorator(f: Callable) -> Callable:
            task_name = name or _callable_name(f)
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
            actor_name = name or _callable_name(f)
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

    # ── OOP: register a Task/Actor instance or any Runnable/Streamable ──

    def add(
        self,
        task: Any,
        *,
        depends_on: list[str] | None = None,
        name: str | None = None,
        remote: Any = None,
    ) -> Workflow:
        """Register a Task / Actor instance (or any Runnable/Streamable).

        Accepts:
        - A :class:`~molexp.workflow.task.Task` / :class:`~molexp.workflow.task.Actor` instance
        - Any object matching the :class:`~molexp.workflow.protocols.Runnable`
          or :class:`~molexp.workflow.protocols.Streamable` protocol
        - A bare callable (treated as a batch task)

        Returns ``self`` to support chaining.
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

    # ── Control-flow decorators ─────────────────────────────────────────

    def parallel_map(
        self,
        *,
        fan_out_over: str,
        depends_on: list[str] | None = None,
        name: str | None = None,
    ) -> Callable:
        """Decorator for fan-out parallel tasks."""

        def decorator(fn: Callable) -> Callable:
            task_name = name or _callable_name(fn)
            self._tasks.append(
                TaskRegistration(
                    name=task_name,
                    fn_or_class=fn,
                    depends_on=depends_on or [],
                    is_actor=False,
                )
            )
            # Stash the fan-out config on the function object; consumed by the
            # graph compiler. ``setattr`` keeps the static checker happy without
            # an ignore directive whose placement is fragile under autoformat.
            setattr(fn, "_parallel_map_config", {"fan_out_over": fan_out_over})
            return fn

        return decorator

    def join(
        self,
        *,
        reducer: str | Callable | None = None,
        depends_on: list[str] | None = None,
        name: str | None = None,
    ) -> Callable:
        """Decorator for collecting and reducing parallel outputs."""

        def decorator(fn: Callable) -> Callable:
            task_name = name or _callable_name(fn)
            self._tasks.append(
                TaskRegistration(
                    name=task_name,
                    fn_or_class=fn,
                    depends_on=depends_on or [],
                    is_actor=False,
                )
            )
            # See ``parallel_map`` for the rationale behind ``setattr``.
            setattr(fn, "_join_config", {"reducer": reducer})
            return fn

        return decorator

    # ── Compile ─────────────────────────────────────────────────────────

    def build(self) -> WorkflowSpec:
        """Compile the registered tasks into a :class:`WorkflowSpec`."""
        tasks = list(self._tasks)
        return WorkflowSpec(
            name=self._name,
            workflow_id=_stable_workflow_id(self._name, tasks),
            tasks=tasks,
            mode=self._mode,
        )


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
