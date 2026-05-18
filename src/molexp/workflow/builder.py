"""WorkflowBuilder — decorator and OOP styles on the same instance."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from ._graph_decl import (
    DependentParamsFn,
    LoopDecl,
    ParallelDecl,
    TaskRegistration,
)
from ._helpers import _callable_name, _stable_workflow_id, _to_snake_case
from .protocols import JSONMapping, Streamable, TaskBody, TaskOutput, UserDeps
from .spec import Workflow


class WorkflowBuilder:
    """OOP workflow definition. Supports decorator and builder styles.

    Instantiate once, then register tasks via the decorators
    (:meth:`task`, :meth:`actor`) or the OOP method :meth:`add`. Wire
    control flow with :meth:`control` / :meth:`branch` / :meth:`loop`
    / :meth:`parallel`. Call :meth:`build` to produce a
    :class:`Workflow`.
    """

    def __init__(
        self,
        name: str,
        mode: str = "batch",
        version: str = "0",
        *,
        entry: str | list[str] | tuple[str, ...] | None = None,
    ) -> None:
        self._name = name
        self._mode = mode
        self._version = version
        self._tasks: list[TaskRegistration] = []
        self._entries: list[str] = []
        self._control_edges: list[tuple[str, str]] = []
        self._branch_edges: list[tuple[str, str, str]] = []
        self._loops: list[LoopDecl] = []
        self._parallels: list[ParallelDecl] = []
        self._reducer: tuple[str, Callable[..., TaskOutput]] | None = None
        if entry is not None:
            if isinstance(entry, str):
                self.entry(entry)
            else:
                for name_ in entry:
                    self.entry(name_)

    @property
    def name(self) -> str:
        return self._name

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def version_label(self) -> str:
        return self._version

    # ── Decorator: function-as-task ───────────────────────────────────────

    def task(
        self,
        fn: Callable | None = None,
        *,
        depends_on: list[str] | None = None,
        name: str | None = None,
        remote: UserDeps = None,
        routes: Mapping[str, str] | None = None,
        next_: str | None = None,
        dependent_params: DependentParamsFn | None = None,
    ) -> Callable:
        """Register a function as a batch workflow task.

        ``routes={label: target}`` declares branch outgoing control edges;
        ``next_=target`` declares a single unconditional control edge.
        The two are mutually exclusive.
        """
        if routes is not None and next_ is not None:
            raise TypeError("WorkflowBuilder.task: routes= and next_= are mutually exclusive")

        def decorator(f: Callable) -> Callable:
            task_name = name or _callable_name(f)
            self._tasks.append(
                TaskRegistration(
                    name=task_name,
                    fn_or_class=f,
                    depends_on=depends_on or [],
                    is_actor=False,
                    remote=remote,
                    dependent_params=dependent_params,
                )
            )
            self._record_decorator_edges(task_name, routes=routes, next_=next_)
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
        routes: Mapping[str, str] | None = None,
        next_: str | None = None,
    ) -> Callable:
        """Register an async generator as a streaming actor.

        Same ``routes=`` / ``next_=`` semantics as :meth:`task`.
        """
        if routes is not None and next_ is not None:
            raise TypeError("WorkflowBuilder.actor: routes= and next_= are mutually exclusive.")

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
            self._record_decorator_edges(actor_name, routes=routes, next_=next_)
            return f

        if fn is not None:
            return decorator(fn)
        return decorator

    # ── OOP: register a Task/Actor instance ───────────────────────────────

    def add(
        self,
        task: TaskBody,
        *,
        depends_on: list[str] | None = None,
        name: str | None = None,
        remote: UserDeps = None,
        task_type: str | None = None,
        config: JSONMapping | None = None,
        routes: Mapping[str, str] | None = None,
        next_: str | None = None,
        dependent_params: DependentParamsFn | None = None,
    ) -> WorkflowBuilder:
        """Register a Task / Actor instance (or any Runnable/Streamable).

        Pass ``task_type`` and ``config`` when the task came from a
        registry factory and you want :meth:`Workflow.to_dict` to
        produce serializable IR.

        Returns ``self`` to support chaining.
        """
        if routes is not None and next_ is not None:
            raise TypeError("WorkflowBuilder.add: routes= and next_= are mutually exclusive.")

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
                task_type=task_type,
                config=config,
                dependent_params=dependent_params,
            )
        )
        self._record_decorator_edges(task_name, routes=routes, next_=next_)
        return self

    # ── Control-flow declarations ─────────────────────────────────────────

    def entry(self, name: str) -> WorkflowBuilder:
        """Declare *name* as a workflow entry point. Multiple calls = multi-entry."""
        if name in self._entries:
            raise ValueError(
                f"WorkflowBuilder {self._name!r}: entry {name!r} declared multiple times"
            )
        self._entries.append(name)
        return self

    def control(self, src: str, to: str) -> WorkflowBuilder:
        """Declare an unconditional control edge ``src -> to``."""
        self._control_edges.append((src, to))
        return self

    def branch(
        self,
        src: str,
        label: str | None = None,
        to: str | None = None,
        *,
        routes: Mapping[str, str] | None = None,
    ) -> WorkflowBuilder:
        """Declare branch (label-routed) control edges on *src*.

        Two forms: ``wf.branch("src", "label", "target")`` or
        ``wf.branch("src", routes={"l1": "t1", "l2": "t2"})``.
        """
        if routes is not None:
            if label is not None or to is not None:
                raise TypeError(
                    "WorkflowBuilder.branch: pass either positional (src, label, to) "
                    "or keyword routes={...}, not both."
                )
            for lbl, target in routes.items():
                self._branch_edges.append((src, lbl, target))
            return self
        if label is None or to is None:
            raise TypeError(
                "WorkflowBuilder.branch: pass (src, label, to) or routes={...}; "
                "received a partial single-edge form."
            )
        self._branch_edges.append((src, label, to))
        return self

    def loop(
        self,
        *,
        body: list[str] | tuple[str, ...],
        until: str,
        max_iters: int,
        on_exit: str = "_end",
    ) -> WorkflowBuilder:
        """Declare a loop: ``body`` runs repeatedly until ``until`` exits.

        ``until`` returns ``Next("continue")`` to loop or ``Next("exit")`` to
        proceed to ``on_exit`` (default: terminate).
        """
        if not body:
            raise ValueError(
                f"WorkflowBuilder.loop: body must contain at least one task name; got {body!r}"
            )
        if max_iters < 1:
            raise ValueError(f"WorkflowBuilder.loop: max_iters must be >= 1; got {max_iters!r}")
        self._loops.append(
            LoopDecl(body=tuple(body), until=until, max_iters=max_iters, on_exit=on_exit)
        )
        return self

    def parallel(
        self,
        *,
        map_over: str,
        body: str,
        join: str,
        max_concurrency: int = 1,
    ) -> WorkflowBuilder:
        """Declare parallel fan-out: run *body* once per element of *map_over* output."""
        if max_concurrency < 1:
            raise ValueError(
                f"WorkflowBuilder.parallel: max_concurrency must be >= 1; got {max_concurrency!r}"
            )
        self._parallels.append(
            ParallelDecl(map_over=map_over, body=body, join=join, max_concurrency=max_concurrency)
        )
        return self

    # ── Cross-replicate reducer ───────────────────────────────────────────

    def reduce(
        self,
        *,
        over: str = "replicate",
    ) -> Callable[[Callable[..., TaskOutput]], Callable[..., TaskOutput]]:
        """Register a cross-replicate reducer (not part of the DAG)."""

        def decorator(fn: Callable[..., TaskOutput]) -> Callable[..., TaskOutput]:
            if self._reducer is not None:
                raise ValueError(
                    f"WorkflowBuilder {self._name!r}: reducer already registered "
                    f"({_callable_name(self._reducer[1])!r})"
                )
            self._reducer = (over, fn)
            return fn

        return decorator

    def _record_decorator_edges(
        self,
        task_name: str,
        *,
        routes: Mapping[str, str] | None,
        next_: str | None,
    ) -> None:
        if next_ is not None:
            self._control_edges.append((task_name, next_))
        if routes is not None:
            for lbl, target in routes.items():
                self._branch_edges.append((task_name, lbl, target))

    # ── Compile ───────────────────────────────────────────────────────────

    def build(self) -> Workflow:
        """Compile registered tasks into a frozen :class:`Workflow`.

        Runs the CFG compiler eagerly so validation errors surface here
        rather than at first execute.
        """
        tasks = list(self._tasks)
        spec = Workflow(
            name=self._name,
            workflow_id=_stable_workflow_id(self._name, tasks),
            tasks=tasks,
            mode=self._mode,
            version=self._version,
            entries=tuple(self._entries),
            control_edges=tuple(self._control_edges),
            branch_edges=tuple(self._branch_edges),
            loops=tuple(self._loops),
            parallels=tuple(self._parallels),
        )
        spec._reducer = self._reducer
        from ._pydantic_graph.compiler import WorkflowGraphCompiler

        WorkflowGraphCompiler().compile(spec)
        return spec
