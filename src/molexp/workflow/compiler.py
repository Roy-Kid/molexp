"""WorkflowCompiler — build + compile in one pass, emitting a CompiledWorkflow.

Merges the old ``WorkflowBuilder`` (fluent authoring) and the internal CFG
``WorkflowGraphCompiler`` (lowering) into a single object. The fluent API is
unchanged in spirit — ``add`` / ``task`` / ``actor`` / ``entry`` / ``control``
/ ``branch`` / ``loop`` / ``parallel`` / ``reduce`` — but instead of
``.build() -> Workflow`` it exposes ``.compile(*, experiment=None,
registry=None) -> CompiledWorkflow``, which lowers the registrations exactly
once, computes per-task snapshots + the workflow version, performs experiment
binding, and returns the single frozen artifact.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

from ._graph_decl import (
    DependentParamsFn,
    LoopDecl,
    ParallelDecl,
    TaskRegistration,
    WorkflowTopology,
)
from ._helpers import _callable_name, _stable_workflow_id, _to_snake_case
from ._pydantic_graph.compiler import WorkflowGraphCompiler
from .binding import default_binding_registry
from .compiled import CompiledWorkflow
from .protocols import Streamable, TaskBody, TaskOutput, UserDeps
from .snapshot import TaskSnapshot
from .version import TaskTopologyEntry, WorkflowVersion

if TYPE_CHECKING:
    from .binding import WorkflowBindingRegistry
    from .compiled import _ExperimentLike

_lowering = WorkflowGraphCompiler()


def compile_registrations(
    *,
    name: str,
    version_label: str,
    tasks: list[TaskRegistration],
    mode: str = "batch",
    entries: tuple[str, ...] = (),
    control_edges: tuple[tuple[str, str], ...] = (),
    branch_edges: tuple[tuple[str, str, str], ...] = (),
    loops: tuple[LoopDecl, ...] = (),
    parallels: tuple[ParallelDecl, ...] = (),
    reducer: tuple[str, Callable[..., TaskOutput]] | None = None,
    experiment: _ExperimentLike | None = None,
    registry: WorkflowBindingRegistry | None = None,
) -> CompiledWorkflow:
    """Lower registrations once and assemble the :class:`CompiledWorkflow`.

    Shared by :meth:`WorkflowCompiler.compile` and
    :meth:`CompiledWorkflow.subgraph`. The ``workflow_id`` is computed
    before lowering (so it reflects the authored topology), then the CFG
    lowering runs once (it may inject parallel-join data deps), and the
    per-task snapshots + version are computed from the lowered tasks — the
    version reuses each snapshot's ``code_hash`` so the two code-hashers
    collapse to one.
    """
    # Resolve each task's serialization slug from the type registry. The slug
    # lives with the task *type* (registered via ``@default_registry.register``),
    # not at the ``add()`` call site. Tasks that already carry a slug — the
    # deserialize path (``ir_to_spec`` sets it from the incoming JSON) — keep it.
    from .registry import default_registry

    for t in tasks:
        if t.task_type is None:
            t.task_type = default_registry.slug_for(t.fn_or_class)

    workflow_id = _stable_workflow_id(name, tasks)
    topology = WorkflowTopology(
        name=name,
        tasks=tasks,
        entries=entries,
        control_edges=control_edges,
        branch_edges=branch_edges,
        loops=loops,
        parallels=parallels,
    )
    graph = _lowering.compile(topology)

    snapshots: dict[str, TaskSnapshot] = {
        t.name: TaskSnapshot.from_task_body(t.name, t.fn_or_class) for t in tasks
    }
    version = WorkflowVersion(
        workflow_id=workflow_id,
        version=version_label,
        name=name,
        topology=tuple(
            TaskTopologyEntry(
                name=t.name,
                qualname=type(t.fn_or_class).__qualname__,
                depends_on=tuple(t.depends_on),
                code_hash=snapshots[t.name].code_hash,
            )
            for t in tasks
        ),
    )

    compiled = CompiledWorkflow(
        name=name,
        workflow_id=workflow_id,
        version_label=version_label,
        tasks=tasks,
        graph=graph,
        snapshots=snapshots,
        version=version,
        mode=mode,
        entries=entries,
        control_edges=control_edges,
        branch_edges=branch_edges,
        loops=loops,
        parallels=parallels,
        reducer=reducer,
    )
    if experiment is not None:
        reg = registry if registry is not None else default_binding_registry
        compiled.binding = reg.bind(experiment, compiled)
    return compiled


class WorkflowCompiler:
    """Fluent workflow authoring + one-pass compile (decorator and OOP styles).

    Instantiate once, register tasks via the decorators (:meth:`task`,
    :meth:`actor`) or :meth:`add`, wire control flow with :meth:`control` /
    :meth:`branch` / :meth:`loop` / :meth:`parallel`, then call
    :meth:`compile` to produce a :class:`CompiledWorkflow`.
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
            raise TypeError("WorkflowCompiler.task: routes= and next_= are mutually exclusive")

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
            raise TypeError("WorkflowCompiler.actor: routes= and next_= are mutually exclusive.")

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
        routes: Mapping[str, str] | None = None,
        next_: str | None = None,
        dependent_params: DependentParamsFn | None = None,
    ) -> WorkflowCompiler:
        """Register a Task / Actor instance (or any Runnable/Streamable).

        The serialization slug is **not** an argument here: it is a property of
        the task *type*, declared once via
        :meth:`TaskTypeRegistry.register` / ``@default_registry.register("slug")``
        and resolved automatically at :meth:`compile` time. A task's build-time
        config is **not** declared here either — it is the task instance's own
        ``__init__`` arguments (captured automatically; see
        :func:`~molexp.workflow.snapshot.task_config_of`), which the cache and IR
        both key on. Returns ``self`` to support chaining.
        """
        if routes is not None and next_ is not None:
            raise TypeError("WorkflowCompiler.add: routes= and next_= are mutually exclusive.")

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
                dependent_params=dependent_params,
            )
        )
        self._record_decorator_edges(task_name, routes=routes, next_=next_)
        return self

    # ── Control-flow declarations ─────────────────────────────────────────

    def entry(self, name: str) -> WorkflowCompiler:
        """Declare *name* as a workflow entry point. Multiple calls = multi-entry."""
        if name in self._entries:
            raise ValueError(
                f"WorkflowCompiler {self._name!r}: entry {name!r} declared multiple times"
            )
        self._entries.append(name)
        return self

    def control(self, src: str, to: str) -> WorkflowCompiler:
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
    ) -> WorkflowCompiler:
        """Declare branch (label-routed) control edges on *src*.

        Two forms: ``wf.branch("src", "label", "target")`` or
        ``wf.branch("src", routes={"l1": "t1", "l2": "t2"})``.
        """
        if routes is not None:
            if label is not None or to is not None:
                raise TypeError(
                    "WorkflowCompiler.branch: pass either positional (src, label, to) "
                    "or keyword routes={...}, not both."
                )
            for lbl, target in routes.items():
                self._branch_edges.append((src, lbl, target))
            return self
        if label is None or to is None:
            raise TypeError(
                "WorkflowCompiler.branch: pass (src, label, to) or routes={...}; "
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
    ) -> WorkflowCompiler:
        """Declare a loop: ``body`` runs repeatedly until ``until`` exits.

        ``until`` returns ``Next("continue")`` to loop or ``Next("exit")`` to
        proceed to ``on_exit`` (default: terminate).
        """
        if not body:
            raise ValueError(
                f"WorkflowCompiler.loop: body must contain at least one task name; got {body!r}"
            )
        if max_iters < 1:
            raise ValueError(f"WorkflowCompiler.loop: max_iters must be >= 1; got {max_iters!r}")
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
    ) -> WorkflowCompiler:
        """Declare parallel fan-out: run *body* once per element of *map_over* output."""
        if max_concurrency < 1:
            raise ValueError(
                f"WorkflowCompiler.parallel: max_concurrency must be >= 1; got {max_concurrency!r}"
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
                    f"WorkflowCompiler {self._name!r}: reducer already registered "
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

    def compile(
        self,
        *,
        experiment: _ExperimentLike | None = None,
        registry: WorkflowBindingRegistry | None = None,
    ) -> CompiledWorkflow:
        """Lower the registrations and emit the :class:`CompiledWorkflow`.

        Validation (cycles, edge shape, reachability) surfaces here. When
        ``experiment`` is given, the artifact is bound into ``registry`` (or
        :data:`~molexp.workflow.binding.default_binding_registry`) so
        ``registry.for_experiment(experiment) is compiled``.
        """
        return compile_registrations(
            name=self._name,
            version_label=self._version,
            tasks=list(self._tasks),
            mode=self._mode,
            entries=tuple(self._entries),
            control_edges=tuple(self._control_edges),
            branch_edges=tuple(self._branch_edges),
            loops=tuple(self._loops),
            parallels=tuple(self._parallels),
            reducer=self._reducer,
            experiment=experiment,
            registry=registry,
        )


__all__ = ["WorkflowCompiler", "compile_registrations"]
