"""Workflow specification — unified OOP API.

Define a workflow by instantiating :class:`WorkflowBuilder` and registering tasks
through its methods, then call :meth:`WorkflowBuilder.build` to produce the
frozen :class:`Workflow` (compiled, executable, content-addressed)::

    wf = WorkflowBuilder(name="pipeline")


    @wf.task
    async def fetch(ctx: TaskContext) -> FetchResult: ...


    @wf.task(depends_on=["fetch"])
    async def validate(ctx: TaskContext) -> ValidateResult: ...


    # OOP style — add Task / Actor instances
    wf.add(ProcessTask(), depends_on=["validate"])

    # Control flow primitives
    wf.control(src="validate", to="emit")
    wf.branch(src="emit", routes={"ok": "publish", "fail": "rollback"})
    wf.loop(body=["compute"], until="check_done", max_iters=100)
    wf.parallel(map_over="items", body="process", join="reduce", max_concurrency=8)

    spec = wf.build()
    result = await spec.execute(run=run)
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING, ClassVar, Protocol

from ._graph_decl import (
    LoopDecl,
    ParallelDecl,
    TaskRegistration,
    _BoundaryStubTask,
)
from ._helpers import _callable_code_hash, _ir_object_list, _require_str, _stable_workflow_id
from .protocols import (
    JSONMapping,
    JSONValue,
    RunContextLike,
    Streamable,
    TaskOutput,
    UserDeps,
)
from .types import WorkflowExecution, WorkflowResult

if TYPE_CHECKING:
    from ._pydantic_graph.runtime import GraphWorkflowRuntime
    from .registry import TaskTypeRegistry
    from .version import WorkflowVersion


class _ExperimentLike(Protocol):
    """Duck-typed handle to anything with a stable string ``id``."""

    @property
    def id(self) -> str: ...


class Workflow:
    """Compiled, executable workflow specification.

    Produced by :meth:`WorkflowBuilder.build`. Frozen — task topology and
    content-hash :attr:`workflow_id` are stable for the lifetime of the
    instance.
    """

    _bindings_registry: ClassVar[dict[str, Workflow]] = {}

    def __init__(
        self,
        name: str,
        workflow_id: str,
        tasks: list[TaskRegistration],
        mode: str = "batch",
        version: str = "0",
        *,
        entries: tuple[str, ...] = (),
        control_edges: tuple[tuple[str, str], ...] = (),
        branch_edges: tuple[tuple[str, str, str], ...] = (),
        loops: tuple[LoopDecl, ...] = (),
        parallels: tuple[ParallelDecl, ...] = (),
    ) -> None:
        self.name = name
        self.workflow_id = workflow_id
        self.version_label = version
        self._tasks = tasks
        self._mode = mode
        self._entries = entries
        self._control_edges = control_edges
        self._branch_edges = branch_edges
        self._loops = loops
        self._parallels = parallels
        self._runtime: GraphWorkflowRuntime | None = None
        self._reducer: tuple[str, Callable[..., TaskOutput]] | None = None

    # ── Experiment binding ────────────────────────────────────────────────

    def bind_to(self, experiment: _ExperimentLike) -> None:
        """Bind this workflow to *experiment* in the current process."""
        exp_id = getattr(experiment, "id", None)
        if not isinstance(exp_id, str) or not exp_id:
            raise ValueError(
                f"bind_to expects an experiment with a non-empty string `id`; got {experiment!r}"
            )
        Workflow._bindings_registry[exp_id] = self

    def unbind_from(self, experiment: _ExperimentLike) -> bool:
        """Drop the binding for *experiment*. Returns True iff one existed."""
        exp_id = getattr(experiment, "id", None)
        if not isinstance(exp_id, str) or not exp_id:
            return False
        return Workflow._bindings_registry.pop(exp_id, None) is not None

    def is_bound_to(self, experiment: _ExperimentLike) -> bool:
        """Return True iff this exact spec is the one bound to *experiment*."""
        exp_id = getattr(experiment, "id", None)
        if not isinstance(exp_id, str) or not exp_id:
            return False
        return Workflow._bindings_registry.get(exp_id) is self

    @classmethod
    def for_experiment(cls, experiment: _ExperimentLike) -> Workflow | None:
        """Return the spec bound to *experiment* in this process, or None."""
        exp_id = getattr(experiment, "id", None)
        if not isinstance(exp_id, str) or not exp_id:
            return None
        return cls._bindings_registry.get(exp_id)

    @classmethod
    def _reset_registry(cls) -> None:
        """Clear every binding. For test isolation only."""
        cls._bindings_registry.clear()

    def _get_runtime(self) -> GraphWorkflowRuntime:
        if self._runtime is None:
            from ._pydantic_graph.runtime import GraphWorkflowRuntime

            self._runtime = GraphWorkflowRuntime()
        return self._runtime

    # ── Boundary introspection ────────────────────────────────────────────

    @property
    def boundary_names(self) -> frozenset[str]:
        """Names of boundary-stub tasks on this spec (empty for full pipelines)."""
        return frozenset(
            t.name for t in self._tasks if isinstance(t.fn_or_class, _BoundaryStubTask)
        )

    @property
    def non_boundary_names(self) -> frozenset[str]:
        """Names of runnable (non-stub) tasks — complement of boundary_names."""
        return frozenset(t.name for t in self._tasks) - self.boundary_names

    # ── Cross-replicate reducer ───────────────────────────────────────────

    @property
    def reducer_dimension(self) -> str | None:
        """Dimension declared on ``@wf.reduce(over=...)``; None if no reducer."""
        return self._reducer[0] if self._reducer is not None else None

    def run_reducer(self, replicate_outputs: list[TaskOutput]) -> TaskOutput:
        """Invoke the registered ``@wf.reduce`` on a list of replicate outputs."""
        if self._reducer is None:
            raise LookupError(
                f"Workflow {self.name!r}: no reducer registered (call @wf.reduce(over=...))"
            )
        return self._reducer[1](replicate_outputs)

    # ── Execution ─────────────────────────────────────────────────────────

    async def execute(
        self,
        *,
        run_context: RunContextLike | None = None,
        run_dir: str | None = None,
        config: JSONMapping | None = None,
        deps: UserDeps = None,
        execution_id: str | None = None,
        seed_outputs: Mapping[str, TaskOutput] | None = None,
    ) -> WorkflowResult:
        """Run the workflow to completion and return the result.

        Args:
            run_context: Duck-typed run-context payload.
            run_dir: Path for ``executions/<id>/``. Mutually exclusive with run_context.
            config: JSON-shaped mapping exposed as ``ctx.config``.
            deps: User dependencies forwarded to ``TaskContext.deps``.
            execution_id: Optional explicit ID (defaults to fresh ``exec-<…>``).
            seed_outputs: Pre-populated ``task_name → output`` map for
                subgraph boundary seeding.
        """
        return await self._get_runtime().execute(
            self,
            run_context=run_context,
            run_dir=run_dir,
            config=config,
            deps=deps,
            execution_id=execution_id,
            seed_outputs=seed_outputs,
        )

    async def start(
        self,
        *,
        run_context: RunContextLike | None = None,
        run_dir: str | None = None,
        config: JSONMapping | None = None,
        deps: UserDeps = None,
        execution_id: str | None = None,
        seed_outputs: Mapping[str, TaskOutput] | None = None,
    ) -> WorkflowExecution:
        """Start the workflow asynchronously and return a handle."""
        return await self._get_runtime().start(
            self,
            run_context=run_context,
            run_dir=run_dir,
            config=config,
            deps=deps,
            execution_id=execution_id,
            seed_outputs=seed_outputs,
        )

    async def run_on(
        self,
        experiment: _ExperimentLike,
        *,
        parameters: Mapping[str, JSONValue] | None = None,
        deps: UserDeps = None,
        profile_config: object | None = None,
        config: JSONMapping | None = None,
    ) -> WorkflowResult:
        """Build a fresh Run, enter its context, execute, and return the result.

        Does NOT call bind_to; call ``self.bind_to(experiment)`` separately
        if you need the workflow recoverable after process restart.
        """
        params_dict = dict(parameters) if parameters is not None else None
        run = experiment.add_run(parameters=params_dict)  # type: ignore[attr-defined]
        with run.start(profile_config=profile_config) as run_ctx:
            result = await self.execute(run_context=run_ctx, config=config, deps=deps)
        if result.status != "completed":
            err = run.metadata.error
            err_msg = (
                f"workflow {self.name!r} ended with status {result.status!r}: "
                f"{err.type}: {err.message}"
                if err is not None
                else f"workflow {self.name!r} ended with status {result.status!r}"
            )
            raise RuntimeError(err_msg)
        return result

    # ── Versioning ────────────────────────────────────────────────────────

    def version(self) -> WorkflowVersion:
        """Build the immutable WorkflowVersion for this spec."""
        from .version import TaskTopologyEntry, WorkflowVersion

        topo: list[TaskTopologyEntry] = []
        for t in self._tasks:
            topo.append(
                TaskTopologyEntry(
                    name=t.name,
                    qualname=type(t.fn_or_class).__qualname__,
                    depends_on=tuple(t.depends_on),
                    code_hash=_callable_code_hash(t.fn_or_class),
                )
            )
        return WorkflowVersion(
            workflow_id=self.workflow_id,
            version=self.version_label,
            name=self.name,
            topology=tuple(topo),
        )

    # ── IR serialization ─────────────────────────────────────────────────

    def to_dict(self) -> dict[str, JSONValue]:
        """Serialize this spec to the JSON IR shape (see ``schema/workflow.json``)."""
        unslugged = [t.name for t in self._tasks if t.task_type is None]
        if unslugged:
            raise ValueError(
                "Cannot serialize workflow to IR: the following tasks have no "
                f"task_type slug: {unslugged}. Use `WorkflowBuilder.add(..., task_type=...)` "
                "or build the spec from IR via Workflow.from_dict()."
            )
        if self._control_edges or self._branch_edges or self._entries:
            raise ValueError(
                "Cannot serialize workflow to IR: spec contains a control edge / "
                "explicit entry (wf.control / wf.branch / wf.entry)."
            )
        task_configs: list[JSONValue] = [
            {
                "task_id": t.name,
                "task_type": t.task_type,
                "config": dict(t.config) if t.config else {},
                "status": "pending",
            }
            for t in self._tasks
        ]
        links: list[JSONValue] = [
            {"source": dep, "target": t.name, "mapping": {}, "status": "pending"}
            for t in self._tasks
            for dep in t.depends_on
        ]
        metadata: dict[str, JSONValue] = {
            "label": None,
            "description": None,
            "tags": [],
            "custom": {},
        }
        return {
            "workflow_id": f"workflow_{self.workflow_id[:8]}",
            "name": self.name,
            "task_configs": task_configs,
            "links": links,
            "metadata": metadata,
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, JSONValue],
        *,
        registry: TaskTypeRegistry | None = None,
    ) -> Workflow:
        """Build a Workflow from JSON IR."""
        if registry is None:
            from .registry import default_registry

            registry = default_registry

        links_raw = _ir_object_list(data.get("links"))
        task_configs_raw = _ir_object_list(data.get("task_configs"))

        deps_by_target: dict[str, list[str]] = {}
        for link in links_raw:
            target = _require_str(link, "target")
            source = _require_str(link, "source")
            deps_by_target.setdefault(target, []).append(source)

        tasks: list[TaskRegistration] = []
        for tc in task_configs_raw:
            slug = _require_str(tc, "task_type")
            task_id = _require_str(tc, "task_id")
            cfg_raw = tc.get("config")
            config: dict[str, JSONValue] = dict(cfg_raw) if isinstance(cfg_raw, dict) else {}
            factory = registry.get(slug)
            instance = factory(config)
            tasks.append(
                TaskRegistration(
                    name=task_id,
                    fn_or_class=instance,
                    depends_on=deps_by_target.get(task_id, []),
                    is_actor=isinstance(instance, Streamable),
                    task_type=slug,
                    config=config,
                )
            )

        known = {t.name for t in tasks}
        for link in links_raw:
            for endpoint in (_require_str(link, "source"), _require_str(link, "target")):
                if endpoint not in known:
                    raise ValueError(
                        f"Link references unknown task_id {endpoint!r}; known: {sorted(known)}"
                    )

        name_raw = data.get("name")
        name = name_raw if isinstance(name_raw, str) else ""
        return cls(
            name=name,
            workflow_id=_stable_workflow_id(name, tasks),
            tasks=tasks,
            mode="batch",
        )

    # ── Subgraph ─────────────────────────────────────────────────────────

    def subgraph(
        self,
        start_nodes: Iterable[str],
        *,
        include_downstream: bool = False,
    ) -> Workflow:
        """Return a frozen Workflow over a subset of this spec's nodes.

        For each boundary upstream (a task referenced by a selected node's
        ``depends_on`` but itself outside the selection), a ``_BoundaryStubTask``
        is registered. The caller must supply the upstream value via
        ``Workflow.execute(seed_outputs=...)``.
        """
        selection = list(start_nodes)
        if not selection:
            raise ValueError("Workflow.subgraph: start_nodes must not be empty")
        registered = {t.name: t for t in self._tasks}
        unknown = [name for name in selection if name not in registered]
        if unknown:
            raise ValueError(
                f"Workflow.subgraph: unknown task name(s) {unknown!r}; "
                f"registered tasks: {sorted(registered)}"
            )

        chosen: set[str] = set(selection)
        if include_downstream:
            forward: dict[str, list[str]] = {name: [] for name in registered}
            for t in self._tasks:
                for dep in t.depends_on:
                    forward.setdefault(dep, []).append(t.name)
            frontier = list(selection)
            while frontier:
                node = frontier.pop()
                for child in forward.get(node, ()):
                    if child not in chosen:
                        chosen.add(child)
                        frontier.append(child)

        new_tasks: list[TaskRegistration] = []
        boundary_upstreams: list[str] = []
        seen_boundaries: set[str] = set()
        for t in self._tasks:
            if t.name not in chosen:
                continue
            new_tasks.append(
                TaskRegistration(
                    name=t.name,
                    fn_or_class=t.fn_or_class,
                    depends_on=list(t.depends_on),
                    is_actor=t.is_actor,
                    remote=t.remote,
                    task_type=t.task_type,
                    config=t.config,
                    dependent_params=t.dependent_params,
                )
            )
            for dep in t.depends_on:
                if dep not in chosen and dep not in seen_boundaries:
                    seen_boundaries.add(dep)
                    boundary_upstreams.append(dep)

        stub_tasks: list[TaskRegistration] = [
            TaskRegistration(
                name=bname,
                fn_or_class=_BoundaryStubTask(bname),
                depends_on=[],
            )
            for bname in boundary_upstreams
        ]
        new_tasks = stub_tasks + new_tasks

        return Workflow(
            name=self.name,
            workflow_id=_stable_workflow_id(self.name, new_tasks),
            tasks=new_tasks,
            mode=self._mode,
            version=self.version_label,
        )
