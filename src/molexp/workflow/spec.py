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

    # Control flow primitives (spec 04 / 05)
    wf.control(src="validate", to="emit")
    wf.branch(src="emit", routes={"ok": "publish", "fail": "rollback"})
    wf.loop(body=["compute"], until="check_done", max_iters=100)
    wf.parallel(map_over="items", body="process", join="reduce", max_concurrency=8)

    spec = wf.build()
    result = await spec.execute(run=run)
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING, ClassVar, Protocol

from .protocols import (
    JSONMapping,
    JSONValue,
    RunContextLike,
    Streamable,
    TaskBody,
    TaskOutput,
    UpstreamViewLike,
    UserDeps,
)
from .types import WorkflowExecution, WorkflowResult

if TYPE_CHECKING:
    from ._pydantic_graph.runtime import GraphWorkflowRuntime
    from .registry import TaskTypeRegistry
    from .version import WorkflowVersion

# Callback shape for ``dependent_params`` — receives a mapping of upstream
# task name → :class:`UpstreamViewLike` and returns an overlay applied on
# top of the task's base config.
type DependentParamsFn = Callable[[Mapping[str, UpstreamViewLike]], JSONMapping]


def _callable_name(f: Callable, fallback: str = "anonymous") -> str:
    """Return a Python function's ``__name__`` if present, else ``fallback``.

    Type-checked codepaths annotate decorator targets as ``Callable``, which
    static checkers cannot prove has ``__name__``. In practice every decorated
    target is a function, so ``getattr`` is sufficient and keeps the annotation
    free of more specific protocols.
    """
    return getattr(f, "__name__", None) or fallback


class LoopDecl:
    """User-declared loop topology compiled by the CFG compiler.

    Spec 04 §4 ``wf.loop`` primitive. The compiler synthesises:

    * a control edge ``body[-1] → until`` (if not already declared), and
    * branch edges on ``until``: ``{"continue": body[0], "exit": on_exit}``.

    ``max_iters`` is enforced by the runtime: when ``until`` dispatches
    ``Next("continue")`` ``max_iters`` times, the runtime forces
    ``Next("exit")`` and emits :class:`LoopMaxItersExceeded`.
    """

    __slots__ = ("body", "max_iters", "on_exit", "until")

    def __init__(
        self,
        body: tuple[str, ...],
        until: str,
        max_iters: int,
        on_exit: str,
    ) -> None:
        self.body = body
        self.until = until
        self.max_iters = max_iters
        self.on_exit = on_exit


class ParallelDecl:
    """User-declared parallel fan-out / map-reduce topology.

    Spec 05 §4 ``wf.parallel`` primitive. The compiler synthesises two
    unconditional control edges per decl:

    * ``map_over → body`` (fan-out trigger), and
    * ``body → join`` (fan-in trigger).

    The runtime ``WorkflowStep`` recognises ``body`` as a parallel
    head, reads ``state.results[map_over]``, and invokes the body's
    ``run`` once per element under ``Semaphore(max_concurrency)``.
    Per-element exceptions are captured into
    :class:`~molexp.workflow.ParallelExecutionError`.
    """

    __slots__ = ("body", "join", "map_over", "max_concurrency")

    def __init__(
        self,
        map_over: str,
        body: str,
        join: str,
        max_concurrency: int,
    ) -> None:
        self.map_over = map_over
        self.body = body
        self.join = join
        self.max_concurrency = max_concurrency


class _BoundaryStubTask:
    """Placeholder task registered by :meth:`Workflow.subgraph` for boundary
    upstreams (tasks referenced by ``depends_on`` but excluded from the
    selection).

    Never runs in practice — :meth:`Workflow.execute(seed_outputs=...)`
    must supply the upstream value, and the runtime's seed-skipping
    logic filters the stub out of the frontier before invocation. If
    the body is ever invoked it means the caller forgot to seed the
    boundary value, so the body raises a clear :class:`RuntimeError`.
    """

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    async def execute(self, ctx: object) -> object:
        del ctx
        raise RuntimeError(
            f"Subgraph boundary stub {self.name!r} was invoked. This means "
            f"Workflow.execute() was called without seed_outputs[{self.name!r}]. "
            "Provide a seeded value for every boundary upstream of the subgraph."
        )


class TaskRegistration:
    """Internal record of a registered task or actor.

    ``task_type`` and ``config`` are populated when the task originates
    from a registry slug (either via :meth:`WorkflowBuilder.add` with
    ``task_type=...`` or via :meth:`Workflow.from_dict`). They are
    required for :meth:`Workflow.to_dict` to produce IR JSON.
    """

    __slots__ = (
        "config",
        "dependent_params",
        "depends_on",
        "fn_or_class",
        "is_actor",
        "name",
        "remote",
        "task_type",
    )

    def __init__(
        self,
        name: str,
        fn_or_class: TaskBody,
        depends_on: list[str],
        is_actor: bool = False,
        remote: UserDeps = None,
        task_type: str | None = None,
        config: JSONMapping | None = None,
        dependent_params: DependentParamsFn | None = None,
    ) -> None:
        self.name = name
        self.fn_or_class = fn_or_class
        self.depends_on = depends_on
        self.is_actor = is_actor
        self.remote = remote
        self.task_type = task_type
        self.config = dict(config) if config else None
        self.dependent_params = dependent_params


# ── Workflow ────────────────────────────────────────────────────────────


class _ExperimentLike(Protocol):
    """Duck-typed handle to anything with a stable string ``id``.

    Workspace's :class:`molexp.workspace.Experiment` satisfies this; tests
    that construct a stand-in object with an ``id`` attribute also work.
    The workflow layer never imports ``molexp.workspace.Experiment``
    directly — keeping the dependency direction one-way.
    """

    @property
    def id(self) -> str: ...


class Workflow:
    """Compiled, executable workflow specification.

    Produced by :meth:`WorkflowBuilder.build`. Frozen — task topology and
    content-hash :attr:`workflow_id` are stable for the lifetime of the
    instance. Carries a process-local registry mapping
    ``experiment.id → Workflow`` so that downstream code (CLI, server,
    cluster workers) can look up the spec bound to a given experiment
    without threading it through every call.
    """

    # Process-local registry: experiment.id → Workflow. Survives across
    # function boundaries within a single Python process; does not survive
    # process restart on its own. Cluster workers re-establish bindings by
    # re-importing the user script, which calls :meth:`bind_to` again.
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
        # Raw control-flow declarations from the WorkflowBuilder.
        # Compiled into ``out_edges`` + ``entry_frontier`` by the CFG compiler.
        self._entries = entries
        self._control_edges = control_edges
        self._branch_edges = branch_edges
        self._loops = loops
        self._parallels = parallels
        self._runtime: GraphWorkflowRuntime | None = None
        # Cross-replicate reducer (set by WorkflowBuilder.build()).
        self._reducer: tuple[str, Callable[..., TaskOutput]] | None = None

    # ── Experiment binding (process-local registry) ─────────────────────

    def bind_to(self, experiment: _ExperimentLike) -> None:
        """Bind this workflow to *experiment* in the current process.

        Re-binding the same experiment overwrites the previous spec —
        the caller controls overwrite semantics. The typical
        cluster-worker flow re-runs the user script and rebinds cleanly.

        Args:
            experiment: Anything with a stable string ``id``. In
                production this is :class:`molexp.workspace.Experiment`.

        Raises:
            ValueError: If *experiment* has no string ``id``.
        """
        exp_id = getattr(experiment, "id", None)
        if not isinstance(exp_id, str) or not exp_id:
            raise ValueError(
                f"bind_to expects an experiment with a non-empty string `id`; got {experiment!r}"
            )
        Workflow._bindings_registry[exp_id] = self

    def unbind_from(self, experiment: _ExperimentLike) -> bool:
        """Drop the binding for *experiment*. Returns ``True`` iff one existed."""
        exp_id = getattr(experiment, "id", None)
        if not isinstance(exp_id, str) or not exp_id:
            return False
        # Only unbind if THIS spec is the bound one (defensive; matches old
        # clear_workflow which removed regardless).
        return Workflow._bindings_registry.pop(exp_id, None) is not None

    def is_bound_to(self, experiment: _ExperimentLike) -> bool:
        """Return ``True`` iff this exact spec is the one bound to *experiment*."""
        exp_id = getattr(experiment, "id", None)
        if not isinstance(exp_id, str) or not exp_id:
            return False
        return Workflow._bindings_registry.get(exp_id) is self

    @classmethod
    def for_experiment(cls, experiment: _ExperimentLike) -> Workflow | None:
        """Return the spec bound to *experiment* in this process, or ``None``."""
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

    # ── Subgraph boundary introspection ──────────────────────────────────

    @property
    def boundary_names(self) -> frozenset[str]:
        """Names of boundary-stub tasks on this spec.

        Boundary stubs are :class:`_BoundaryStubTask` placeholders registered
        by :meth:`subgraph` for upstream dependencies that lie outside the
        selected subgraph. They never execute; :meth:`execute` seeds their
        values via ``seed_outputs``.

        Returns an empty frozenset when this spec is a full pipeline (no
        subgraph boundaries exist).
        """
        return frozenset(
            t.name for t in self._tasks if isinstance(t.fn_or_class, _BoundaryStubTask)
        )

    @property
    def non_boundary_names(self) -> frozenset[str]:
        """Names of runnable (non-stub) tasks on this spec.

        Complement of :attr:`boundary_names` — every task whose ``fn_or_class``
        is not a :class:`_BoundaryStubTask`.
        """
        return frozenset(t.name for t in self._tasks) - self.boundary_names

    # ── Cross-replicate reducer ─────────────────────────────────────────

    @property
    def reducer_dimension(self) -> str | None:
        """Dimension declared on ``@wf.reduce(over=...)``; ``None`` if no reducer."""
        return self._reducer[0] if self._reducer is not None else None

    def run_reducer(self, replicate_outputs: list[TaskOutput]) -> TaskOutput:
        """Invoke the registered ``@wf.reduce`` on a list of replicate outputs.

        Raises ``LookupError`` if no reducer was registered. Callers attach
        the return value to the experiment-scope asset library; this method
        does not write anywhere itself.
        """
        if self._reducer is None:
            raise LookupError(
                f"Workflow {self.name!r}: no reducer registered (call @wf.reduce(over=...))"
            )
        return self._reducer[1](replicate_outputs)

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
            run_context: Opaque duck-typed run-context payload (anything
                exposing ``.work_dir`` / ``.config`` / ``.run`` is accepted;
                forwarded as-is to ``TaskContext.run_context``).
            run_dir: Optional path under which ``executions/<id>/`` is
                created. Mutually exclusive with *run_context* (which
                supplies its own ``.work_dir``).
            config: JSON-shaped mapping exposed as ``ctx.config``.
                When *run_context* is passed, the context's ``.config``
                takes precedence.
            deps: Optional user dependencies forwarded to ``TaskContext.deps``.
            execution_id: Optional explicit ID for the execution
                (defaults to a fresh ``exec-<…>`` derived from run_id).
            seed_outputs: Optional pre-populated ``task_name → output``
                map. Each named task is treated as already-completed:
                the runtime injects the value into the results dict and
                marks the task ``completed`` so downstream nodes
                consume the seed via ``ctx.inputs`` without re-running
                the task body. Used by the PlanMode review→repair loop
                to skip boundary-upstream re-execution after
                :meth:`subgraph`. Unknown task names raise
                :class:`ValueError` before any task body runs.
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
        """Start the workflow asynchronously and return a handle.

        See :meth:`execute` for ``seed_outputs`` semantics.
        """
        return await self._get_runtime().start(
            self,
            run_context=run_context,
            run_dir=run_dir,
            config=config,
            deps=deps,
            execution_id=execution_id,
            seed_outputs=seed_outputs,
        )

    # ── Happy-path one-liner ────────────────────────────────────────────

    async def run_on(
        self,
        experiment: _ExperimentLike,
        *,
        parameters: Mapping[str, JSONValue] | None = None,
        deps: UserDeps = None,
        profile_config: object | None = None,
        config: JSONMapping | None = None,
    ) -> WorkflowResult:
        """Build a fresh ``Run``, enter its context, execute, and return the result.

        Convenience wrapper for the canonical pattern::

            run = experiment.add_run(parameters=parameters)
            with run.start(profile_config=profile_config) as run_ctx:
                return await self.execute(run_context=run_ctx, deps=deps, config=config)

        Note: ``run_on`` does **not** call :meth:`bind_to`. If the caller
        also wants the workflow recoverable by experiment ID after the
        process restarts (CLI / cluster worker dispatch), call
        ``self.bind_to(experiment)`` separately.

        Args:
            experiment: Workspace ``Experiment`` to attach the new run to.
            parameters: Run-level parameter dict (passed to ``Experiment.Run``).
            deps: User dependencies forwarded to ``TaskContext.deps``.
            profile_config: Active molcfg profile; forwarded to ``Run.start``.
            config: JSON-shaped mapping exposed as ``ctx.config``.

        Returns:
            The :class:`WorkflowResult` from the workflow execution.
        """
        # Imported lazily to keep module import cheap for tests that
        # never construct a Run.
        params_dict: dict[str, JSONValue] | None
        params_dict = dict(parameters) if parameters is not None else None
        run = experiment.add_run(parameters=params_dict)  # type: ignore[attr-defined]
        with run.start(profile_config=profile_config) as run_ctx:
            result = await self.execute(
                run_context=run_ctx,
                config=config,
                deps=deps,
            )
        # Mirror Python's "exceptions propagate" idiom: if the workflow
        # ended in a non-success status, surface it. The original task
        # exception was already recorded on the run's metadata; here we
        # rebuild a representative exception so callers can wrap
        # ``await wf.run_on(...)`` in ``try / except`` naturally.
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

    # ── Versioning ──────────────────────────────────────────────────────

    def version(self) -> WorkflowVersion:
        """Build the immutable :class:`~molexp.workflow.version.WorkflowVersion`.

        Returns:
            A :class:`WorkflowVersion` capturing this spec's
            ``workflow_id``, ``version_label``, ``name``, and topology
            snapshot (one entry per registered task).
        """
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

    # ── IR serialization (JSON workflow IR) ─────────────────────────────

    def to_dict(self) -> dict[str, JSONValue]:
        """Serialize this spec to the JSON IR shape (see ``schema/workflow.json``).

        Every task must have been registered with a ``task_type`` slug
        (passed via :meth:`WorkflowBuilder.add` or set by
        :meth:`Workflow.from_dict`); otherwise this raises
        :class:`ValueError`. Decorator-style functions and ad-hoc Task
        instances added without a slug are not serializable — they live
        only in process memory.
        """
        unslugged = [t.name for t in self._tasks if t.task_type is None]
        if unslugged:
            raise ValueError(
                "Cannot serialize workflow to IR: the following tasks have no "
                f"task_type slug: {unslugged}. Use `WorkflowBuilder.add(..., task_type=...)` "
                "or build the spec from IR via Workflow.from_dict()."
            )

        # Spec 03 — IR currently models data edges only. Workflows that declare
        # control edges (`wf.control` / `wf.branch`) or explicit entries must
        # refuse serialization rather than silently dropping the loop topology.
        if self._control_edges or self._branch_edges or self._entries:
            raise ValueError(
                "Cannot serialize workflow to IR: spec contains a control edge / "
                "explicit entry (wf.control / wf.branch / wf.entry). IR "
                "serialization of cyclic / branch topology is out of scope for "
                "spec 03; keep this workflow in-process or rewrite as a pure DAG."
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
        """Build a :class:`Workflow` from JSON IR.

        Each ``task_configs[]`` entry's ``task_type`` is looked up in
        *registry* (defaults to the module-level
        :data:`~molexp.workflow.registry.default_registry`); the
        registered factory is invoked with the entry's ``config`` to
        produce the runnable task object.

        ``links[]`` are aggregated into per-task ``depends_on`` lists.
        The ``mapping`` field is reserved for richer wiring and is
        currently ignored — the runtime passes upstream outputs directly
        (single value or ``dict[name, value]``).

        The returned spec's ``workflow_id`` is recomputed from the
        topology hash; the IR's ``workflow_id`` field is informational.
        """
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

        # Validate every link target / source is a known task_id
        known = {t.name for t in tasks}
        for link in links_raw:
            for endpoint in (
                _require_str(link, "source"),
                _require_str(link, "target"),
            ):
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

    # ── Partial-rerun primitive ─────────────────────────────────────────

    def subgraph(
        self,
        start_nodes: Iterable[str],
        *,
        include_downstream: bool = False,
    ) -> Workflow:
        """Return a frozen :class:`Workflow` over a subset of this spec's nodes.

        Used by the PlanMode review→repair loop to construct a partial-
        rerun spec — e.g. "re-run only ``DraftImplementationPlan`` and
        everything downstream from it" — without invalidating the rest
        of the materialized artifacts.

        For each *boundary upstream* (a task referenced by a selected
        node's ``depends_on`` but itself outside the selection), the
        returned spec registers a :class:`_BoundaryStubTask` placeholder
        with that upstream's name. The stub's body raises if invoked, so
        the caller is forced to supply the upstream value via
        ``Workflow.execute(seed_outputs=...)``; the runtime's seed-
        skipping logic ensures the stub never actually runs when seeded.

        The returned spec is a fresh frozen :class:`Workflow` with a
        recomputed ``workflow_id`` reflecting the new topology. Control
        edges (``wf.control`` / ``wf.branch`` / ``wf.loop`` / explicit
        entries) are intentionally **not** carried over: the use case is
        partial-rerun of a pure data DAG and re-applying the original
        control wiring would re-introduce loops the caller is trying to
        break out of.

        Args:
            start_nodes: Names of tasks to include. Must be non-empty
                and every name must already be registered on the spec.
            include_downstream: When ``True``, every transitively
                reachable downstream task (forward edges in the data
                DAG) is added to the selection.

        Returns:
            New :class:`Workflow` whose ``_tasks`` contains the
            selection (in the original registration order) plus a
            :class:`_BoundaryStubTask` for each boundary upstream.
            ``depends_on`` is preserved intact so that downstream tasks
            still observe the boundary value via ``ctx.inputs`` once
            ``seed_outputs`` is provided.

        Raises:
            ValueError: If *start_nodes* is empty, or contains a name
                that is not a registered task on this spec.
        """
        # Materialize once so we can iterate twice + report a stable order.
        selection = list(start_nodes)
        if not selection:
            raise ValueError(
                "Workflow.subgraph: start_nodes must not be empty; provide at "
                "least one registered task name to include in the partial spec."
            )
        registered = {t.name: t for t in self._tasks}
        unknown = [name for name in selection if name not in registered]
        if unknown:
            raise ValueError(
                f"Workflow.subgraph: unknown task name(s) {unknown!r}; "
                f"registered tasks: {sorted(registered)}"
            )

        chosen: set[str] = set(selection)
        if include_downstream:
            # Forward-closure walk through the data-DAG (depends_on edges).
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

        # Preserve registration order so deterministic iteration (and
        # the workflow_id hash) matches user expectations. Walk the
        # original task list and either copy the registration verbatim
        # (when in selection) or skip (otherwise).
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

        # Register a stub for each boundary upstream so the compiler's
        # depends_on check finds the name. Insert stubs at the front so
        # they appear before any task that depends on them; this keeps
        # the topological frontier computation deterministic.
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


# ── WorkflowBuilder (unified OOP API) ──────────────────────────────────────────────


class WorkflowBuilder:
    """OOP workflow definition. Supports decorator and builder styles.

    Instantiate once, then register tasks via the decorators
    (:meth:`task`, :meth:`actor`) or the OOP method :meth:`add`. Wire
    control flow with :meth:`control` / :meth:`branch` / :meth:`loop`
    / :meth:`parallel`. Call :meth:`build` to produce a
    :class:`Workflow`.

    Example (decorator)::

        wf = WorkflowBuilder(name="pipeline")


        @wf.task
        async def fetch(ctx: TaskContext) -> dict: ...

    Example (OOP)::

        wf = WorkflowBuilder(name="pipeline")
        wf.add(FetchTask())
        wf.add(ProcessTask(), depends_on=["fetch"])
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

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return self._name

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def version_label(self) -> str:
        return self._version

    # ── Decorator: function-as-task ─────────────────────────────────────

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

        ``routes={label: target}`` declares branch (label-routed) outgoing
        control edges; the task body must return ``Next(label)`` (see
        spec 03 §3, §5). ``next_=target`` declares a single unconditional
        control edge. The two are mutually exclusive.

        Usage::

            @wf.task
            async def fetch(ctx): ...


            @wf.task(depends_on=["fetch"])
            async def validate(ctx): ...


            @wf.task(routes={"ok": "emit", "fail": "rollback"})
            async def classify(ctx) -> Next: ...


            @wf.task(next_="emit")
            async def normalize(ctx) -> Doc: ...
        """
        if routes is not None and next_ is not None:
            raise TypeError(
                "WorkflowBuilder.task: routes= and next_= are mutually exclusive "
                "(spec 03 §3 — pick branch or unconditional, not both)."
            )

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

        Same ``routes=`` / ``next_=`` semantics as :meth:`task`; the
        actor's terminating ``yield`` may be ``Next(label)`` / ``End()`` /
        ``(value, Next(label))`` / ``(value, End())`` (spec 03 §5).
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

    # ── OOP: register a Task/Actor instance or any Runnable/Streamable ──

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

        Accepts:
        - A :class:`~molexp.workflow.task.Task` / :class:`~molexp.workflow.task.Actor` instance
        - Any object matching the :class:`~molexp.workflow.protocols.Runnable`
          or :class:`~molexp.workflow.protocols.Streamable` protocol
        - A bare callable (treated as a batch task)

        Pass ``task_type`` (the registry slug) and ``config`` (the
        kwargs originally given to the factory) when the task came from
        :data:`~molexp.workflow.registry.default_registry` and you want
        the resulting spec to be serializable via :meth:`Workflow.to_dict`.

        ``routes=`` / ``next_=`` mirror :meth:`task` — sugar for declaring
        outgoing control edges on the task being added (spec 03 §3, §7).

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

    # ── Control-flow declarations (spec 03) ─────────────────────────────

    def entry(self, name: str) -> WorkflowBuilder:
        """Declare *name* as a workflow entry point.

        Multiple calls add multiple entries (multi-entry workflows are run in
        parallel from frontier 0). Duplicate names raise :class:`ValueError`.
        Validation that *name* refers to a registered task is deferred to
        :meth:`build` so entries can be declared before tasks.
        """
        if name in self._entries:
            raise ValueError(
                f"WorkflowBuilder {self._name!r}: entry {name!r} declared multiple times"
            )
        self._entries.append(name)
        return self

    def control(self, src: str, to: str) -> WorkflowBuilder:
        """Declare an unconditional control edge ``src → to``.

        Multiple calls per *src* fan out to multiple unconditional successors.
        Mixing with :meth:`branch` on the same *src* is rejected at compile
        time (``EdgeShapeError``, spec 03 §3).
        """
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
        """Declare branch (label-routed) outgoing control edges on *src*.

        Two equivalent calling forms (spec 03 §7):

        * ``wf.branch("src", "label", "target")`` — single edge.
        * ``wf.branch("src", routes={"l1": "t1", "l2": "t2"})`` — bulk dict.

        The two forms are mutually exclusive; passing both raises
        :class:`TypeError`. The task body of *src* must return ``Next(label)``
        selecting one of the declared labels (otherwise compile / runtime
        error per spec 03 §3, §5).
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

        Spec 04 §4. The loop's ``until`` task is a normal task whose body
        returns ``Next("continue")`` to re-enter ``body[0]`` or
        ``Next("exit")`` to leave the loop and proceed to ``on_exit``
        (default: terminate the workflow).

        The compiler expands this into:

        * a control edge ``body[-1] → until`` (added only if no equivalent
          edge already exists from data deps or :meth:`control`), and
        * branch edges on ``until``:
          ``{"continue": body[0], "exit": on_exit}``.

        Both expansions live in the static graph — the runtime never adds
        new edges; it only picks among the declared ones.

        Args:
            body: Ordered task names that form the loop body. Each task
                must be registered. ``body[0]`` is the loop head;
                ``body[-1]`` is the tail.
            until: Name of the task that decides ``continue`` vs ``exit``.
                Must be a registered task that returns
                :class:`~molexp.workflow.Next`.
            max_iters: Cap on the number of loop iterations. The runtime
                forces ``Next("exit")`` and emits
                :class:`~molexp.workflow.LoopMaxItersExceeded` when
                reached. Must be ``>= 1``.
            on_exit: Target task that runs after the loop exits. Defaults
                to the special sentinel ``"_end"``, which terminates the
                workflow.

        Returns ``self`` to support chaining.
        """
        if not body:
            raise ValueError(
                f"WorkflowBuilder.loop: body must contain at least one task name; got {body!r}"
            )
        if max_iters < 1:
            raise ValueError(f"WorkflowBuilder.loop: max_iters must be >= 1; got {max_iters!r}")
        self._loops.append(
            LoopDecl(
                body=tuple(body),
                until=until,
                max_iters=max_iters,
                on_exit=on_exit,
            )
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
        """Declare a parallel fan-out: run *body* once per element of *map_over*.

        Spec 05 §4. ``map_over`` is the name of an upstream task whose
        output is an iterable; ``body`` is the per-element worker task
        (single name, not a chain — see Out of scope D1); ``join`` is
        the reducer that receives ``ctx.inputs == [out_0, out_1, …]``
        in element-iteration order.

        The compiler expands this into:

        * an unconditional control edge ``map_over → body``, and
        * an unconditional control edge ``body → join``.

        Both expansions live in the static graph — the runtime spawns
        N body call coroutines but never grows the node set.

        Args:
            map_over: Name of the upstream task whose output (a
                materialised iterable) supplies the per-element inputs.
                Must be a registered task.
            body: Name of the worker task invoked once per element.
                Must be a registered task; may not have explicit
                ``depends_on``, ``wf.control``, ``wf.branch``,
                ``routes=``, or ``next_=`` declarations (the parallel
                primitive owns its wiring).
            join: Name of the reducer task. Receives the aggregated
                ``list[per_element_output]`` via ``ctx.inputs``.
            max_concurrency: Cap on concurrent in-flight body
                invocations. Must be ``>= 1``.

        Returns ``self`` to support chaining.
        """
        if max_concurrency < 1:
            raise ValueError(
                f"WorkflowBuilder.parallel: max_concurrency must be >= 1; got {max_concurrency!r}"
            )
        self._parallels.append(
            ParallelDecl(
                map_over=map_over,
                body=body,
                join=join,
                max_concurrency=max_concurrency,
            )
        )
        return self

    # ── Cross-replicate reducer ─────────────────────────────────────────

    def reduce(
        self,
        *,
        over: str = "replicate",
    ) -> Callable[[Callable[..., TaskOutput]], Callable[..., TaskOutput]]:
        """Register a cross-replicate reducer.

        The reducer is *not* part of the workflow DAG; it is invoked by any
        caller of :meth:`Workflow.run_reducer` after all replicate runs
        of this workflow complete. Output is intended for the
        experiment-level asset library — callers attach it to the
        experiment scope, not the run.

        Args:
            over: Name of the dimension being reduced over (typically
                ``"replicate"``). Recorded on the spec via
                :attr:`Workflow.reducer_dimension`.
        """

        def decorator(fn: Callable[..., TaskOutput]) -> Callable[..., TaskOutput]:
            if self._reducer is not None:
                raise ValueError(
                    f"WorkflowBuilder {self._name!r}: reducer already registered "
                    f"({_callable_name(self._reducer[1])!r}); "
                    "only one reducer per workflow."
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
        """Translate decorator-side ``routes=`` / ``next_=`` into edge declarations."""
        if next_ is not None:
            self._control_edges.append((task_name, next_))
        if routes is not None:
            for lbl, target in routes.items():
                self._branch_edges.append((task_name, lbl, target))

    # ── Compile ─────────────────────────────────────────────────────────

    def build(self) -> Workflow:
        """Compile the registered tasks into a :class:`Workflow`.

        Runs the CFG compiler eagerly so the spec arrives validated:
        data-DAG / edge-shape / entry / reachability errors raise here
        rather than at first execute (spec 03 §8). The compiled
        per-task BaseNode classes are surfaced on
        :attr:`Workflow._compiled_node_classes` for tooling.
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
        # Cross-replicate reducer — not part of the compiled DAG;
        # carried as side-data on the spec for downstream callers.
        spec._reducer = self._reducer
        # Compile eagerly — surfaces CFG validation errors at build time.
        # The compiled artifact is intentionally discarded: the runtime
        # caches its own compilation result keyed by ``workflow_id``.
        from ._pydantic_graph.compiler import WorkflowGraphCompiler

        WorkflowGraphCompiler().compile(spec)
        return spec


# ── Helpers ─────────────────────────────────────────────────────────────────


def _to_snake_case(name: str) -> str:
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    name = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", name)
    return name.lower()


def _callable_code_hash(target: TaskBody) -> str | None:
    """Best-effort AST-normalized code hash for a task callable.

    Mirrors :class:`~molexp.workflow.snapshot.TaskSnapshot`'s code hashing
    (whitespace and comments are normalized away) but works on bare
    callables — both decorator-registered async functions and Task /
    Actor instances. Returns ``None`` when inspection fails (built-ins,
    lambdas without source, etc.).
    """
    import ast
    import inspect
    import textwrap

    fn = getattr(target, "execute", None) or getattr(target, "run", None) or target
    if not callable(fn):
        return None
    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError):
        return None
    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node.decorator_list = []
    normalized = ast.dump(tree, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(normalized.encode()).hexdigest()[:32]


def _stable_workflow_id(name: str, tasks: list[TaskRegistration]) -> str:
    """Deterministic workflow ID from name + task topology."""
    parts = [name]
    for t in tasks:
        dep_str = ",".join(sorted(t.depends_on))
        parts.append(f"{t.name}:{type(t.fn_or_class).__qualname__}:[{dep_str}]")
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _ir_object_list(value: JSONValue | None) -> list[dict[str, JSONValue]]:
    """Narrow a ``JSONValue`` IR field expected to hold a list of JSON objects.

    Accepts ``None`` / non-list / list-with-non-dict entries by returning
    only the dict entries. Used by :meth:`Workflow.from_dict` so the
    rest of the parser can rely on a homogeneous shape.
    """
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _require_str(obj: dict[str, JSONValue], key: str) -> str:
    """Read a string field from an IR object, raising on absent / wrong type."""
    value = obj.get(key)
    if not isinstance(value, str):
        raise ValueError(
            f"IR object is missing required string field {key!r} (got {type(value).__name__})"
        )
    return value
