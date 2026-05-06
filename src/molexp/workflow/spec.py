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
from collections.abc import Callable, Mapping
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


class LoopDecl:
    """User-declared loop topology compiled by the CFG compiler.

    Spec 04 §4 ``wf.loop`` primitive. The compiler synthesises:

    * a control edge ``body[-1] → until`` (if not already declared), and
    * branch edges on ``until``: ``{"continue": body[0], "exit": on_exit}``.

    ``max_iters`` is enforced by the runtime: when ``until`` dispatches
    ``Next("continue")`` ``max_iters`` times, the runtime forces
    ``Next("exit")`` and emits :class:`LoopMaxItersExceeded`.
    """

    __slots__ = ("body", "until", "max_iters", "on_exit")

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

    __slots__ = ("map_over", "body", "join", "max_concurrency")

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


class SanityHook:
    """A post-task predicate hook.

    See :meth:`Workflow.sanity_check`.  ``predicate`` is called with the
    workflow ``WorkflowState`` after the task named in ``after`` has
    completed; on a falsy return value the runtime acts according to
    :attr:`on_fail`.
    """

    __slots__ = ("after", "predicate", "on_fail")

    def __init__(
        self,
        *,
        after: str,
        predicate: Callable[[Any], bool],
        on_fail: str,
    ) -> None:
        if on_fail not in ("halt", "replan", "continue"):
            raise ValueError(
                f"sanity_check.on_fail must be one of "
                f"'halt' / 'replan' / 'continue'; got {on_fail!r}"
            )
        self.after = after
        self.predicate = predicate
        self.on_fail = on_fail


class TaskRegistration:
    """Internal record of a registered task or actor.

    ``task_type`` and ``config`` are populated when the task originates
    from a registry slug (either via :meth:`Workflow.add` with
    ``task_type=...`` or via :meth:`WorkflowSpec.from_dict`). They are
    required for :meth:`WorkflowSpec.to_dict` to produce IR JSON.
    """

    __slots__ = (
        "name",
        "fn_or_class",
        "depends_on",
        "is_actor",
        "remote",
        "task_type",
        "config",
        "dependent_params",
    )

    def __init__(
        self,
        name: str,
        fn_or_class: Any,
        depends_on: list[str],
        is_actor: bool = False,
        remote: Any = None,
        task_type: str | None = None,
        config: dict[str, Any] | None = None,
        dependent_params: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    ) -> None:
        self.name = name
        self.fn_or_class = fn_or_class
        self.depends_on = depends_on
        self.is_actor = is_actor
        self.remote = remote
        self.task_type = task_type
        self.config = dict(config) if config else None
        self.dependent_params = dependent_params


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
        self.version = version
        self._tasks = tasks
        self._mode = mode
        # Raw control-flow declarations from the Workflow builder.
        # Compiled into ``out_edges`` + ``entry_frontier`` by the CFG compiler.
        self._entries = entries
        self._control_edges = control_edges
        self._branch_edges = branch_edges
        self._loops = loops
        self._parallels = parallels
        self._runtime: Any = None  # WorkflowRuntime, lazy
        # Sweep-level cross-replicate reducer (set by Workflow.build()).
        self._reducer: tuple[str, Callable[..., Any]] | None = None
        # Sanity-check hooks (set by Workflow.build()).
        self._sanity_hooks: tuple[SanityHook, ...] = ()

    def _get_runtime(self) -> Any:
        if self._runtime is None:
            from .runtime import create_default_runtime

            self._runtime = create_default_runtime()
        return self._runtime

    # ── Sweep-level reducer ─────────────────────────────────────────────

    @property
    def reducer_dimension(self) -> str | None:
        """Dimension declared on ``@wf.reduce(over=...)``; ``None`` if no reducer."""
        return self._reducer[0] if self._reducer is not None else None

    def run_reducer(self, replicate_outputs: list[Any]) -> Any:
        """Invoke the registered ``@wf.reduce`` on a list of replicate outputs.

        Raises ``LookupError`` if no reducer was registered. Callers attach
        the return value to the experiment-scope asset library; this method
        does not write anywhere itself.
        """
        if self._reducer is None:
            raise LookupError(
                f"WorkflowSpec {self.name!r}: no reducer registered (call @wf.reduce(over=...))"
            )
        return self._reducer[1](replicate_outputs)

    @property
    def sanity_hooks(self) -> tuple[SanityHook, ...]:
        """Frozen view of the sanity-check hooks declared on the parent ``Workflow``."""
        return self._sanity_hooks

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

    # ── Versioning ──────────────────────────────────────────────────────

    def to_workflow_version(self) -> Any:
        """Build the immutable :class:`~molexp.workflow.version.WorkflowVersion`.

        Returns:
            A :class:`WorkflowVersion` capturing this spec's
            ``workflow_id``, ``version``, ``name``, and topology
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
            version=self.version,
            name=self.name,
            topology=tuple(topo),
        )

    def register(self, workspace: Any) -> Any:
        """Persist this spec's :class:`WorkflowVersion` under the workspace.

        Idempotent on identical ``(workflow_id, version)`` re-registers;
        raises :class:`~molexp.workflow.version.WorkflowVersionConflictError`
        when ``workflow_id`` already maps to a different ``version``.

        Args:
            workspace: A :class:`~molexp.workspace.Workspace` instance.

        Returns:
            The :class:`WorkflowVersion` record that is now on disk.
        """
        from .version import write_record

        record = self.to_workflow_version()
        workspace._ensure_materialized()
        write_record(workspace, record)
        return record

    # ── IR serialization (JSON workflow IR) ─────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize this spec to the JSON IR shape (see ``schema/workflow.json``).

        Every task must have been registered with a ``task_type`` slug
        (passed via :meth:`Workflow.add` or set by
        :meth:`WorkflowSpec.from_dict`); otherwise this raises
        :class:`ValueError`. Decorator-style functions and ad-hoc Task
        instances added without a slug are not serializable — they live
        only in process memory.
        """
        unslugged = [t.name for t in self._tasks if t.task_type is None]
        if unslugged:
            raise ValueError(
                "Cannot serialize workflow to IR: the following tasks have no "
                f"task_type slug: {unslugged}. Use `Workflow.add(..., task_type=...)` "
                "or build the spec from IR via WorkflowSpec.from_dict()."
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

        task_configs = [
            {
                "task_id": t.name,
                "task_type": t.task_type,
                "config": dict(t.config) if t.config else {},
                "status": "pending",
            }
            for t in self._tasks
        ]
        links = [
            {"source": dep, "target": t.name, "mapping": {}, "status": "pending"}
            for t in self._tasks
            for dep in t.depends_on
        ]
        return {
            "workflow_id": f"workflow_{self.workflow_id[:8]}",
            "name": self.name,
            "task_configs": task_configs,
            "links": links,
            "metadata": {
                "label": None,
                "description": None,
                "tags": [],
                "custom": {},
            },
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        registry: Any = None,
    ) -> WorkflowSpec:
        """Build a :class:`WorkflowSpec` from JSON IR.

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
            from .registry import default_registry as registry  # type: ignore[no-redef]

        from .protocols import Streamable

        deps_by_target: dict[str, list[str]] = {}
        for link in data.get("links", []):
            target = link["target"]
            deps_by_target.setdefault(target, []).append(link["source"])

        tasks: list[TaskRegistration] = []
        for tc in data.get("task_configs", []):
            slug = tc["task_type"]
            config = dict(tc.get("config") or {})
            factory = registry.get(slug)
            instance = factory(config)
            tasks.append(
                TaskRegistration(
                    name=tc["task_id"],
                    fn_or_class=instance,
                    depends_on=deps_by_target.get(tc["task_id"], []),
                    is_actor=isinstance(instance, Streamable),
                    task_type=slug,
                    config=config,
                )
            )

        # Validate every link target / source is a known task_id
        known = {t.name for t in tasks}
        for link in data.get("links", []):
            for endpoint in (link["source"], link["target"]):
                if endpoint not in known:
                    raise ValueError(
                        f"Link references unknown task_id {endpoint!r}; known: {sorted(known)}"
                    )

        name = data.get("name") or ""
        return cls(
            name=name,
            workflow_id=_stable_workflow_id(name, tasks),
            tasks=tasks,
            mode="batch",
        )


# ── Workflow (unified OOP API) ──────────────────────────────────────────────


class Workflow:
    """OOP workflow definition. Supports decorator and builder styles.

    Instantiate once, then register tasks via the decorators
    (:meth:`task`, :meth:`actor`) or the OOP method :meth:`add`. Wire
    control flow with :meth:`control` / :meth:`branch` / :meth:`loop`
    / :meth:`parallel`. Call :meth:`build` to produce a
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
        self._reducer: tuple[str, Callable[..., Any]] | None = None
        self._task_path_registry: dict[str, type] = {}
        self._sanity_hooks: list[SanityHook] = []
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
    def version(self) -> str:
        return self._version

    # ── Decorator: function-as-task ─────────────────────────────────────

    def task(
        self,
        fn: Callable | None = None,
        *,
        depends_on: list[str] | None = None,
        name: str | None = None,
        remote: Any = None,
        routes: Mapping[str, str] | None = None,
        next_: str | None = None,
        dependent_params: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
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
                "Workflow.task: routes= and next_= are mutually exclusive "
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
            raise TypeError("Workflow.actor: routes= and next_= are mutually exclusive.")

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
        task: Any,
        *,
        depends_on: list[str] | None = None,
        name: str | None = None,
        remote: Any = None,
        task_type: str | None = None,
        config: dict[str, Any] | None = None,
        routes: Mapping[str, str] | None = None,
        next_: str | None = None,
        dependent_params: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    ) -> Workflow:
        """Register a Task / Actor instance (or any Runnable/Streamable).

        Accepts:
        - A :class:`~molexp.workflow.task.Task` / :class:`~molexp.workflow.task.Actor` instance
        - Any object matching the :class:`~molexp.workflow.protocols.Runnable`
          or :class:`~molexp.workflow.protocols.Streamable` protocol
        - A bare callable (treated as a batch task)

        Pass ``task_type`` (the registry slug) and ``config`` (the
        kwargs originally given to the factory) when the task came from
        :data:`~molexp.workflow.registry.default_registry` and you want
        the resulting spec to be serializable via :meth:`WorkflowSpec.to_dict`.

        ``routes=`` / ``next_=`` mirror :meth:`task` — sugar for declaring
        outgoing control edges on the task being added (spec 03 §3, §7).

        Returns ``self`` to support chaining.
        """
        from .protocols import Streamable

        if routes is not None and next_ is not None:
            raise TypeError("Workflow.add: routes= and next_= are mutually exclusive.")

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

    def entry(self, name: str) -> Workflow:
        """Declare *name* as a workflow entry point.

        Multiple calls add multiple entries (multi-entry workflows are run in
        parallel from frontier 0). Duplicate names raise :class:`ValueError`.
        Validation that *name* refers to a registered task is deferred to
        :meth:`build` so entries can be declared before tasks.
        """
        if name in self._entries:
            raise ValueError(f"Workflow {self._name!r}: entry {name!r} declared multiple times")
        self._entries.append(name)
        return self

    def control(self, src: str, to: str) -> Workflow:
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
    ) -> Workflow:
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
                    "Workflow.branch: pass either positional (src, label, to) "
                    "or keyword routes={...}, not both."
                )
            for lbl, target in routes.items():
                self._branch_edges.append((src, lbl, target))
            return self
        if label is None or to is None:
            raise TypeError(
                "Workflow.branch: pass (src, label, to) or routes={...}; "
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
    ) -> Workflow:
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
                f"Workflow.loop: body must contain at least one task name; got {body!r}"
            )
        if max_iters < 1:
            raise ValueError(f"Workflow.loop: max_iters must be >= 1; got {max_iters!r}")
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
    ) -> Workflow:
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
                f"Workflow.parallel: max_concurrency must be >= 1; got {max_concurrency!r}"
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

    # ── Cross-replicate reducer (sweep-level fan-in) ────────────────────

    def reduce(
        self,
        *,
        over: str = "replicate",
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Register a cross-replicate reducer for sweep-level fan-in.

        The reducer is *not* part of the workflow DAG; it is invoked by the
        sweep runner (or any caller of :meth:`WorkflowSpec.run_reducer`)
        after all replicate runs of this workflow complete. Output is
        intended for the experiment-level asset library — callers attach
        it to the experiment scope, not the run.

        Args:
            over: Name of the dimension being reduced over (typically
                ``"replicate"``). Recorded on the spec via
                :attr:`WorkflowSpec.reducer_dimension`.
        """

        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            if self._reducer is not None:
                raise ValueError(
                    f"Workflow {self._name!r}: reducer already registered "
                    f"({self._reducer[1].__name__!r}); only one reducer per workflow."
                )
            self._reducer = (over, fn)
            return fn

        return decorator

    # ── Agent-authored Task hot-load ────────────────────────────────────

    def register_task_path(self, path: Any) -> Workflow:
        """Load a Python file and register every ``Task`` / ``Actor`` subclass.

        Discovered classes are stored in this workflow's local registry and
        can later be resolved by :meth:`resolve_task_class`. Scope is
        per-workflow — sibling :class:`Workflow` instances do not see each
        other's loaded classes (acceptance ac-003 / ac-009).

        The caller is responsible for keeping ``path`` inside whatever
        sandboxed scratch directory the agent service uses (e.g.
        ``workspace_root/.scratch/agent_tasks/``); this method does not
        enforce that boundary itself.
        """
        import importlib.util
        from inspect import isclass
        from pathlib import Path

        from .protocols import Runnable, Streamable
        from .task import Actor
        from .task import Task as _Task

        p = Path(path)
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(f"register_task_path: {p} is not a file")

        # Build a unique module name so repeated loads do not clobber sys.modules
        # entries from sibling workflows.
        module_name = f"_molexp_agent_task_{abs(hash((self._name, str(p.resolve()))))}"
        spec_obj = importlib.util.spec_from_file_location(module_name, p)
        if spec_obj is None or spec_obj.loader is None:
            raise OSError(f"register_task_path: cannot load module spec for {p}")

        module = importlib.util.module_from_spec(spec_obj)
        spec_obj.loader.exec_module(module)

        for attr_name in dir(module):
            obj = getattr(module, attr_name)
            if not isclass(obj):
                continue
            # Skip the imported base classes themselves.
            if obj in (_Task, Actor):
                continue
            if (
                issubclass(obj, _Task)
                or issubclass(obj, Actor)
                or isinstance(obj, type)
                and (issubclass(obj, Runnable) or issubclass(obj, Streamable))
            ):
                self._task_path_registry[attr_name] = obj
        return self

    def resolve_task_class(self, class_name: str) -> type:
        """Look up an agent-loaded Task class by name.

        Raises ``KeyError`` when the class was not loaded into *this*
        workflow's registry (sibling workflows are isolated).
        """
        try:
            return self._task_path_registry[class_name]
        except KeyError as exc:
            raise KeyError(
                f"Workflow {self._name!r}: task class {class_name!r} not loaded; "
                f"call register_task_path() first."
            ) from exc

    # ── Sanity-check hook ───────────────────────────────────────────────

    def sanity_check(
        self,
        *,
        after: str,
        predicate: Callable[[Any], bool],
        on_fail: str = "halt",
    ) -> Workflow:
        """Register a post-task predicate hook.

        After the task named ``after`` completes, the runtime calls
        ``predicate(state)`` with the live :class:`WorkflowState`. A truthy
        return passes; a falsy return triggers ``on_fail``:

        - ``"halt"`` — raise :class:`SanityCheckFailed`, which the runtime
          surfaces as ``WorkflowResult.status == "failed"``.
        - ``"replan"`` — record a structured event on
          ``WorkflowResult.sanity_events`` and continue (the agent service
          consumes the event to drive a replan turn).
        - ``"continue"`` — log only; no halt, no event.
        """
        self._sanity_hooks.append(SanityHook(after=after, predicate=predicate, on_fail=on_fail))
        return self

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

    def build(self) -> WorkflowSpec:
        """Compile the registered tasks into a :class:`WorkflowSpec`.

        Runs the CFG compiler eagerly so the spec arrives validated:
        data-DAG / edge-shape / entry / reachability errors raise here
        rather than at first execute (spec 03 §8). The compiled
        per-task BaseNode classes are surfaced on
        :attr:`WorkflowSpec._compiled_node_classes` for tooling.
        """
        tasks = list(self._tasks)
        spec = WorkflowSpec(
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
        # Sweep-level reducer + sanity hooks — not part of the compiled DAG;
        # carried as side-data on the spec for the runtime / sweep runner.
        spec._reducer = self._reducer
        spec._sanity_hooks = tuple(self._sanity_hooks)
        # Compile eagerly — surfaces CFG validation errors at build time.
        from ._pydantic_graph.compiler import WorkflowGraphCompiler

        compiled = WorkflowGraphCompiler().compile(spec)
        spec._cached_compiled = compiled  # type: ignore[attr-defined]
        return spec


# ── Helpers ─────────────────────────────────────────────────────────────────


def _to_snake_case(name: str) -> str:
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    name = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", name)
    return name.lower()


def _callable_code_hash(target: Any) -> str | None:
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
