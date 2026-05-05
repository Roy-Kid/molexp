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
    ) -> None:
        self.name = name
        self.fn_or_class = fn_or_class
        self.depends_on = depends_on
        self.is_actor = is_actor
        self.remote = remote
        self.task_type = task_type
        self.config = dict(config) if config else None


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
    ) -> None:
        self.name = name
        self.workflow_id = workflow_id
        self.version = version
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

    def __init__(self, name: str, mode: str = "batch", version: str = "0") -> None:
        self._name = name
        self._mode = mode
        self._version = version
        self._tasks: list[TaskRegistration] = []

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
        task_type: str | None = None,
        config: dict[str, Any] | None = None,
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
                task_type=task_type,
                config=config,
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
            version=self._version,
        )


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
