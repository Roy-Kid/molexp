"""Internal graph-declaration types used by spec, builder, and compiler."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from .protocols import JSONMapping, TaskBody, UpstreamViewLike, UserDeps

type DependentParamsFn = Callable[[Mapping[str, UpstreamViewLike]], JSONMapping]


class LoopDecl:
    """``wf.loop`` primitive (spec 04 §4). Compiler synthesises control+branch edges."""

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
    """``wf.parallel`` primitive (spec 05 §4). Compiler synthesises two control edges."""

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
    """Placeholder for subgraph boundary upstreams. Raises if ever invoked."""

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    async def execute(self, ctx: object) -> object:
        del ctx
        raise RuntimeError(
            f"Subgraph boundary stub {self.name!r} was invoked — "
            f"seed_outputs[{self.name!r}] was not provided."
        )


class WorkflowTopology:
    """Decoupled topology carrier handed to the CFG lowering pass.

    Holds exactly the declaration state the lowering reads — task
    registrations plus the control/branch/loop/parallel/entry decls — so
    ``_pydantic_graph`` no longer needs to import the public
    ``WorkflowCompiler`` / ``CompiledWorkflow``. The private attribute
    names mirror the old ``Workflow`` spec so the lowering bodies are
    unchanged.
    """

    __slots__ = (
        "_branch_edges",
        "_control_edges",
        "_entries",
        "_loops",
        "_parallels",
        "_tasks",
        "name",
    )

    def __init__(
        self,
        *,
        name: str,
        tasks: list[TaskRegistration],
        entries: tuple[str, ...] = (),
        control_edges: tuple[tuple[str, str], ...] = (),
        branch_edges: tuple[tuple[str, str, str], ...] = (),
        loops: tuple[LoopDecl, ...] = (),
        parallels: tuple[ParallelDecl, ...] = (),
    ) -> None:
        self.name = name
        self._tasks = tasks
        self._entries = entries
        self._control_edges = control_edges
        self._branch_edges = branch_edges
        self._loops = loops
        self._parallels = parallels


class TaskRegistration:
    """Internal record of one registered task or actor."""

    __slots__ = (
        "dependent_params",
        "depends_on",
        "fn_or_class",
        "is_actor",
        "name",
        "position",
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
        dependent_params: DependentParamsFn | None = None,
        position: tuple[float, float] | None = None,
    ) -> None:
        self.name = name
        self.fn_or_class = fn_or_class
        self.depends_on = depends_on
        self.is_actor = is_actor
        self.remote = remote
        self.task_type = task_type
        self.dependent_params = dependent_params
        # Editor-canvas coordinate metadata (free-layout graph). Pure UI
        # metadata: it round-trips through the IR but never enters the
        # ``TaskSnapshot`` content hash (the snapshot reads the task instance's
        # code + captured ``__init__`` config), so moving a node never
        # invalidates the cache.
        self.position = position
