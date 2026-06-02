"""SweepMap — map a callable over a ParamSpace, one ArtifactAsset per cell.

``SweepMap`` is a reusable batch :class:`~molexp.workflow.task.Task` that fans a
user callable across every parameter cell of an iterable parameter space (e.g.
:class:`molexp.workspace.GridSpace`) and persists each cell's result as a single
``ArtifactAsset`` in the active run, tagged with that cell's parameters.

It is the canonical primitive for variable-matrix sweeps: callers supply a pure
``fn(cell) -> result`` and a space; the task handles per-cell persistence and
lineage so no glue code is needed in downstream projects.

Example::

    from molexp.workflow import SweepMap, WorkflowCompiler
    from molexp.workspace import GridSpace

    space = GridSpace({"scheme": ["int8", "int4"], "dataset": ["qm9"]})
    wf = WorkflowCompiler(name="sweep")
    wf.add(SweepMap(lambda cell: {"scheme": cell["scheme"]}, space), name="cells")

The callable may be sync or async. Errors raised by the callable propagate
(they are never silently swallowed). Requires a run context with an artifact
accessor (``ctx.run_context.artifact``); raised as ``RuntimeError`` otherwise.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Any

from molexp.workflow.context import TaskContext
from molexp.workflow.task import Task

if TYPE_CHECKING:
    from molexp.workspace.assets.artifact import ArtifactAsset

Cell = Mapping[str, Any]
SweepFn = Callable[[Cell], Any]


class SweepMap(Task[Any, Any, Any, "list[ArtifactAsset]"]):
    """Map a callable over each cell of a parameter space, saving one asset per cell.

    Args:
        fn: Callable applied to each parameter cell. Receives the cell mapping
            and returns a JSON-serializable value (or bytes / ``Path`` / str)
            accepted by :meth:`ArtifactAccessor.save`. May be async.
        space: Any iterable of parameter cells (mappings), such as a
            :class:`molexp.workspace.GridSpace`. An empty space writes nothing.
        name_prefix: Filename prefix for each per-cell artifact; the cell index
            is appended (``"<name_prefix>-<i>.json"``).
        mime: Optional MIME hint forwarded to :meth:`ArtifactAccessor.save`.
    """

    def __init__(
        self,
        fn: SweepFn,
        space: Iterable[Cell],
        *,
        name_prefix: str = "cell",
        mime: str | None = None,
    ) -> None:
        self._fn = fn
        self._space = space
        self._name_prefix = name_prefix
        self._mime = mime

    async def execute(self, ctx: TaskContext[Any, Any, Any]) -> list[ArtifactAsset]:
        """Run ``fn`` for every cell and persist each result as an artifact."""
        run_context = ctx.run_context
        if run_context is None:
            raise RuntimeError(
                "SweepMap requires a run_context; execute the workflow with "
                "WorkflowRuntime().execute(..., run_context=ctx)."
            )

        assets: list[ArtifactAsset] = []
        for index, cell in enumerate(self._space):
            result = self._fn(cell)
            if inspect.isawaitable(result):
                result = await result
            tags = {key: str(value) for key, value in cell.items()}
            tags["sweep_index"] = str(index)
            name = f"{self._name_prefix}-{index}.json"
            assets.append(run_context.artifact.save(name, result, tags=tags, mime=self._mime))
        return assets
