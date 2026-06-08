"""SweepMap — map a callable over a ParamSpace, one result record per cell.

``SweepMap`` is a reusable batch :class:`~molexp.workflow.task.Task` that fans a
user callable across every parameter cell of an iterable parameter space (e.g.
:class:`molexp.workspace.GridSpace`) and **returns** one result record per cell.

Under the pure-task-context contract a task never reaches a ``run_context``; it
returns its products and the engine's materialization layer persists them. So
``SweepMap.execute`` no longer calls ``artifact.save`` — it returns a list of
``{"name", "tags", "result"}`` records (the cell's computed value plus its
naming + parameter tags), which the engine persists as the node's artifact.

It is the canonical primitive for variable-matrix sweeps: callers supply a pure
``fn(cell) -> result`` and a space; the task computes every cell.

Example::

    from molexp.workflow import SweepMap, WorkflowCompiler
    from molexp.workspace import GridSpace

    space = GridSpace({"scheme": ["int8", "int4"], "dataset": ["qm9"]})
    wf = WorkflowCompiler(name="sweep")
    wf.add(SweepMap(lambda cell: {"scheme": cell["scheme"]}, space), name="cells")

The callable may be sync or async. Errors raised by the callable propagate
(they are never silently swallowed).
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from molexp.workflow.context import TaskContext
from molexp.workflow.task import Task

Cell = Mapping[str, Any]
SweepFn = Callable[[Cell], Any]
NameFn = Callable[[Cell, int], str]
CellRecord = dict[str, Any]


class SweepMap(Task[Any, Any, "list[CellRecord]"]):
    """Map a callable over each cell of a parameter space, one record per cell.

    Args:
        fn: Callable applied to each parameter cell. Receives the cell mapping
            and returns its (JSON-serializable) result. May be async.
        space: Any iterable of parameter cells (mappings), such as a
            :class:`molexp.workspace.GridSpace`. An empty space returns ``[]``.
        name_prefix: Filename prefix for each per-cell record; the cell index
            is appended (``"<name_prefix>-<i>.json"``). Ignored when ``name_fn``
            is given.
        mime: Optional MIME hint recorded alongside each cell record.
        name_fn: Optional ``name_fn(cell, index) -> str`` that returns the full
            per-cell name, giving the caller control over naming and extension.
            When omitted the default ``"<name_prefix>-<i>.json"`` is used.
    """

    def __init__(
        self,
        fn: SweepFn,
        space: Iterable[Cell],
        *,
        name_prefix: str = "cell",
        mime: str | None = None,
        name_fn: NameFn | None = None,
    ) -> None:
        self._fn = fn
        self._space = space
        self._name_prefix = name_prefix
        self._mime = mime
        self._name_fn = name_fn

    async def execute(self, ctx: TaskContext[Any, Any]) -> list[CellRecord]:  # noqa: ARG002
        """Run ``fn`` for every cell and return one record per cell.

        Each record is ``{"name", "tags", "result", "mime"}``; the engine's
        materialization layer persists the returned list as this node's artifact.
        """
        records: list[CellRecord] = []
        for index, cell in enumerate(self._space):
            result = self._fn(cell)
            if inspect.isawaitable(result):
                result = await result
            tags = {key: str(value) for key, value in cell.items()}
            tags["sweep_index"] = str(index)
            name = (
                self._name_fn(cell, index)
                if self._name_fn is not None
                else f"{self._name_prefix}-{index}.json"
            )
            records.append({"name": name, "tags": tags, "result": result, "mime": self._mime})
        return records


__all__ = ["SweepMap"]
