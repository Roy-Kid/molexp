"""Convenience base classes for workflow tasks.

``Task`` and ``Actor`` implement the :class:`~Runnable` and
:class:`~Streamable` protocols respectively. Using them is **optional** —
any object whose signature matches the protocol works equally well.

``Task`` and ``Actor`` are **plain abstract classes** — they do not
inherit ``pydantic_graph.BaseNode``. Each task is lowered to a genuine
pydantic-graph ``Step`` (one per task; see
:mod:`molexp.workflow._pydantic_graph.compiler`); the Step body invokes
``execute(ctx)`` / ``run(ctx)`` directly via duck typing.

A task is a pure ``execute(inputs, config) -> output`` function: it reads only
``ctx.inputs`` and ``ctx.config`` (see :class:`~molexp.workflow.TaskContext`).

Simple (no generics)::

    class Square(Task):
        async def execute(self, ctx: TaskContext) -> int:
            return ctx.inputs**2

Typed (state/input/output generics)::

    class Fetch(Task[WorkflowState, str, DataFrame]):
        async def execute(self, ctx: TaskContext[WorkflowState, str]) -> DataFrame:
            # source path arrives as an input, not via ambient deps.
            return read_frame(ctx.inputs)

Third-party (no molexp import)::

    # Works in any ``Workflow.add(...)`` because it satisfies Runnable
    class ExternalProcessor:
        async def execute(self, ctx) -> dict: ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from .context import TaskContext


class Task[StateT, InputT, OutputT](ABC):
    """Batch task base class.

    Subclass and implement :meth:`execute`. Generic parameters are
    optional — plain ``Task`` defaults to ``Any``.
    """

    @abstractmethod
    async def execute(self, ctx: TaskContext[StateT, InputT]) -> OutputT:
        """Run this task and return the output."""
        ...


class Actor[StateT, InputT, OutputT](ABC):
    """Streaming actor base class.

    Subclass and implement :meth:`run` as an async generator yielding
    output chunks (the terminal yield may be ``Next(label)`` / ``End()``
    per spec 03 §5).
    """

    # ``run`` is declared without ``async`` because async-generator
    # functions return their iterator directly on call (no ``await``);
    # marking the abstract method ``async def`` would let static
    # checkers conclude callers must ``await`` first, breaking the
    # ``async for chunk in actor.run(ctx)`` dispatch in the Step body
    # (node._invoke_body_with_ctx).
    @abstractmethod
    def run(self, ctx: TaskContext[StateT, InputT]) -> AsyncIterator[OutputT]:
        """Run continuously, yielding outputs."""
        ...
