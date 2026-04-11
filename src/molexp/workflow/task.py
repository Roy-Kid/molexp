"""Convenience base classes for workflow tasks.

``Task`` and ``Actor`` implement the :class:`~Runnable` and
:class:`~Streamable` protocols respectively. Using them is **optional** —
any object whose signature matches the protocol works equally well.

Simple (no generics)::

    class Square(Task):
        async def execute(self, ctx: TaskContext) -> int:
            return ctx.inputs ** 2

Typed (full generics)::

    class Fetch(Task[PipeState, PipeDeps, None, DataFrame]):
        async def execute(
            self, ctx: TaskContext[PipeState, PipeDeps, None]
        ) -> DataFrame:
            return ctx.deps.storage.read(self.source)

Third-party (no molexp import)::

    # This also works in a WorkflowBuilder because it satisfies Runnable
    class ExternalProcessor:
        async def execute(self, ctx) -> dict: ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator, Generic

from .context import ActorContext, TaskContext
from .types import DepsT, InputT, OutputT, StateT


class Task(ABC, Generic[StateT, DepsT, InputT, OutputT]):
    """Batch task base class (implements :class:`~Runnable` protocol).

    Subclass and implement ``execute()``.
    Generic parameters are optional — plain ``Task`` defaults to ``Any``.
    """

    @abstractmethod
    async def execute(self, ctx: TaskContext[StateT, DepsT, InputT]) -> OutputT:
        """Run this task and return the output."""
        ...


class Actor(ABC, Generic[StateT, DepsT, InputT, OutputT]):
    """Streaming actor base class (implements :class:`~Streamable` protocol).

    Subclass and implement ``run()`` as an async generator.
    """

    @abstractmethod
    async def run(
        self, ctx: ActorContext[StateT, DepsT, InputT]
    ) -> AsyncIterator[OutputT]:
        """Run continuously, yielding outputs."""
        ...
