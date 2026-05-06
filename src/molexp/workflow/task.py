"""Convenience base classes for workflow tasks.

``Task`` and ``Actor`` implement the :class:`~Runnable` and
:class:`~Streamable` protocols respectively. Using them is **optional** —
any object whose signature matches the protocol works equally well.

Spec 03 — both ``Task`` and ``Actor`` publicly inherit
:class:`pydantic_graph.BaseNode`. OOP-style user subclasses are therefore
already pydantic-graph nodes; the workflow compiler wraps each task in a
per-workflow BaseNode subclass that synthesises the right ``run`` return
annotation (for successor type checking) and forwards the call to the
user's ``execute`` (or async-generator ``run``).

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

    # Works in any ``Workflow.add(...)`` because it satisfies Runnable
    class ExternalProcessor:
        async def execute(self, ctx) -> dict: ...
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, AsyncIterator, Generic

from pydantic_graph import BaseNode

from .context import ActorContext, TaskContext
from .types import DepsT, InputT, OutputT, StateT


class Task(
    BaseNode[Any, Any, Any],
    Generic[StateT, DepsT, InputT, OutputT],
):
    """Batch task base class.

    Subclass and implement :meth:`execute`. Generic parameters are
    optional — plain ``Task`` defaults to ``Any``.

    ``Task`` publicly inherits :class:`pydantic_graph.BaseNode` per spec 03 §10
    so ``issubclass(MyTask, pydantic_graph.BaseNode)`` is true and the
    compiler can register OOP-style task classes directly into the
    pydantic-graph ``Graph``. The base ``run`` here is a placeholder that
    delegates to :meth:`execute`; the compiler always overrides it on the
    per-workflow wrapper class so the return-type annotation matches the
    task's declared out-edges.
    """

    @abstractmethod
    async def execute(self, ctx: TaskContext[StateT, DepsT, InputT]) -> OutputT:
        """Run this task and return the output."""
        ...

    async def run(self, ctx: Any) -> Any:  # noqa: D401 — abstract-method satisfier
        """Default ``BaseNode.run`` — overridden by the workflow compiler.

        Calling this directly (without compile-time wiring) is a programming
        error: ``Task`` instances must be registered via ``Workflow.add(...)``
        or a ``@wf.task`` decorator and then executed via ``WorkflowSpec.execute``.
        """
        raise RuntimeError(
            f"{type(self).__name__}.run() called without compile-time wiring. "
            "Register the task via Workflow.add(...) or @wf.task and execute "
            "through WorkflowSpec.execute()."
        )


class Actor(
    BaseNode[Any, Any, Any],
    Generic[StateT, DepsT, InputT, OutputT],
):
    """Streaming actor base class.

    Subclass and implement :meth:`run` as an async generator yielding
    output chunks (the terminal yield may be ``Next(label)`` / ``End()``
    per spec 03 §5).

    ``Actor`` publicly inherits :class:`pydantic_graph.BaseNode`. The actor's
    own ``run`` is an async generator (not a coroutine returning the next
    node); the workflow compiler wraps it so the generated per-workflow
    subclass exposes a ``run`` coroutine that consumes the generator and
    dispatches on its terminal yield.
    """

    @abstractmethod  # type: ignore[misc]  # signature differs from BaseNode.run by design
    async def run(  # type: ignore[override]
        self, ctx: ActorContext[StateT, DepsT, InputT]
    ) -> AsyncIterator[OutputT]:
        """Run continuously, yielding outputs."""
        ...
