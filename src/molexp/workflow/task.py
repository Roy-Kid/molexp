"""Convenience base classes for workflow tasks.

``Task`` and ``Actor`` implement the :class:`~Runnable` and
:class:`~Streamable` protocols respectively. Using them is **optional** —
any object whose signature matches the protocol works equally well.

``Task`` and ``Actor`` are **plain abstract classes** — they do not
inherit ``pydantic_graph.BaseNode``. Each task is lowered to a genuine
pydantic-graph ``Step`` (one per task; see
:mod:`molexp.workflow._pydantic_graph.compiler`); the Step body invokes
``execute(ctx)`` / ``run(ctx)`` directly via duck typing.

Identity has three orthogonal parts: **code** (the ``execute`` body), **config**
(the ``__init__`` arguments — a task instance *is* its config), and **input**
(the data delivered at run time via ``ctx.inputs``). Build-time config is supplied
to the constructor and captured automatically (``self._task_config``); the body
reads it as plain ``self.*`` attributes. ``ctx`` carries only the runtime input.
The content-addressed cache and IR serialization both key on the captured config,
so a task never re-declares it anywhere else (no ``builder.add(config=)``).

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

import functools
import inspect
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from .context import TaskContext


class _CapturesInitConfig:
    """Mixin: record a task's ``__init__`` arguments as its config identity.

    A task instance *is* its build-time config — the args it was constructed with.
    This mixin wraps each subclass's ``__init__`` to snapshot the bound arguments
    (defaults applied, ``self`` dropped) into ``self._task_config``. That dict is
    the single source of truth for the content-addressed cache's ``config_hash``
    and for IR serialization / ``cls(**config)`` reconstruction — so config is
    declared exactly once, at the call site that constructs the task.

    Only the most-derived ``__init__`` is captured: the guard skips re-capture
    when a subclass ``__init__`` calls ``super().__init__()``.
    """

    _task_config: dict[str, Any]

    def __init_subclass__(cls, **kwargs: Any) -> None:  # noqa: ANN401
        super().__init_subclass__(**kwargs)
        init = cls.__dict__.get("__init__")
        if init is None:  # inherits a parent's (already-wrapped) __init__
            return
        sig = inspect.signature(init)

        @functools.wraps(init)
        def _wrapped(self: Any, *args: Any, **kw: Any) -> None:  # noqa: ANN401
            if not hasattr(self, "_task_config"):
                bound = sig.bind(self, *args, **kw)
                bound.apply_defaults()
                self._task_config = {
                    name: value for name, value in bound.arguments.items() if name != "self"
                }
            init(self, *args, **kw)

        cls.__init__ = _wrapped  # ty: ignore[invalid-assignment]


class Task[StateT, InputT, OutputT](_CapturesInitConfig, ABC):
    """Batch task base class.

    Subclass and implement :meth:`execute`. Generic parameters are
    optional — plain ``Task`` defaults to ``Any``. Constructor arguments are the
    task's build-time config (captured into ``self._task_config``; read in the
    body as ``self.*``); ``ctx`` delivers only the runtime input.
    """

    @abstractmethod
    async def execute(self, ctx: TaskContext[StateT, InputT]) -> OutputT:
        """Run this task and return the output."""
        ...


class Actor[StateT, InputT, OutputT](_CapturesInitConfig, ABC):
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
