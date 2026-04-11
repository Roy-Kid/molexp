"""Structural protocols for workflow nodes.

Any object matching these protocols can participate in a molexp workflow
**without importing molexp**. This enables zero-dependency integration
with third-party libraries.

Three ways to define a task — all equivalent at runtime:

1. Function decorated with ``@wf.task``::

       @wf.task
       async def fetch(ctx): ...

2. Subclass ``Task`` (convenience base, optional)::

       class Fetch(Task):
           async def execute(self, ctx: TaskContext) -> dict: ...

3. Any object with a matching method (third-party, zero import)::

       # In another package — no molexp dependency needed
       class ExternalFetch:
           async def execute(self, ctx) -> dict: ...
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Protocol, runtime_checkable


@runtime_checkable
class Runnable(Protocol):
    """Protocol for batch task nodes.

    Any object with ``async execute(ctx) -> output`` qualifies.
    The ``ctx`` argument will be a :class:`~molexp.workflow.context.TaskContext`
    at runtime, but the protocol deliberately types it as ``Any`` so that
    third-party implementations need not import molexp.

    Example (no molexp import needed)::

        class MyProcessor:
            async def execute(self, ctx) -> dict:
                data = ctx.inputs
                return {"processed": data}
    """

    async def execute(self, ctx: Any) -> Any: ...


@runtime_checkable
class Streamable(Protocol):
    """Protocol for streaming actor nodes.

    Any object with ``async run(ctx) -> AsyncIterator`` qualifies.

    Example::

        class MyStreamer:
            async def run(self, ctx):
                while True:
                    msg = await ctx.receive()
                    yield transform(msg)
    """

    async def run(self, ctx: Any) -> AsyncIterator[Any]: ...
