"""Structural protocols for workflow nodes.

Any object matching these protocols can participate in a molexp workflow
**without importing molexp**. This enables zero-dependency integration
with third-party libraries.

Three ways to define a task — all equivalent at runtime:

1. Function registered via :meth:`Workflow.task` decorator::

       wf = Workflow(name="pipeline")


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

from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

# Re-exported from the cross-layer typing root so workflow code can use these
# without reaching into ``molexp._typing`` directly. The aliases live in one
# place to keep the layering DAG cycle-free.
from .._typing import (
    JSONMapping,
    JSONValue,
    TaskInput,
    TaskOutput,
    UserDeps,
)

__all__ = [
    "AssetsViewLike",
    "JSONMapping",
    "JSONValue",
    "RunContextLike",
    "RunLike",
    "Runnable",
    "Streamable",
    "TaskBody",
    "TaskInput",
    "TaskOutput",
    "UpstreamViewLike",
    "UserDeps",
]

if TYPE_CHECKING:
    # ``ArtifactAccessor`` types ``RunContextLike.artifact`` precisely; the
    # workflow→workspace import is layering-legal and TYPE_CHECKING-only, so it
    # adds no runtime cost and ``runtime_checkable`` still matches by attribute.
    from molexp.workspace.assets.accessors import ArtifactAccessor

    # Imported under TYPE_CHECKING only — ``task.py`` imports back into this
    # module for ``TaskInput`` / ``TaskOutput``, so a runtime import would
    # create a cycle. The PEP 695 ``type`` alias below is evaluated lazily,
    # so the names need only be resolvable to a type-checker.
    from .task import Actor, Task

# Anything that may be the body of a registered task. The runtime dispatches
# bodies via isinstance / Protocol checks (see ``_invoke_body_with_ctx``):
# OOP ``Task`` / ``Actor`` instances, third-party objects matching
# :class:`Runnable` / :class:`Streamable`, or a bare async callable.
type TaskBody = "Task | Actor | Runnable | Streamable | Callable[..., Awaitable[TaskOutput]]"


@runtime_checkable
class RunLike(Protocol):
    """Duck-typed shape of ``molexp.workspace.run.Run`` used by the workflow runtime.

    Defining the contract here as a Protocol — instead of importing the
    workspace ``Run`` class — is what lets the workflow layer remain
    independent of the workspace layer (CLAUDE.md § *Workflow ↔ pydantic-graph
    boundary*). Members are read-only properties so the concrete ``Run`` (whose
    ``id`` is a property) structurally satisfies the protocol.
    """

    @property
    def id(self) -> str: ...


@runtime_checkable
class UpstreamViewLike(Protocol):
    """View handed to a ``dependent_params`` callback per upstream task.

    Exposes the upstream task's recorded output and (when a workspace
    ``RunContext`` is attached) a producer-task-filtered asset query handle.
    """

    output: TaskOutput
    assets: AssetsViewLike | None


@runtime_checkable
class AssetsViewLike(Protocol):
    """Duck-typed shape of ``workspace.assets.AssetsView`` used by the workflow runtime.

    Only the ``.query(**filters)`` method is reached by workflow code; this
    protocol covers it without importing the workspace's ``Asset`` /
    ``AssetList`` types into the workflow layer.
    """

    def query(
        self,
        *,
        kind: str | type | None = ...,
        producer_run: str | None = ...,
        producer_task: str | None = ...,
        tag: tuple[str, str] | None = ...,
        limit: int | None = ...,
        recursive: bool = ...,
    ) -> TaskOutput: ...


@runtime_checkable
class RunContextLike(Protocol):
    """Duck-typed shape of ``workspace.run.RunContext`` used by the workflow runtime.

    Captures only the surface the workflow scheduler reaches into: the
    run reference, the work directory, the artifact accessor, and the actor
    channel methods. Members are read-only properties so the concrete
    ``RunContext`` (whose ``work_dir`` / ``run`` are properties) structurally
    satisfies the protocol. Anything else on a real ``RunContext`` is out of
    scope for the workflow layer.
    """

    @property
    def work_dir(self) -> Path: ...

    @property
    def run(self) -> RunLike: ...

    @property
    def artifact(self) -> ArtifactAccessor: ...

    async def receive(self, channel: str) -> TaskInput: ...

    async def emit(self, channel: str, message: TaskOutput) -> None: ...


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

    async def execute(self, ctx: TaskInput) -> TaskOutput: ...


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

    # Async-generator functions return their iterator on call (no
    # ``await``), so the Protocol method is declared ``def`` not
    # ``async def`` to match how ``async for`` consumes the result.
    def run(self, ctx: TaskInput) -> AsyncIterator[TaskOutput]: ...
