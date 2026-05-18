"""Molexp workflow layer — public OOP API.

Define a workflow by instantiating :class:`WorkflowBuilder`, registering
tasks via its methods, then calling :meth:`WorkflowBuilder.build` to
produce the frozen, content-addressed :class:`Workflow`. Three equivalent
styles share the builder class:

1. **Decorator** (functions as tasks)::

       wf_builder = WorkflowBuilder(name="pipeline")


       @wf_builder.task
       async def fetch(ctx: TaskContext) -> FetchResult: ...


       wf = wf_builder.build()
       result = await wf.execute()

2. **OOP** (subclass ``Task`` and ``.add()``)::

       class FetchTask(Task):
           async def execute(self, ctx: TaskContext) -> FetchResult: ...


       wf = WorkflowBuilder(name="pipeline").add(FetchTask()).build()

3. **Protocol** (any object with ``async execute(ctx)``)::

       class ExternalProcessor:
           async def execute(self, ctx) -> dict: ...


       wf = WorkflowBuilder(name="pipeline").add(ExternalProcessor()).build()

Bind a built :class:`Workflow` to an experiment via
``wf.bind_to(experiment)`` so that downstream code (CLI / server /
cluster workers) can recover it via
:meth:`Workflow.for_experiment(experiment) <Workflow.for_experiment>`.
"""

from ._names import generate_name
from ._pydantic_graph.runtime import make_execution_id
from .builder import WorkflowBuilder
from .cache import Caching
from .cache_store import CacheStore
from .context import ActorContext, TaskContext
from .contract import WorkflowContract
from .promote import promote_callable, resolve_callable_entrypoint, resolve_spec_entrypoint
from .protocols import Runnable, Streamable
from .registry import TaskTypeRegistry, default_registry
from .serializer import WorkflowCompiler, default_compiler
from .spec import Workflow
from .task import Actor, Task
from .types import End, WorkflowError, WorkflowExecution, WorkflowResult
from .version import WorkflowVersion

__all__ = [
    "Actor",
    "ActorContext",
    "CacheStore",
    "Caching",
    "End",
    "Runnable",
    "Streamable",
    "Task",
    "TaskContext",
    "TaskTypeRegistry",
    "Workflow",
    "WorkflowBuilder",
    "WorkflowCompiler",
    "WorkflowContract",
    "WorkflowError",
    "WorkflowExecution",
    "WorkflowResult",
    "WorkflowVersion",
    "default_compiler",
    "default_registry",
    "generate_name",
    "make_execution_id",
    "promote_callable",
    "resolve_callable_entrypoint",
    "resolve_spec_entrypoint",
]
