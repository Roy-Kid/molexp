"""Molexp workflow layer — public API.

Three ways to define tasks:

1. **Functional** (decorator)::

       wf = workflow(name="pipeline")

       @wf.task
       async def fetch(ctx: TaskContext) -> FetchResult: ...

       result = await wf.build().execute(run=run)

2. **OOP** (subclass ``Task``)::

       class FetchTask(Task):
           async def execute(self, ctx: TaskContext) -> FetchResult: ...

       wf = WorkflowBuilder(name="pipeline").add(FetchTask()).build()

3. **Protocol** (any object with ``async execute(ctx)``)::

       # Third-party code — no molexp import required
       class ExternalProcessor:
           async def execute(self, ctx) -> dict: ...

       wf = WorkflowBuilder(name="pipeline").add(ExternalProcessor()).build()
"""

from .cache import Caching
from .context import ActorContext, TaskContext
from .protocols import Runnable, Streamable
from .runtime import WorkflowRuntime
from .snapshot import TaskSnapshot
from .spec import WorkflowBuilder, WorkflowSpec, join, parallel_map, workflow
from .task import Actor, Task
from .types import WorkflowExecution, WorkflowResult

__all__ = [
    # Protocols (for third-party integration)
    "Runnable",
    "Streamable",
    # Convenience base classes
    "Task",
    "Actor",
    # Contexts
    "TaskContext",
    "ActorContext",
    # Workflow building
    "WorkflowSpec",
    "WorkflowBuilder",
    "workflow",
    "parallel_map",
    "join",
    # Execution
    "WorkflowRuntime",
    "WorkflowResult",
    "WorkflowExecution",
    # Utilities
    "Caching",
    "TaskSnapshot",
]
