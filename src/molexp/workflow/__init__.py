"""Molexp workflow layer — public OOP API.

Define a workflow as an instance of :class:`Workflow` and register tasks
via its methods. Three equivalent styles share the same class:

1. **Decorator** (functions as tasks)::

       wf = Workflow(name="pipeline")

       @wf.task
       async def fetch(ctx: TaskContext) -> FetchResult: ...

       result = await wf.build().execute(run=run)

2. **OOP** (subclass ``Task`` and ``.add()``)::

       class FetchTask(Task):
           async def execute(self, ctx: TaskContext) -> FetchResult: ...

       wf = Workflow(name="pipeline").add(FetchTask())

3. **Protocol** (any object with ``async execute(ctx)``)::

       class ExternalProcessor:
           async def execute(self, ctx) -> dict: ...

       wf = Workflow(name="pipeline").add(ExternalProcessor())
"""

from .cache import Caching
from .context import ActorContext, TaskContext
from .protocols import Runnable, Streamable
from .registry import TaskTypeRegistry, default_registry
from .runtime import WorkflowRuntime
from .snapshot import TaskSnapshot
from .spec import Workflow, WorkflowSpec
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
    # Workflow building (unified OOP API)
    "Workflow",
    "WorkflowSpec",
    # Execution
    "WorkflowRuntime",
    "WorkflowResult",
    "WorkflowExecution",
    # Task-type registry (for IR-driven workflows)
    "TaskTypeRegistry",
    "default_registry",
    # Utilities
    "Caching",
    "TaskSnapshot",
]
