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

from ._pydantic_graph.runtime import make_execution_id
from .cache import Caching
from .compiler import (
    CompileError,
    WorkflowCompiler,
    compile_proposal,
    default_compiler,
)
from .context import ActorContext, TaskContext
from .preview import WorkflowPreviewView
from .proposal import (
    BranchSpec,
    InterventionPoint,
    LoopSpec,
    ParallelSpec,
    ParameterizedWorkflowSpec,
    PlanProposal,
    SanitySpec,
    SweepSpec,
    TaskProposal,
)
from .protocols import Runnable, Streamable
from .registry import TaskTypeRegistry, default_registry
from .runtime import WorkflowRuntime
from .snapshot import TaskSnapshot
from .spec import Workflow, WorkflowSpec
from .task import Actor, Task
from .types import (
    BranchEdges,
    CycleError,
    EdgeShapeError,
    End,
    EntryAmbiguousError,
    LoopMaxItersExceeded,
    MissingRouteError,
    Next,
    OutEdges,
    ParallelExecutionError,
    UnconditionalEdges,
    UnknownRouteError,
    UnknownTaskError,
    UnreachableTaskError,
    WorkflowDeadlockError,
    WorkflowError,
    WorkflowExecution,
    WorkflowResult,
)
from .version import (
    TaskTopologyEntry,
    WorkflowVersion,
    WorkflowVersionConflictError,
)

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
    # Compiler (IR ↔ Python ↔ Mermaid ↔ Spec; PlanProposal ↔ ParameterizedWorkflowSpec)
    "WorkflowCompiler",
    "default_compiler",
    "CompileError",
    "compile_proposal",
    # Plan-side data contracts (Part A.1)
    "PlanProposal",
    "TaskProposal",
    "SanitySpec",
    "ParallelSpec",
    "LoopSpec",
    "BranchSpec",
    "SweepSpec",
    "InterventionPoint",
    "WorkflowPreviewView",
    "ParameterizedWorkflowSpec",
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
    "make_execution_id",
    # Route selection sentinels (spec 03 — control edges + routes)
    "Next",
    "End",
    # Edge sum types
    "UnconditionalEdges",
    "BranchEdges",
    "OutEdges",
    # Errors
    "WorkflowError",
    "CycleError",
    "EdgeShapeError",
    "EntryAmbiguousError",
    "UnknownTaskError",
    "UnreachableTaskError",
    "UnknownRouteError",
    "MissingRouteError",
    "WorkflowDeadlockError",
    "ParallelExecutionError",
    # Warnings (non-fatal)
    "LoopMaxItersExceeded",
    # Versioning
    "TaskTopologyEntry",
    "WorkflowVersion",
    "WorkflowVersionConflictError",
]
