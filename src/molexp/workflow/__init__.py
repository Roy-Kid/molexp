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
from .cache_store import CacheStore, FileCacheStore
from .codec import WorkflowCodec, default_codec
from .context import ActorContext, TaskContext
from .contract import (
    ArtifactDecl,
    Severity,
    TaskInputSpec,
    TaskIO,
    TaskOutputSpec,
    ValidationCheck,
    ValidationCheckId,
    ValidationIssue,
    ValidationReport,
    WorkflowContract,
    default_validation_checks,
    validate_workflow_contract,
)
from .ir import (
    GraphLoopIR,
    GraphParallelIR,
    GraphTaskIR,
    WorkflowGraphIR,
    build_workflow_graph_ir,
)
from .mermaid import render_workflow_mermaid
from .promote import promote_callable, resolve_callable_entrypoint, resolve_spec_entrypoint
from .protocols import Runnable, Streamable
from .registry import TaskTypeRegistry, default_registry
from .snapshot import TaskSnapshot
from .snapshot_ref import WorkflowSnapshotRef
from .spec import Workflow
from .task import Actor, Task
from .types import (
    BranchEdges,
    CycleError,
    EdgeShapeError,
    End,
    EntryAmbiguousError,
    LoopMaxItersExceeded,
    MissingRouteError,
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
    "Actor",
    "ActorContext",
    "ArtifactDecl",
    "BranchEdges",
    "CacheStore",
    "Caching",
    "CycleError",
    "EdgeShapeError",
    "End",
    "EntryAmbiguousError",
    "FileCacheStore",
    "GraphLoopIR",
    "GraphParallelIR",
    "GraphTaskIR",
    "LoopMaxItersExceeded",
    "MissingRouteError",
    "OutEdges",
    "ParallelExecutionError",
    "Runnable",
    "Severity",
    "Streamable",
    "Task",
    "TaskContext",
    "TaskIO",
    "TaskInputSpec",
    "TaskOutputSpec",
    "TaskSnapshot",
    "TaskTopologyEntry",
    "TaskTypeRegistry",
    "UnconditionalEdges",
    "UnknownRouteError",
    "UnknownTaskError",
    "UnreachableTaskError",
    "ValidationCheck",
    "ValidationCheckId",
    "ValidationIssue",
    "ValidationReport",
    "Workflow",
    "WorkflowBuilder",
    "WorkflowCodec",
    "WorkflowContract",
    "WorkflowDeadlockError",
    "WorkflowError",
    "WorkflowExecution",
    "WorkflowGraphIR",
    "WorkflowResult",
    "WorkflowSnapshotRef",
    "WorkflowVersion",
    "WorkflowVersionConflictError",
    "build_workflow_graph_ir",
    "default_codec",
    "default_registry",
    "default_validation_checks",
    "generate_name",
    "make_execution_id",
    "promote_callable",
    "render_workflow_mermaid",
    "resolve_callable_entrypoint",
    "resolve_spec_entrypoint",
    "validate_workflow_contract",
]
