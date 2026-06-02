"""Molexp workflow layer — public OOP API.

Define a workflow by instantiating :class:`WorkflowCompiler`, registering
tasks via its methods, then calling :meth:`WorkflowCompiler.compile` to
produce the frozen, content-addressed :class:`CompiledWorkflow` (graph +
per-task snapshots + version + optional experiment binding). Three
equivalent styles share the compiler class:

1. **Decorator** (functions as tasks)::

       wf = WorkflowCompiler(name="pipeline")


       @wf.task
       async def fetch(ctx: TaskContext) -> FetchResult: ...


       compiled = wf.compile()
       result = await WorkflowRuntime().execute(compiled)

2. **OOP** (subclass ``Task`` and ``.add()``)::

       class FetchTask(Task):
           async def execute(self, ctx: TaskContext) -> FetchResult: ...


       compiled = WorkflowCompiler(name="pipeline").add(FetchTask()).compile()

3. **Protocol** (any object with ``async execute(ctx)``)::

       class ExternalProcessor:
           async def execute(self, ctx) -> dict: ...


       compiled = WorkflowCompiler(name="pipeline").add(ExternalProcessor()).compile()

Execution lives on :class:`WorkflowRuntime`
(``runtime.execute(compiled)`` / ``.start`` / ``.run_on``), not on the
artifact. Bind a compiled workflow to an experiment via
``wf.compile(experiment=exp)`` or
:data:`default_binding_registry`.bind(exp, compiled)`` so that downstream
code (CLI / server / cluster workers) can recover it via
``default_binding_registry.for_experiment(experiment)``.
"""

from ._names import generate_name
from ._pydantic_graph.runtime import WorkflowRuntime, make_execution_id
from .binding import WorkflowBinding, WorkflowBindingRegistry, default_binding_registry
from .cache import Caching
from .cache_store import CacheStore, FileCacheStore
from .codec import WorkflowCodec, default_codec
from .compiled import CompiledWorkflow
from .compiler import WorkflowCompiler
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
from .sweep import SweepMap
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
    "CompiledWorkflow",
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
    "SweepMap",
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
    "WorkflowBinding",
    "WorkflowBindingRegistry",
    "WorkflowCodec",
    "WorkflowCompiler",
    "WorkflowContract",
    "WorkflowDeadlockError",
    "WorkflowError",
    "WorkflowExecution",
    "WorkflowGraphIR",
    "WorkflowResult",
    "WorkflowRuntime",
    "WorkflowSnapshotRef",
    "WorkflowVersion",
    "WorkflowVersionConflictError",
    "build_workflow_graph_ir",
    "default_binding_registry",
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
