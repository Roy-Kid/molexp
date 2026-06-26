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

Control flow beyond the DAG shape is declared on the compiler:
``wf.parallel`` (runtime-sized fan-out), ``wf.branch`` (label-routed
edges) and ``wf.loop`` (repeat-until). A branch or loop-``until`` task
returns ``(value, Next("label"))``; the routed target receives ``value``
as its ``ctx.inputs`` (values-on-edges delivery — see
``docs/guide/control-flow.md``).

Execution lives on :class:`WorkflowRuntime`
(``runtime.execute(compiled)`` / ``.start`` / ``.run_on``), not on the
artifact. Bind a compiled workflow to an experiment via
``wf.compile(experiment=exp)`` or
:data:`default_binding_registry`.bind(exp, compiled)`` so that downstream
code (CLI / server / cluster workers) can recover it via
``default_binding_registry.for_experiment(experiment)``.
"""

from ._names import generate_name
from ._pydantic_graph.persistence import read_node_outputs, seed_from_execution
from ._pydantic_graph.runtime import WorkflowRuntime, make_execution_id
from .binding import WorkflowBinding, WorkflowBindingRegistry, default_binding_registry
from .cache import Caching
from .cache_store import CacheStore, FileCacheStore
from .codec import WorkflowCodec, default_codec
from .command_task import CommandTask
from .compiled import CompiledWorkflow
from .compiler import WorkflowCompiler
from .context import TaskContext
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
    EdgeKind,
    GraphEdgeIR,
    GraphLoopIR,
    GraphNodePosition,
    GraphParallelIR,
    GraphTaskIR,
    WorkflowGraphIR,
    build_workflow_graph_ir,
)
from .mermaid import render_workflow_mermaid
from .outputs import RegisterArtifact, RegisterMetric
from .promote import promote_callable, resolve_callable_entrypoint, resolve_spec_entrypoint
from .protocols import Runnable, Streamable
from .registry import TaskTypeRegistry, default_registry
from .snapshot import TaskSnapshot
from .snapshot_ref import WorkflowSnapshotRef
from .subworkflow import SubWorkflow
from .sweep import SweepMap
from .task import Actor, Task
from .types import (
    BranchEdges,
    CommandError,
    CycleError,
    EdgeShapeError,
    End,
    EntryAmbiguousError,
    LoopMaxItersExceeded,
    MissingRouteError,
    MissingUpstreamResultError,
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
    "Actor",
    "ArtifactDecl",
    "BranchEdges",
    "CacheStore",
    "Caching",
    "CommandError",
    "CommandTask",
    "CompiledWorkflow",
    "CycleError",
    "EdgeKind",
    "EdgeShapeError",
    "End",
    "EntryAmbiguousError",
    "FileCacheStore",
    "GraphEdgeIR",
    "GraphLoopIR",
    "GraphNodePosition",
    "GraphParallelIR",
    "GraphTaskIR",
    "LoopMaxItersExceeded",
    "MissingRouteError",
    "MissingUpstreamResultError",
    "Next",
    "OutEdges",
    "ParallelExecutionError",
    "RegisterArtifact",
    "RegisterMetric",
    "Runnable",
    "Severity",
    "Streamable",
    "SubWorkflow",
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
    "read_node_outputs",
    "render_workflow_mermaid",
    "resolve_callable_entrypoint",
    "resolve_spec_entrypoint",
    "seed_from_execution",
    "validate_workflow_contract",
]
