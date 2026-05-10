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

Layer position: **workflow uses workspace, agent uses both**. The
graph algorithm (compiler, scheduler, IR round-trip) is workspace-
agnostic, but caching and run-state persistence delegate through
workspace storage primitives — :class:`WorkspaceCacheStore`
(backed by ``Workspace.subsystem_store("workflow.cache")``) for the
content-addressed result cache, and :func:`molexp.workspace.atomic_write_json`
for ``workflow.json`` snapshots under the run dir. Cross-layer
payloads coming *down* from the agent flow through duck-typed
``run_context`` (opaque) or ``Mapping[str, JSONValue]`` config — see
``§ Layer charters`` in CLAUDE.md and the import-guard tests under
``tests/test_workflow/`` for the binding rules.
"""

from ._pydantic_graph.runtime import make_execution_id
from .cache import Caching
from .cache_store import (
    WORKFLOW_CACHE_SUBSYSTEM_KIND,
    CacheStore,
    FileCacheStore,
    WorkspaceCacheStore,
)
from .compiler import (
    WorkflowCompiler,
    default_compiler,
)
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
from .promote import (
    promote_callable,
    resolve_callable_entrypoint,
    resolve_spec_entrypoint,
)
from .protocols import Runnable, Streamable
from .registry import TaskTypeRegistry, default_registry
from .snapshot import TaskSnapshot
from .snapshot_ref import WorkflowSnapshotRef
from .spec import Workflow, WorkflowBuilder
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
    RepairBudgetExceeded,
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
    # Sidecar contract layer (typed I/O + artifact decls + validation)
    "ArtifactDecl",
    "BranchEdges",
    # Utilities
    "CacheStore",
    "Caching",
    "CycleError",
    "EdgeShapeError",
    "FileCacheStore",
    # Workflow terminator (re-exported from pydantic_graph)
    "End",
    "EntryAmbiguousError",
    # Warnings (non-fatal)
    "LoopMaxItersExceeded",
    "MissingRouteError",
    "OutEdges",
    "ParallelExecutionError",
    # Protocols (for third-party integration)
    "RepairBudgetExceeded",
    "Runnable",
    "Severity",
    "Streamable",
    # Convenience base classes
    "Task",
    # Contexts
    "TaskContext",
    "TaskIO",
    "TaskInputSpec",
    "TaskOutputSpec",
    "TaskSnapshot",
    # Versioning
    "TaskTopologyEntry",
    # Task-type registry (for IR-driven workflows)
    "TaskTypeRegistry",
    # Edge sum types (declarative IR vocabulary)
    "UnconditionalEdges",
    "UnknownRouteError",
    "UnknownTaskError",
    "UnreachableTaskError",
    "ValidationCheck",
    "ValidationCheckId",
    "ValidationIssue",
    "ValidationReport",
    # Workflow building (unified OOP API)
    "Workflow",
    # Compiler (IR ↔ Python ↔ Mermaid ↔ Spec)
    "WorkflowCompiler",
    "WORKFLOW_CACHE_SUBSYSTEM_KIND",
    # Sidecar contract wrapper
    "WorkflowContract",
    "WorkflowDeadlockError",
    # Errors
    "WorkflowError",
    "WorkflowExecution",
    # Execution
    "WorkflowResult",
    # Snapshot reference (on-disk shape stored in run.json)
    "WorkflowSnapshotRef",
    "WorkflowVersion",
    "WorkflowVersionConflictError",
    # Builder for the Workflow (decorator + OOP, calls .build() to freeze)
    "WorkflowBuilder",
    "WorkspaceCacheStore",
    "default_compiler",
    "default_registry",
    # Default validation check tuple (applied when contract.validation_checks is empty)
    "default_validation_checks",
    "make_execution_id",
    # Callable → Workflow promotion (worker re-import support)
    "promote_callable",
    "resolve_callable_entrypoint",
    "resolve_spec_entrypoint",
    # Static contract validation entry point
    "validate_workflow_contract",
]
