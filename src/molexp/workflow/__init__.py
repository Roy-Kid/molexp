"""Molexp workflow layer — public OOP API.

Define a workflow as an instance of :class:`Workflow` and register tasks
via its methods. Three equivalent styles share the same class:

1. **Decorator** (functions as tasks)::

       wf = Workflow(name="pipeline")


       @wf.task
       async def fetch(ctx: TaskContext) -> FetchResult: ...


       result = await wf.build().execute()

2. **OOP** (subclass ``Task`` and ``.add()``)::

       class FetchTask(Task):
           async def execute(self, ctx: TaskContext) -> FetchResult: ...


       wf = Workflow(name="pipeline").add(FetchTask())

3. **Protocol** (any object with ``async execute(ctx)``)::

       class ExternalProcessor:
           async def execute(self, ctx) -> dict: ...


       wf = Workflow(name="pipeline").add(ExternalProcessor())

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
from .promote import (
    promote_callable,
    resolve_callable_entrypoint,
    resolve_spec_entrypoint,
)
from .protocols import Runnable, Streamable
from .registry import TaskTypeRegistry, default_registry
from .snapshot import TaskSnapshot
from .snapshot_ref import WorkflowSnapshotRef
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
    "Runnable",
    "Streamable",
    # Convenience base classes
    "Task",
    # Contexts
    "TaskContext",
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
    # Workflow building (unified OOP API)
    "Workflow",
    # Compiler (IR ↔ Python ↔ Mermaid ↔ Spec)
    "WorkflowCompiler",
    "WORKFLOW_CACHE_SUBSYSTEM_KIND",
    "WorkflowDeadlockError",
    # Errors
    "WorkflowError",
    "WorkflowExecution",
    # Execution
    "WorkflowResult",
    # Snapshot reference (on-disk shape stored in run.json)
    "WorkflowSnapshotRef",
    "WorkflowSpec",
    "WorkflowVersion",
    "WorkflowVersionConflictError",
    "WorkspaceCacheStore",
    "default_compiler",
    "default_registry",
    "make_execution_id",
    # Callable → WorkflowSpec promotion (worker re-import support)
    "promote_callable",
    "resolve_callable_entrypoint",
    "resolve_spec_entrypoint",
]
