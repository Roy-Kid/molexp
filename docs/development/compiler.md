# Compiler Internals

This page is for contributors who need to understand — or modify — how a `Workflow` is built and then turned into an executable graph. End-users should read the [Quick Start](../getting-started/quick-start.md) first.

There is **no intermediate JSON representation**. Compilation happens in memory, on demand, the first time `spec.execute()` / `spec.start()` is called. The implementation lives in `molexp.workflow._pydantic_graph.*` and is treated as private to the workflow package.

## Compilation Flow

```
@wf.task / WorkflowBuilder.add(...)  →  TaskRegistration[]
                                        │
                                        ▼
                              Workflow (name, workflow_id, tasks)
                                        │
                                        ▼     (first execute() / start())
                                        ▼
                              WorkflowGraphCompiler
                                        │
                                        ▼
                              pydantic-graph Graph  →  runtime.execute(...)
```

Two compilation boundaries:

1. `DSL → Workflow` (at `.build()`). Stores the task registrations and computes the `workflow_id`.
2. `Workflow → pydantic-graph Graph` (on first execute). Validates the DAG, groups tasks into topological levels, and wires up the trampoline node.

## Source Layout

```
src/molexp/workflow/
├── spec.py                # Workflow, WorkflowDSL, WorkflowBuilder, parallel_map, join
├── task.py                # Task / Actor convenience base classes
├── context.py             # TaskContext (single context for batch + streaming)
├── protocols.py           # Runnable / Streamable
├── runtime.py             # WorkflowRuntime ABC, create_default_runtime()
├── cache.py               # Caching (LRU by snapshot + input hash)
├── snapshot.py            # TaskSnapshot (AST-normalized code hash + config hash)
└── _pydantic_graph/       # PRIVATE — do not import from outside
    ├── compiler.py        # WorkflowGraphCompiler: spec → pydantic-graph Graph
    ├── node.py            # WorkflowStep trampoline node + _StepEntry
    ├── runtime.py         # GraphWorkflowRuntime (the concrete runtime)
    ├── state.py           # WorkflowState, WorkflowDeps
    └── persistence.py     # pydantic-graph persistence adapters
```

The outer API (`molexp.workflow.__init__`) is the public boundary. Anything under `_pydantic_graph/` is an implementation detail and can break between releases.

## Deterministic Workflow ID

`workflow_id` is a 16-hex-character `sha256` over the workflow `name` plus, for each task, `name + class.__qualname__ + sorted(depends_on)`:

```python
spec = wf.build()
print(spec.workflow_id)   # e.g. "c3f9e2b8a7d4e1f0"
```

Properties:

- **Deterministic** — the same task topology on two machines produces the same ID.
- **Topology-scoped** — changing only the *implementation* of a task keeps the ID stable; changing edges or class identity changes it.
- **Cheap** — no source hashing happens at this level; that's `TaskSnapshot`'s job.

Workspace runs capture the bound workflow source via `WorkflowSnapshotRef(source=..., git_commit=...)` — pair that with `workflow_id` and `config_hash` when you need to group runs that share a topology + source version.

## Validation and Levelling

`WorkflowGraphCompiler.compile(spec)` runs:

```python
import graphlib

sorter = graphlib.TopologicalSorter({t.name: set(t.depends_on) for t in spec._tasks})
sorter.prepare()          # raises CycleError on cycles / missing deps
```

This rejects cycles, `depends_on` references to non-existent tasks, and self-dependencies. Errors surface synchronously from `await spec.execute(...)` — before any user task code runs.

After validation, tasks are grouped into **levels** by BFS distance from roots. Same-level tasks share no transitive dependency on each other and run concurrently under `asyncio.gather`. You don't configure workers or pool sizes — parallelism is a consequence of the graph shape.

```
fetch ─┬─► parse ─┐
       │          ├─► merge ─► write
       └─► clean ─┘
```

| Level | Tasks |
|-------|-------|
| 0 | `fetch` |
| 1 | `parse`, `clean` |
| 2 | `merge` |
| 3 | `write` |

Each `_StepEntry` carries the resolved callable, its `is_actor` flag, and its computed level.

## Trampoline Graph

Instead of emitting one pydantic-graph node per task, the compiler emits a **single** `WorkflowStep` node and iterates over the level list. On each invocation, `WorkflowStep`:

1. Pops the next level.
2. `asyncio.gather`s one coroutine per task in that level.
3. Feeds results back into `WorkflowState`.
4. Transitions either to another `WorkflowStep` (next level) or to `End(WorkflowState)`.

This keeps the compiled graph topology independent of the user's DAG size — whether the spec has 5 tasks or 500, pydantic-graph always sees one node type and one trampoline.

## State and Deps Injection

`WorkflowDeps` is a pydantic dataclass. At execute time, `CompiledWorkflow.make_deps(run=, run_context=, config=, user_deps=)` populates it with everything a task might need:

```python
deps = _DepsWithStepList(run=run, run_context=run_context, config=config, user_deps=user_deps)
deps.step_list = self._sorted_steps
deps.levels = self._levels
deps.remote_executor = remote_executor
deps.run_dir = run_dir
```

`WorkflowStep` reads `deps.step_list` / `deps.levels` to know what to run, and calls into `run_context` for workspace lifecycle events.

## Runtime Entry

`Workflow._get_runtime()` lazily instantiates `GraphWorkflowRuntime` and caches it on the spec. `WorkflowRuntime.execute(...)` is the abstract base (`runtime.py`):

```python
class WorkflowRuntime(ABC):
    async def execute(self, spec, run=None, run_context=None, *,
                      profile_config=None, **kwargs) -> WorkflowResult: ...
    async def start(self, spec, run=None, run_context=None, *,
                    profile_config=None, **kwargs) -> WorkflowExecution: ...
```

A swap-in alternative runtime must honor the same signature and consume a `Workflow` directly — for example a Dask-backed runtime could reinterpret `spec._tasks` against a Dask futures graph. In practice `GraphWorkflowRuntime` is the only one we ship.

## TaskSnapshot — Code + Config Identity

For caching and snapshotting, each task carries a `TaskSnapshot`:

```python
class TaskSnapshot:
    task_id: str
    task_type: str        # module.qualname
    code_hash: str        # sha256 of AST-normalized execute() source
    config_hash: str      # sha256 of the task's serialized config
    code_source: str
    created_at: datetime
    config_data: dict
```

The **code hash** uses AST normalization — `_normalize_ast` strips comments, whitespace, and decorators before dumping the AST. Two tasks that differ only in formatting share a hash; a real semantic change produces a new one.

The combined identity key is `f"{code_hash}:{config_hash}"`. `Caching` (`cache.py`) is orthogonal to the runtime — tasks that opt in call `cache.get(snapshot, inputs)` / `cache.put(...)` themselves. The full cache key is `f"{snapshot.key}:{input_hash}"`. Code hashes fall back to bytecode hashing if the source isn't available.

```python
from molexp.workflow import Caching, WorkspaceCacheStore, WorkflowBuilder

# Workspace-rooted cache: storage lives under
# ``<workspace_root>/.subsystems/workflow.cache/``.
cache = Caching(store=WorkspaceCacheStore(workspace), max_entries=1000)
cache.initialize()

hit = cache.get(snapshot, inputs)
if hit is None:
    result = await run_task(...)
    cache.put(snapshot, inputs, result)
```

For workspace-less callers (e.g. ad-hoc scripts), use `FileCacheStore`
or pass a plain `store_dir=...` argument:

```python
from molexp.workflow import Caching, WorkflowBuilder

cache = Caching(store_dir=Path("./cache"), max_entries=1000)
```

The user-home `~/.molexp/cache/` shortcut from earlier MolExp versions
is gone — caching is always either workspace-rooted via
`WorkspaceCacheStore` or explicitly directed to a path the caller
chose.

## What the Compiler Does Not Do

- **It does not serialize workflows.** There is no JSON IR or `Workflow.load(...)` path. Workflows are authored in Python and re-imported on each execution. Use `workflow_id` + `WorkflowSnapshotRef` + the bound `workflow_source` for traceability.
- **It does not allocate workers or pick execution backends.** Per-task `remote=` hints are carried through but interpreted by the runtime / plugins, not the compiler.
- **It does not infer data contracts.** `depends_on` is about ordering; the types flowing through `ctx.inputs` are your responsibility (use generics on `Task` / `TaskContext` if you want static checking).

## Extension Points

Keep the boundaries:

- **New DSL surface** → add to `spec.py`, make sure `_stable_workflow_id` still produces a stable hash.
- **Different runtime** → subclass `WorkflowRuntime`, expose via a factory similar to `create_default_runtime()`.
- **New cache strategy** → extend `Caching` or compose it with a different storage backend; don't teach the runtime about caching directly.
- **New snapshot semantics** → change `TaskSnapshot`; bump `CACHE_FORMAT_VERSION` in `cache.py` so stale cache entries are invalidated.

All other public behaviour should route through `molexp.workflow`'s re-exports.

## Pointer to the Implementation

- `molexp.workflow.spec._stable_workflow_id` — workflow ID hash
- `molexp.workflow._pydantic_graph.compiler.WorkflowGraphCompiler` — spec → pydantic-graph
- `molexp.workflow.snapshot.TaskSnapshot` — code/config identity
- `molexp.workflow.cache.Caching` — LRU cache keyed by snapshot + inputs
