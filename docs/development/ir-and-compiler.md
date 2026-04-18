# Compiler Internals

This page is for contributors who want to understand — or modify — how `WorkflowSpec` is turned into an executable graph. End-users should read the [Quick Start](../tutorial/quick-start.md) and [Compiler](compiler.md) instead.

There is **no intermediate JSON representation**. Compilation happens entirely in memory, on demand, the first time `spec.execute()` / `spec.start()` is called. The implementation lives in `molexp.workflow._pydantic_graph.*` and is treated as private to the workflow package.

## Source Layout

```
src/molexp/workflow/
├── spec.py                # WorkflowSpec, WorkflowDSL, WorkflowBuilder, parallel_map, join
├── task.py                # Task / Actor convenience base classes
├── context.py             # TaskContext / ActorContext
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

## Compilation Pipeline

### 1. Authoring → `WorkflowSpec`

`WorkflowDSL` / `WorkflowBuilder` collect `TaskRegistration` records (name, callable or instance, `depends_on`, `is_actor`, optional `remote=`). `build()` produces:

```python
WorkflowSpec(
    name=...,
    workflow_id=_stable_workflow_id(name, tasks),   # sha256[:16]
    tasks=[TaskRegistration(...), ...],
    mode="batch" | "streaming",
)
```

`_stable_workflow_id` hashes `name + [(task_name, fn_or_class.__qualname__, sorted(depends_on))]` so the ID survives cosmetic code edits but changes the moment you add / remove an edge or swap a class.

### 2. Validation + Levelling

`WorkflowGraphCompiler.compile(spec)` runs:

```python
import graphlib

sorter = graphlib.TopologicalSorter({t.name: set(t.depends_on) for t in spec._tasks})
sorter.prepare()          # raises CycleError on cycles / missing deps
```

After validation, tasks are grouped into **levels** by BFS distance from roots — same-level tasks share no transitive dependency on each other. Each `_StepEntry` carries the resolved callable, its `is_actor` flag, and its computed level.

### 3. Trampoline Graph

Instead of emitting one pydantic-graph node per task, the compiler emits a **single** `WorkflowStep` node and iterates over the level list. On each invocation, `WorkflowStep`:

1. Pops the next level.
2. `asyncio.gather`s one coroutine per task in that level.
3. Feeds results back into `WorkflowState`.
4. Transitions either to another `WorkflowStep` (next level) or to `End(WorkflowState)`.

This keeps the compiled graph topology independent of the user's DAG size — whether the spec has 5 tasks or 500, pydantic-graph always sees one node type and one trampoline.

### 4. State and Deps Injection

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

`WorkflowSpec._get_runtime()` lazily instantiates `GraphWorkflowRuntime` and caches it on the spec. `WorkflowRuntime.execute(...)` is the abstract base (`runtime.py`):

```python
class WorkflowRuntime(ABC):
    async def execute(self, spec, run=None, run_context=None, *,
                      profile_config=None, **kwargs) -> WorkflowResult: ...
    async def start(self, spec, run=None, run_context=None, *,
                    profile_config=None, **kwargs) -> WorkflowExecution: ...
```

A swap-in alternative runtime would need to honor the same signature and consume a `WorkflowSpec` directly — for example a Dask-backed runtime could reinterpret `spec._tasks` against a Dask futures graph. In practice `GraphWorkflowRuntime` is the only one we ship.

## Caching

`Caching` (`cache.py`) is orthogonal to the runtime — tasks that opt in call `cache.get(snapshot, inputs)` / `cache.put(...)` themselves. The key is `f"{snapshot.key}:{input_hash}"` where `snapshot.key = f"{code_hash}:{config_hash}"`. Code hashes come from `TaskSnapshot._compute_code_hash`, which AST-normalizes `execute.__source__` (strip decorators / comments / whitespace) and falls back to bytecode hashing if the source isn't available.

## Sweep-Level Graph

`molexp.sweep.graph.SweepRoot` is a second, even simpler pydantic-graph node used at the outer level: one node per `run_sweep` call, fanning out replicas with a bounded `asyncio.Semaphore(jobs)`. It catches per-replica exceptions and records them in `SweepState.failures` instead of cancelling peers. Future phases (see `docs/spec/unified-pydantic-graph-dispatch.md`) will specialize the single-node graph to one-node-per-replica to unlock per-replica backend routing.

## Extension Points

If you're adding features, keep the boundaries:

- **New DSL surface** → add to `spec.py`, make sure `_stable_workflow_id` still produces a stable hash.
- **Different runtime** → subclass `WorkflowRuntime`, expose via a factory similar to `create_default_runtime()`.
- **New cache strategy** → extend `Caching` or compose it with a different storage backend; don't teach the runtime about caching directly.
- **New snapshot semantics** → change `TaskSnapshot`; bump `CACHE_FORMAT_VERSION` in `cache.py` so stale cache entries are invalidated.

All other public behaviour should route through `molexp.workflow`'s re-exports.
