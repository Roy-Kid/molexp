# Workflow Layer Architecture

`molexp.workflow` is the only workflow abstraction in molexp. Every
graph-shaped scientific workflow — planning, executable, repair,
dry-run, etc. — must be represented through this layer.

The execution engine is molexp-owned: compilation lowers the workflow
to a frozen `ExecutionPlan` walked by a structural values-on-edges
engine. molexp has **no** `pydantic-graph` dependency — the `End`
sentinel is molexp's own (`molexp.workflow.types.End`), the private
engine package is `src/molexp/workflow/_engine/`, and nothing under
`src/` may import `pydantic_graph`. No class under `workflow/`
subclasses any third-party engine base class — user-side `Task` and
`Actor` included.

## Layer position

`workflow` sits *above* `workspace` in the dependency DAG. It uses
workspace storage primitives to persist its own state:

```
agent           ───────► workflow ───────► workspace
(uses both)              (uses workspace    (pure storage primitive,
                          for caching and    no upstream deps)
                          atomic JSON)
```

Concretely the workflow layer reaches downward for:

- `ws.cache.as_cache_store()` — the workspace's singleton cache
  folder, backing the content-addressed result cache. The user-home
  `~/.molexp/cache/` shortcut is gone.
- `workspace.atomic_write_json` — used by the execution-document
  writer (`_engine/persistence.py`) to write `workflow.json`
  under each run's `executions/<exec_id>/` directory. Atomicity is
  workspace's guarantee, not a workflow-layer reinvention.
- `workspace.Run`, `workspace.RunContext` — accepted as the canonical
  execution unit by `WorkflowRuntime.execute(..., run_context=ctx)` /
  `WorkflowRuntime.run_on(...)`.

The workflow layer does **not** import from `molexp.agent`,
`molexp.plugins`, `molexp.server`, or `molexp.cli`.
Cross-layer payloads coming *down* from the agent (e.g. opaque
RunContext-shaped objects, `Mapping[str, JSONValue]` config) flow
through duck-typed parameters that the workflow scheduler treats as
opaque.

## Responsibilities

`molexp.workflow` owns:

- workflow declaration (`WorkflowCompiler` builder → frozen
  `CompiledWorkflow`)
- task / actor abstractions (`Task`, `Actor`, the single `TaskContext`,
  plus the structural `Runnable` / `Streamable` protocols)
- task-type registry (`TaskTypeRegistry`) for IR-driven round-trip
- snapshotting and content-addressed identity (`TaskSnapshot`,
  `WorkflowVersion`)
- **caching**: `Caching` orchestrates the cache policy (key
  derivation, format version, LRU eviction) on top of a pluggable
  `CacheStore` (`FileCacheStore` for plain directories,
  `ws.cache.as_cache_store()` for workspace-rooted caches)
- **persistence**: the coalescing execution-document writer
  (`_engine/persistence.py` — `open_execution_document` /
  bounded-staleness flush / mandatory synchronous flush on failures
  and terminal states) writes a single `workflow.json` per execution
  attempt through workspace's atomic-write helper
- the IR ↔ Python ↔ Mermaid codec (`WorkflowCodec`)
- declarative IR sugar (`wf.loop` / `wf.parallel` / `wf.branch`)
- the structural execution engine — the frozen `ExecutionPlan`
  lowering plus `engine.run_plan`, which owns values-on-edges
  scheduling (data deps, branching, loops, parallel fan-out,
  `max_concurrency`) and structural deadlock detection
  (`WorkflowDeadlockError`, zero timing constants)
- the `End` sentinel — molexp-owned, defined in `molexp.workflow.types`

It does **not** own scheduler dispatch (Slurm, PBS, …), job
monitoring, backend-specific transport, or session orchestration.

## Editable nodes

Every workflow node carries:

- stable `node_id`
- human-readable name
- node kind
- input / output schema
- status
- provenance
- dependencies
- editable fields
- validation rules

The workflow layer exposes (or supports through its IR round-trip)
operations equivalent to: `get_node`, `patch_node`, `replace_node`,
`rewrite_node`, `remove_node`, `insert_node`,
`mark_downstream_stale`, `validate_subgraph`,
`render_subgraph_preview`. Exact method names may evolve, but the
capabilities are required.

## Public boundary

Allowed outside `molexp.workflow`:

```python
from molexp.workflow import (
    WorkflowCompiler,
    CompiledWorkflow,
    WorkflowRuntime,
    Task,
    Actor,
    TaskContext,
    Caching,
    FileCacheStore,
    promote_callable,
    WorkflowSnapshotRef,
)
```

Forbidden outside `molexp.workflow`:

```python
import pydantic_graph                 # dependency removed — forbidden everywhere in src/
import molexp.workflow._engine        # private subtree
```

The import-boundary firewall is enforced by
`tests/test_workflow/test_import_guard.py` (forbids upstream layers,
zero `pydantic_graph` imports under `workflow/`) and
`tests/test_workflow/test_engine_boundary.py` (zero `pydantic_graph`
imports anywhere under `src/`, `End` is molexp-owned in
`workflow/types.py` with no duplicate sentinel, no `BaseNode`
subclasses or new scheduler-shaped classes under `workflow/`, the
lowering compiler never builds a pg `Graph`).
