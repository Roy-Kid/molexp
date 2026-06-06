# Workflow Layer Architecture

`molexp.workflow` is the only workflow abstraction in molexp. Every
graph-shaped scientific workflow — planning, executable, repair,
dry-run, etc. — must be represented through this layer.

`molexp.workflow` may internally use `pydantic-graph` for state-machine
plumbing, but that dependency is private: anything that imports
`pydantic_graph` directly must live under
`src/molexp/workflow/_pydantic_graph/`. `WorkflowStep` is the only
class molexp exposes to pg as a `BaseNode`; user-side `Task` and
`Actor` do not subclass `BaseNode`.

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

- `Workspace.subsystem_store("workflow.cache")` — backs
  `WorkspaceCacheStore`, the content-addressed result cache. The
  user-home `~/.molexp/cache/` shortcut is gone.
- `workspace.atomic_write_json` — used by `RunStorePersistence` to
  write `workflow.json` snapshots under each run's
  `executions/<exec_id>/` directory. Atomicity is workspace's
  guarantee, not a workflow-layer reinvention.
- `workspace.Run`, `workspace.RunContext` — accepted as the canonical
  execution unit by `Workflow.execute(run=run)` / `Workflow.start(...)`.

The workflow layer does **not** import from `molexp.agent`,
`molexp.plugins`, `molexp.server`, `molexp.cli`, or `molexp.sweep`.
Cross-layer payloads coming *down* from the agent (e.g. opaque
RunContext-shaped objects, `Mapping[str, JSONValue]` config) flow
through duck-typed parameters that the workflow scheduler treats as
opaque.

## Responsibilities

`molexp.workflow` owns:

- workflow declaration (`Workflow` builder, `Workflow` compiled)
- task / actor abstractions (`Task`, `Actor`, the single `TaskContext`,
  plus the structural `Runnable` / `Streamable` protocols)
- task-type registry (`TaskTypeRegistry`) for IR-driven round-trip
- snapshotting and content-addressed identity (`TaskSnapshot`,
  `WorkflowVersion`)
- **caching**: `Caching` orchestrates the cache policy (key
  derivation, format version, LRU eviction) on top of a pluggable
  `CacheStore` (`FileCacheStore` for plain directories,
  `WorkspaceCacheStore` for workspace-rooted caches)
- **persistence**: `RunStorePersistence` (a pg `BaseStatePersistence`
  subclass) writes a single `workflow.json` per execution attempt
  through workspace's atomic-write helper
- the IR ↔ Python ↔ Mermaid codec (`WorkflowCodec`)
- declarative IR sugar (`wf.loop` / `wf.parallel` / `wf.branch`)
- the `WorkflowStep` scheduler — the sole `pydantic_graph.BaseNode`
  subclass molexp exposes to pg, wrapping the entire frontier-advance
  scheduler (data deps, branching, loops, parallel, `max_concurrency`)
- the `End` re-export — `molexp.workflow.End is pydantic_graph.End`

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
    Workflow,
    Workflow,
    Task,
    Actor,
    TaskContext,
    Caching,
    WorkspaceCacheStore,
    promote_callable,
    WorkflowSnapshotRef,
)
```

Forbidden outside `molexp.workflow`:

```python
from pydantic_graph import Graph, BaseNode  # pg is workflow's private dep
import pydantic_graph
import molexp.workflow._pydantic_graph        # private subtree
```

The import-boundary firewall is enforced by
`tests/test_workflow/test_import_guard.py` (forbids upstream layers,
confines `pydantic_graph` to `_pydantic_graph/`) and
`tests/test_workflow/test_pydantic_graph_boundary.py` (`WorkflowStep`
is the sole `BaseNode`, no duplicate `End` sentinel, etc.).
