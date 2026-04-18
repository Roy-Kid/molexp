# Compilation and Identity

MolExp compiles a Python-authored workflow into a `WorkflowSpec`: an immutable, replayable plan with a deterministic identity. This page covers the three things that matter about the compilation step — validation, the `workflow_id`, and the per-task `TaskSnapshot`.

## Compilation Flow

```
@wf.task / WorkflowBuilder.add(...)  →  TaskRegistration[]
                                        │
                                        ▼
                              WorkflowSpec (name, workflow_id, tasks)
                                        │
                                        ▼     (first execute() / start())
                                        ▼
                              WorkflowGraphCompiler
                                        │
                                        ▼
                              pydantic-graph Graph  →  runtime.execute(...)
```

Two compilation boundaries:

1. `DSL → WorkflowSpec` (happens at `.build()`). Stores the task registrations and computes the `workflow_id`.
2. `WorkflowSpec → pydantic-graph Graph` (happens on first execute). Validates the DAG, groups tasks into topological levels, and wires up the trampoline node.

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

## DAG Validation

At compile time (first execute), `graphlib.TopologicalSorter` rejects:

- Cycles between tasks
- `depends_on` references to non-existent tasks
- Self-dependencies

Errors surface synchronously from `await spec.execute(...)` — before any user task code runs.

## Topological Levels and Parallelism

The compiler assigns each task a **level** (distance from the nearest root). Tasks sharing a level have no mutual dependency and run concurrently under `asyncio.gather`. You don't configure workers or pool sizes — parallelism is a consequence of the graph shape.

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

`parse` and `clean` both wait for `fetch`, then execute in parallel.

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

The combined identity key is `f"{code_hash}:{config_hash}"`. That's what the `Caching` layer uses to look up memoized outputs:

```python
from molexp.workflow import Caching

cache = Caching(store_dir=Path("~/.molexp/cache"), max_entries=1000)
cache.initialize()

hit = cache.get(snapshot, inputs)
if hit is None:
    result = await run_task(...)
    cache.put(snapshot, inputs, result)
```

## What the Compiler Does Not Do

- **It does not serialize workflows.** There is no JSON IR or `Workflow.load(...)` path. Workflows are authored in Python and re-imported on each execution. Use `workflow_id` + `WorkflowSnapshotRef` + the bound `workflow_source` for traceability.
- **It does not allocate workers or pick execution backends.** Per-task `remote=` hints are carried through but interpreted by the runtime / plugins, not the compiler.
- **It does not infer data contracts.** `depends_on` is about ordering; the types flowing through `ctx.inputs` are your responsibility (use generics on `Task` / `TaskContext` if you want static checking).

## Pointer to the Implementation

- `molexp.workflow.spec._stable_workflow_id` — workflow ID hash
- `molexp.workflow._pydantic_graph.compiler.WorkflowGraphCompiler` — spec → pydantic-graph
- `molexp.workflow.snapshot.TaskSnapshot` — code/config identity
- `molexp.workflow.cache.Caching` — LRU cache keyed by snapshot + inputs
