# Workflow Runtime

The workflow runtime is the execution backend behind `WorkflowSpec`. In ordinary usage you do not instantiate it yourself. You call `spec.execute(...)` or `spec.start(...)`, and MolExp creates the default runtime lazily when it is needed.

```python
await spec.execute(run=run)
handle = await spec.start(run=run)
```

The concrete implementation currently shipped by MolExp is `GraphWorkflowRuntime`, backed by `pydantic-graph`, but most user code should treat that as an implementation detail rather than as a direct API surface.

## What the Runtime Actually Does

When execution begins, the runtime validates the compiled graph, groups tasks by dependency level, and then drives those levels in order. Tasks at the same level have no unresolved data dependency between them, so they can run concurrently through `asyncio.gather`. That is why most workflow parallelism in MolExp is implicit. You declare the dependency structure, and the runtime extracts the available concurrency from that structure.

If a `Run` is attached, the runtime also owns the `RunContext` lifecycle. It opens the run context at entry, applies profile metadata, appends an `ExecutionRecord`, and closes the context on success, failure, or cancellation. The task code sees that lifecycle only indirectly through `TaskContext`, but the runtime is what actually ties the workflow execution to the persistent run record.

The runtime also relies on the compiled workflow identity rather than on ad hoc process state. `workflow_id` is derived deterministically from the workflow name and task topology, which is what makes it useful for correlating equivalent graphs across executions and machines.

## Where Persistent State Lives

Workflow execution state — the `workflow.json` snapshot under
`<run_dir>/executions/<exec_id>/` — is written through workspace's
public `atomic_write_json` helper, not raw filesystem calls. The
atomicity guarantee is workspace's, not a runtime-layer reinvention.
This is the runtime side of the *workflow → workspace* dependency
direction documented in CLAUDE.md.

## Caching

Cache persistence is pluggable. `Caching` orchestrates the cache
policy (key derivation, format version, LRU eviction); the storage
primitive is a `CacheStore` impl supplied at construction time:

- `WorkspaceCacheStore(workspace)` — content-addressed entries land
  under `<workspace_root>/.subsystems/workflow.cache/`. This is the
  preferred form for in-process workflow runs that already have a
  workspace.
- `FileCacheStore(path)` — a plain filesystem directory. Useful when
  the caller has no workspace (e.g. ad-hoc scripts; the FastAPI
  server's process-local cache).

The user-home `~/.molexp/cache/` shortcut from earlier MolExp
versions is gone — caching is always either workspace-rooted or
explicitly opted into via `FileCacheStore`.

## Blocking Execution and Background Execution

`spec.execute(...)` is the block-and-return path. It runs the workflow to completion and returns a `WorkflowResult`. `spec.start(...)` launches the same execution through an async handle and returns a `WorkflowExecution`, which can later be awaited or cancelled. The two entry points differ in control style, not in workflow semantics.

## Attaching Persistent State

There are three practical execution modes. The first is pure in-memory execution with no workspace attached. The second is execution under a `Run`, where the runtime opens the run context for you. The third is execution under an already-opened `RunContext`, which is useful only when you need to manage that lifecycle manually.

```python
await spec.execute()
await spec.execute(run=run)

with run.start(profile_config=cfg) as ctx:
    await spec.execute(run_context=ctx)
```

The second and third forms should not be mixed. If you already have a live `RunContext`, that context owns the active config and the runtime should not be asked to open another one around it.

## Cancellation and Failure

When execution runs through the background handle returned by `spec.start(...)`, cancellation flows through that handle. A cancelled execution closes the run context and marks the run accordingly. A failed execution records structured error information and closes the same lifecycle. In other words, cancellation and failure are not special side channels; they are part of the same run history model as successful completion.

## The Useful Boundary

The most useful way to think about the runtime is that it is responsible for turning a compiled workflow definition into one concrete execution, possibly under a persistent run record. It is not where you should model research structure, scheduler policy, or asset scoping. Those remain separate concerns, which is why the runtime can stay small even as the rest of the system grows around it.

## Runnable Example

`examples/workflow/workflow_runtime.py` shows both entry points — `spec.execute()` blocking to completion and `spec.start()` returning a `WorkflowExecution` handle.
