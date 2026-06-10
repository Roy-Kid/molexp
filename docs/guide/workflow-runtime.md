# Workflow Runtime

Execution lives on `WorkflowRuntime`, not on the compiled artifact. You instantiate the runtime (it is cheap and stateless apart from an optional cache) and hand it a `CompiledWorkflow`:

```python
from molexp.workflow import WorkflowRuntime

runtime = WorkflowRuntime()
result = await runtime.execute(compiled)
handle = await runtime.start(compiled)
```

The compiled artifact carries a frozen, molexp-owned `ExecutionPlan` that a structural engine executes with values-on-edges semantics (task inputs are delivered from upstream outputs; `pydantic-graph` survives only as the `End` sentinel re-export) — but most user code should treat that as an implementation detail rather than as a direct API surface.

## What the Runtime Actually Does

When execution begins, the runtime builds the initial workflow state (optionally seeded with already-known task outputs), injects root-task inputs, and drives the lowered graph to completion. Tasks whose dependencies are all satisfied run concurrently. That is why most workflow parallelism in MolExp is implicit: you declare the dependency structure, and the runtime extracts the available concurrency from that structure.

If a `RunContext` is attached (`execute(..., run_context=ctx)`), the runtime threads the run's profile config into every task's `ctx.config`, injects the run's sweep parameters and a content-addressed `workdir` into root-task `ctx.inputs`, persists per-node outputs under the run's execution directory, and back-propagates task failures so the run's final status is correct.

The runtime also relies on the compiled workflow identity rather than on ad hoc process state. `workflow_id` is derived deterministically from the workflow name and task topology, which is what makes it useful for correlating equivalent graphs across executions and machines.

## Where Persistent State Lives

Workflow execution state — the `workflow.json` snapshot under `<run_dir>/executions/<exec_id>/` — is written through workspace's public `atomic_write_json` helper, not raw filesystem calls. The atomicity guarantee is workspace's, not a runtime-layer reinvention. This is the runtime side of the *workflow → workspace* dependency direction documented in CLAUDE.md. Resume is caller-driven: a failed run preserves completed node outputs on disk, and the caller (e.g. `molexp run --resume`) re-seeds them via `execute(..., seed_outputs=...)`.

## Caching

Cache persistence is pluggable. `Caching` orchestrates the cache policy (key derivation, format version, LRU eviction); the storage primitive is a `CacheStore` implementation supplied at construction time:

- `ws.cache.as_cache_store()` — the workspace's singleton cache folder, the preferred form for runs that already have a workspace. When you pass a `run_context`, the runtime builds this workspace-backed cache automatically.
- `FileCacheStore(path)` — a plain filesystem directory. Useful when the caller has no workspace (e.g. ad-hoc scripts; the FastAPI server's process-local cache).

The user-home `~/.molexp/cache/` shortcut from earlier MolExp versions is gone — caching is always either workspace-rooted or explicitly opted into via `FileCacheStore`.

## Blocking Execution and Background Execution

`runtime.execute(...)` is the block-and-return path. It runs the workflow to completion and returns a `WorkflowResult`. `runtime.start(...)` launches the same execution as a background asyncio task and returns a `WorkflowExecution` handle, which can later be awaited (`await handle.wait()`) or cancelled (`await handle.cancel()`). The two entry points differ in control style, not in workflow semantics.

## Attaching Persistent State

There are two practical execution modes. The first is pure in-memory execution with no workspace attached. The second is execution under an opened `RunContext`:

```python
await runtime.execute(compiled)
await runtime.execute(compiled, config={"scale": 2.0})

with run.start(profile_config=cfg) as ctx:
    await runtime.execute(compiled, run_context=ctx)
```

When a live `RunContext` is passed, its profile config owns `ctx.config` (the `config=` kwarg is ignored in its favor), and the run's lifecycle — status transitions, execution records, error capture — is managed by the `with run.start()` block. For the common "build a fresh run and execute on it" shape, `runtime.run_on(compiled, experiment, parameters=...)` does both steps in one call.

## Cancellation and Failure

When execution runs through the background handle returned by `runtime.start(...)`, cancellation flows through that handle. A failed task does not raise out of `execute(...)`: the result comes back with `status="failed"`, completed outputs preserved (for `seed_outputs=` resume), and the failure recorded on the run when a `run_context` is attached. Programming errors in the workflow definition itself (cycles, unknown routes, …) do raise, as `WorkflowError` subclasses. Cancellation and failure are not special side channels; they are part of the same run history model as successful completion.

## The Useful Boundary

The most useful way to think about the runtime is that it is responsible for turning a compiled workflow definition into one concrete execution, possibly under a persistent run record. It is not where you should model research structure, scheduler policy, or asset scoping. Those remain separate concerns, which is why the runtime can stay small even as the rest of the system grows around it.

## Runnable Example

`examples/workflow/workflow_runtime.py` shows both entry points — `runtime.execute()` blocking to completion and `runtime.start()` returning a `WorkflowExecution` handle.
