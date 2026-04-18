# Workflow Runtime

The workflow runtime is the execution backend for a `WorkflowSpec`. MolExp ships a single concrete implementation — `GraphWorkflowRuntime`, backed by `pydantic-graph` — which is created lazily the first time you call `spec.execute(...)` or `spec.start(...)`.

```python
await spec.execute(run=run)                      # block-and-return
handle = await spec.start(run=run)               # async handle
```

Users rarely interact with the runtime class directly; `WorkflowSpec` hides the abstraction.

## Execution Model

`GraphWorkflowRuntime` compiles the spec via `WorkflowGraphCompiler` and runs it through `pydantic-graph`:

1. **Validate** — `graphlib.TopologicalSorter` rejects cycles and missing dependencies.
2. **Level-group** — tasks are bucketed into topological levels; same-level tasks have no data dependencies and run concurrently.
3. **Trampoline** — a single `WorkflowStep` pydantic-graph node walks the levels, fanning out each level via `asyncio.gather` and yielding control back between levels.
4. **RunContext lifecycle** — if you pass `run=`, the runtime opens the `RunContext` at entry and closes it on completion / failure (appending an `ExecutionRecord`).
5. **Persist** — pydantic-graph's persistence layer captures per-node snapshots so a killed workflow can be resumed.

The deterministic `workflow_id` (sha256 of the topology) is the key you can use to correlate runs of the same graph across machines.

## Return Types

```python
@dataclass
class WorkflowResult:
    status: str             # "completed" | "failed" | "cancelled"
    outputs: dict[str, Any] # task_name -> output
    run_id: str | None
    execution_id: str | None
```

`WorkflowExecution` (returned by `spec.start`) exposes `await handle.wait()` and `await handle.cancel()`.

## Passing Run / RunContext / Config

You can drive execution in three modes:

```python
# 1. Pure computation — no workspace
await spec.execute()

# 2. Attach a Run — the runtime opens a RunContext for you
await spec.execute(run=run)

# 3. Attach an already-opened RunContext (for hand-rolled lifecycle)
with run.start(profile_config=cfg) as ctx:
    await spec.execute(run_context=ctx)
```

Options 2 and 3 are mutually exclusive. When both `run_context` and `profile_config` are passed, the context's own config wins.

## Parallelism

Parallelism is **implicit**: same-level tasks run via `asyncio.gather`. You don't schedule workers, pick thread pools, or configure a queue — if you want the tasks to be concurrent, declare them at the same dependency level.

For CPU-bound tasks that would otherwise block the loop, wrap them with `asyncio.to_thread` inside the task body.

## Cancellation

Cancel via the execution handle or by cancelling the enclosing task group:

```python
handle = await spec.start(run=run)
try:
    result = await handle.wait()
except KeyboardInterrupt:
    await handle.cancel()
```

Cancellation marks the run as `"cancelled"` and closes the `RunContext`.

## Sweep-Level Fan-Out

For sweeping a single workflow across many `(experiment, Run)` pairs, use `molexp.sweep.run_sweep`. It owns the outer loop via a single-node `pydantic-graph` with a bounded `jobs` semaphore:

```python
from molexp.sweep import SweepReplica, run_sweep

replicas = [
    SweepReplica(mol_run=run, experiment=exp)
    for exp in project.list_experiments()
    for run in exp.list_runs()
]

state = await run_sweep(replicas, profile_config=cfg, jobs=4)
print(state.outputs)     # replica_id -> WorkflowResult
print(state.failures)    # replica_id -> "<ExcType>: <message>"
```

One replica failure does **not** cancel the others — failures are captured and reported at the end. This is the entry point the CLI uses under `molexp run`.

## Implementation Pointer

The default runtime lives in `molexp.workflow._pydantic_graph.runtime.GraphWorkflowRuntime`. It is a private module — treat it as an implementation detail and interact with the runtime only through the `WorkflowSpec` / `WorkflowRuntime` abstraction.
