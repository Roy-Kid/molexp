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

## Sweeps

When the real problem is not one workflow run but a whole collection of `(experiment, run)` pairs, MolExp provides `molexp.sweep.run_sweep`. That utility owns the outer loop while still delegating each replica to the same underlying workflow execution path. One failed replica does not automatically cancel the others. Instead, sweep state collects outputs and failures separately so the caller can inspect the whole batch after execution finishes.

## The Useful Boundary

The most useful way to think about the runtime is that it is responsible for turning a compiled workflow definition into one concrete execution, possibly under a persistent run record. It is not where you should model research structure, scheduler policy, or asset scoping. Those remain separate concerns, which is why the runtime can stay small even as the rest of the system grows around it.

## Runnable Example

`examples/workflow/workflow_runtime.py` shows both entry points — `spec.execute()` blocking to completion and `spec.start()` returning a `WorkflowExecution` handle.
