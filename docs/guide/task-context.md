# TaskContext

`TaskContext` is the execution boundary between a workflow definition and the code inside one task. It is the object that carries upstream inputs, shared state, injected dependencies, active profile data, and, when a run is attached, the workspace-backed helpers that let task code record results and find assets.

## The Core View of a Task

Every task sees the same basic shape:

```python
class TaskContext[StateT, DepsT, InputT]:
    state: StateT
    deps: DepsT
    inputs: InputT
    config: ProfileConfig
    run_context: RunContext | None
```

`ctx.state` is the shared mutable state object for the workflow execution. `ctx.deps` is where runtime-injected dependencies live. `ctx.inputs` is the value produced by the immediate upstream task, or a keyed mapping when fan-in creates more than one upstream parent. `ctx.config` is the resolved `ProfileConfig`, which always exists even when no named profile was selected. `ctx.run_context` is the underlying workspace execution context when the workflow is running under a `Run`.

## Reading Configuration

`ctx.config` behaves like a read-only mapping. The common pattern is to read optional settings with `.get()` and supply workflow-level defaults in task code:

```python
class Train(Task):
    async def execute(self, ctx: TaskContext) -> dict:
        lr = ctx.config.get("lr", 1e-3)
        batch = ctx.config.get("batch", 32)
        return {"lr": lr, "batch": batch}
```

That design keeps profile semantics in user code. MolExp resolves and preserves the selected profile, but it does not attach special meaning to arbitrary keys.

## Working Under a Run

When execution happens under a persistent run, `TaskContext` becomes the gateway into the workspace layer through typed accessors that delegate to the underlying `RunContext`. `ctx.artifact.save(...)` writes a file into the run's artifacts directory and registers an `ArtifactAsset` in the manifest and the catalog. `ctx.log(name)` returns a bound log handle whose `.append(line)` writes a line to `logs/<name>.log` and updates the `LogAsset`. `ctx.checkpoint(name, data=...)` writes a `CheckpointAsset` chained to the previous checkpoint. `ctx.find_asset(...)` searches the asset hierarchy from run → experiment → project → workspace. `ctx.set_result(...)` and `ctx.get_result(...)` expose the lightweight result map persisted with the run metadata.

```python
class Train(Task):
    async def execute(self, ctx: TaskContext) -> dict:
        dataset = ctx.find_asset("training-data")
        loss = 0.05
        ctx.set_result("final_loss", loss)
        ctx.artifact.save("metrics.json", {"loss": loss})
        ctx.log("train").append(f"loss={loss}")
        return {"loss": loss, "dataset": dataset.path if dataset else None}
```

These helpers are intentionally soft when no run is attached. In pure in-memory execution, `ctx.artifact`, `ctx.log(name)`, `ctx.find_asset`, and `ctx.checkpoint` all return `None`. That allows the same task code to be used during early local iteration before a workspace exists.

Assets written this way carry a `Producer` record automatically. If the runtime has called `ctx.set_active_task(task_id)` before your task ran, the task_id is attached as well, so queries like `catalog.query_assets(producer_task="train")` work without the task having to tag its own outputs. See the [Unified Asset Model](assets.md) guide for the complete picture of scopes, catalog, and per-kind subclasses.

## ActorContext

`ActorContext` extends `TaskContext` for streaming tasks. It adds `receive()` and `send()` so an actor can consume messages from the default input channel and emit messages to the default output channel:

```python
class Monitor(Actor):
    async def run(self, ctx: ActorContext):
        while True:
            message = await ctx.receive()
            processed = transform(message)
            await ctx.send(processed)
            yield processed
```

Those methods require a connected `RunContext`. Without one, they raise `NotImplementedError`, which is why actor-style execution only makes sense when the runtime has a real execution context to route through.

## Typing and Ergonomics

For quick prototypes, plain `TaskContext` is usually enough. When you want static typing to line up across a larger workflow, parameterize the upstream and downstream task contexts explicitly. Task subclasses can carry the same type information through their generic parameters, which is often the cleaner style once workflows become reusable modules rather than one-file experiments.

If you need the broader runtime lifecycle around this context object, the next page to read is [Workflow Runtime](workflow-runtime.md).

## Runnable Example

`examples/workflow/task_context.py` exercises `ctx.inputs`, `ctx.deps`, `ctx.config`, and the workspace helpers inside one tracked run.
