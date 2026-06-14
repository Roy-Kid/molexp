# TaskContext

`TaskContext` is the execution boundary between a workflow definition and the code inside one task. It is deliberately small: a task receives its runtime inputs and its configuration, and nothing else. A task cannot climb up from its context to the Run, the workspace, or injected services — that is the **pure task context** contract, and it is what makes a task's cache identity (code + config + inputs) complete.

## The Core View of a Task

Every task sees the same frozen shape:

```python
class TaskContext[StateT, InputT]:
    inputs: InputT            # runtime data flowing in along the graph's edges
    config: Mapping           # build-time / profile configuration (read-only)
    workdir: Path | None      # content-addressed scratch dir for THIS task
```

`ctx.inputs` is the value produced by the immediate upstream task, or a keyed mapping when fan-in joins more than one upstream parent. For a **root task of a tracked run**, the engine injects `{"params": <run params>, "workdir": <Path>}` — the run's sweep parameters and the working directory arrive *as inputs*, not through an ambient handle. Loop-back and branch-routed values are also delivered on the edges: when a task returns `(value, Next(label))`, the routed target receives `value` as its `ctx.inputs` (a declared `depends_on` interface always wins), so loop accumulation reads the previous iteration's value from `ctx.inputs`.

`ctx.config` is the active configuration mapping: the resolved `ProfileConfig` when the workflow runs under a tracked run, or the `config=` kwarg of `WorkflowRuntime.execute(...)` otherwise. It always behaves like a read-only mapping.

`ctx.workdir` is a content-addressed scratch directory derived from the task's content identity — the sanctioned place a task writes intermediate files. It is a bare `pathlib.Path`, stable across runs for identical task content, and `None` when no workspace run is attached. A fan-out body shares one `workdir` across elements, so per-element bodies should sub-namespace it.

!!! warning "Deprecated: `ctx.state`"
    `ctx.state` is deprecated and scheduled for removal: accessing it emits a `DeprecationWarning` and returns a **read-only snapshot** (engine state cannot be mutated through it). Everything it was used for — reading the previous loop iteration's value, picking up a branch-routed value — now arrives via `ctx.inputs`. Migrate any remaining reads to the inputs channel.

There is **no** `ctx.run_context` and **no** `ctx.deps`. Capabilities that used to be reached through them — artifact persistence, asset lookup, run metadata — live on the driver-side `RunContext` (see below) or are delivered as inputs by the engine.

## Reading Configuration

The common pattern is to read optional settings with `.get()` and supply workflow-level defaults in task code:

```python
class Train(Task):
    async def execute(self, ctx: TaskContext) -> dict:
        lr = ctx.config.get("lr", 1e-3)
        batch = ctx.config.get("batch", 32)
        return {"lr": lr, "batch": batch}
```

That design keeps profile semantics in user code. MolExp resolves and preserves the selected profile, but it does not attach special meaning to arbitrary keys.

## Working Under a Run

When execution happens under a persistent run, the workspace helpers live on the `RunContext` the **driver** opened via `run.start()` — outside the task bodies:

```python
from molexp.workflow import WorkflowRuntime

with run.start(profile_config=cfg) as ctx:
    result = await WorkflowRuntime().execute(compiled, run_context=ctx)
    ctx.set_result("final_loss", result.outputs["train"])
    ctx.artifact.save("metrics.json", result.outputs["train"])
    ctx.log("train").append("done")

print(run.get_result("final_loss"))   # public read-back on the Run entity
```

`ctx.set_result(...)` stores lightweight values on the run record, `ctx.artifact.save(...)` registers an `ArtifactAsset`, `ctx.log(name)` appends to a `LogAsset`, `ctx.checkpoint(...)` chains `CheckpointAsset`s, and `ctx.find_asset(...)` walks run → experiment → project → workspace. Assets written this way carry a `Producer` record automatically; while a task body is executing, the engine tags the active task id so queries like `catalog.query_assets(producer_task="train")` work. See the [Unified Asset Model](assets.md) guide for the complete picture of scopes, catalog, and per-kind subclasses.

Inside the task, the run shows up only as data: root tasks get the sweep `params` and `workdir` injected into `ctx.inputs`, and every task gets the resolved profile as `ctx.config`. The same task code therefore runs unchanged in pure in-memory execution — there is simply no workdir and whatever `config=` the caller passed.

## Streaming tasks (Actor)

Streaming `Actor` bodies receive the **same** `TaskContext` as batch tasks — there is no separate context type. An actor's `run()` is an async generator; the engine drives it to exhaustion and records the last yielded value as the task's output:

```python
class Monitor(Actor):
    async def run(self, ctx: TaskContext):
        for item in ctx.inputs:
            yield transform(item)
```

There is no inter-task message-passing channel: an earlier `receive()` / `send()` surface was never wired (every path raised `NotImplementedError`) and has been removed. An actor consumes `ctx.inputs` and yields outputs; it does not exchange messages mid-run with peer tasks.

## Typing and Ergonomics

For quick prototypes, plain `TaskContext` is usually enough. When you want static typing to line up across a larger workflow, parameterize the upstream and downstream task contexts explicitly (`TaskContext[StateT, InputT]`). Task subclasses can carry the same type information through their generic parameters (`Task[StateT, InputT, OutputT]`), which is often the cleaner style once workflows become reusable modules rather than one-file experiments.

If you need the broader runtime lifecycle around this context object, the next page to read is [Workflow Runtime](workflow-runtime.md).

## Runnable Example

`examples/workflow/task_context.py` exercises `ctx.inputs`, `ctx.config`, and `ctx.workdir` inside one tracked run, with the workspace helpers on the driver-side `RunContext`.
