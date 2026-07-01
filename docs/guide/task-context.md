# TaskContext

`TaskContext` is the execution boundary between a workflow definition and the code inside one task. It is deliberately small: a task declares the runtime values it consumes as **named parameters**, and the only thing it reads off the context object itself is a per-task scratch directory (`ctx.workdir`). A task cannot climb up from its context to the Run, the workspace, or injected services — that is the **pure task context** contract, and it is what makes a task's cache identity (code + config + inputs) complete.

## How a Task Receives Its Inputs

A task body declares the runtime values it consumes as **named parameters**. The engine binds each parameter *by name*; a task no longer reads its inputs or its configuration off `ctx` at all. The leading `ctx` parameter is optional: include it only when the body needs the per-task scratch directory (below).

```python
from molexp.workflow import Task, TaskContext

class Record(Task):
    async def execute(self, ctx: TaskContext, value: int, scale: int = 1) -> int:
        return value * scale
```

`value` and `scale` are filled from one merged map, `{build-time config} | {upstream outputs | run params}`, where the dynamic inputs (upstream outputs / run params) **win** over build-time config. A parameter with no matching key falls back to its declared default; a *required* parameter (no default) with no match raises `MissingTaskInputError`. The three sources each bind by name:

- **Upstream task outputs**, keyed by the upstream task's name (its `depends_on` entry). With a single upstream, a `dict` output binds by field name and a scalar output binds positionally to the sole free parameter. With multiple upstreams, each dep's `dict` output merges into one flat map (later deps win) and a scalar dep is keyed by its dep name; a `**kwargs` parameter absorbs whatever is left.
- **Run sweep params** on a root task bind by name: `params={"base": [1]}` fills `async def seed(base: int = 1)`. (The engine still wraps a root envelope internally, but the body reads run params *as named parameters*, never by indexing into an inputs mapping.)
- **Build-time config** — the `config=` kwarg of `WorkflowRuntime.execute(...)`, or a tracked run's resolved `ProfileConfig` — binds by name: `config={"scale": 10}` fills a parameter `scale: int = 1`.

## The Only Thing on `ctx`: the Workdir

When present (named `ctx`, or annotated `TaskContext`), the leading parameter receives the `TaskContext`, whose **only** data surface is `ctx.workdir`:

```python
class TaskContext[StateT, InputT]:
    workdir: Path | None      # content-addressed scratch dir for THIS task
```

`ctx.workdir` is a content-addressed scratch directory derived from the task's content identity — the sanctioned place a task writes intermediate files. It is a bare `pathlib.Path`, stable across runs for identical task content, and `None` when no workspace run is attached. A fan-out body shares one `workdir` across elements, so per-element bodies should sub-namespace it. Include `ctx` in the signature only when the body actually writes there.

!!! warning "Deprecated: `ctx.state`"
    `ctx.state` is deprecated and scheduled for removal: accessing it emits a `DeprecationWarning` and returns a **read-only snapshot** (engine state cannot be mutated through it). Everything it was used for — reading the previous loop iteration's value, picking up a branch-routed value — now arrives as **named parameters** (the values-on-edges engine delivers loop-back and branch-routed values as the body's own arguments). Migrate any remaining reads to named parameters.

There is **no** `ctx.run_context` and **no** `ctx.deps`. Capabilities that used to be reached through them — artifact persistence, asset lookup, run metadata — live on the driver-side `RunContext` (see below) or are delivered to the body as named parameters by the engine.

## Reading Configuration

Build-time and profile configuration reach a task the same way as any other input — **by name**, with the body supplying a default for anything optional. A key from `config={...}` (or the resolved `ProfileConfig`) binds to the like-named parameter:

```python
class Train(Task):
    async def execute(self, ctx: TaskContext, lr: float = 1e-3, batch: int = 32) -> dict:
        return {"lr": lr, "batch": batch}
```

Running `execute(compiled, config={"lr": 5e-4})` fills `lr`; `batch` falls back to its default. That design keeps profile semantics in user code: MolExp resolves and preserves the selected profile, but it does not attach special meaning to arbitrary keys.

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

Inside the task, the run shows up only as data: a root task's sweep `params` bind to its like-named parameters, `ctx.workdir` points into the execution directory, and the resolved profile config binds by name too. The same task code therefore runs unchanged in pure in-memory execution — there is simply no workdir, and whatever `config=` the caller passed binds the same way.

## Streaming tasks (Actor)

Streaming `Actor` bodies receive the **same** `TaskContext` as batch tasks — there is no separate context type — and bind their non-`ctx` parameters by name from the same merged map (`{config} | {upstream outputs | run params}`). The only streaming-specific behaviour is that the engine drives the async generator to exhaustion and records the **last yielded value** as the task's output:

```python
class Monitor(Actor):
    async def run(self, ctx: TaskContext, source: list[int]):
        for item in source:            # ``source`` binds the upstream output
            yield {"seen": item}       # last yield becomes the task output
```

There is no inter-task message-passing channel: an earlier `receive()` / `send()` surface was never wired (every path raised `NotImplementedError`) and has been removed. An actor yields its outputs; it does not exchange messages mid-run with peer tasks.

## Typing and Ergonomics

For quick prototypes, plain `TaskContext` is usually enough. When you want static typing to line up across a larger workflow, parameterize the upstream and downstream task contexts explicitly (`TaskContext[StateT, InputT]`). Task subclasses can carry the same type information through their generic parameters (`Task[StateT, InputT, OutputT]`), which is often the cleaner style once workflows become reusable modules rather than one-file experiments.

If you need the broader runtime lifecycle around this context object, the next page to read is [Workflow Runtime](workflow-runtime.md).

## Runnable Example

`examples/workflow/task_context.py` exercises name-bound inputs (a root-task run param, an upstream output, and build-time `config`) plus `ctx.workdir` inside one tracked run, with the workspace helpers on the driver-side `RunContext`.
