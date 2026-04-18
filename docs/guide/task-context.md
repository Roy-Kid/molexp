# TaskContext

`TaskContext` is the **single object** every user-defined task receives. It bridges the workflow type system (state, deps, inputs, config) with optional workspace capabilities (artifacts, assets, checkpoints, results).

```python
class TaskContext[StateT, DepsT, InputT]:
    state:   StateT            # shared mutable workflow state
    deps:    DepsT             # injected dependencies (user-defined)
    inputs:  InputT            # typed output from the upstream task
    config:  ProfileConfig     # active molcfg profile (read-only mapping)
    run_context: RunContext | None
```

`ActorContext` extends `TaskContext` with async `receive()` / `send()` primitives for streaming.

## Workflow-Side Properties

These are always available, with or without a workspace:

| Property | Meaning |
|----------|---------|
| `ctx.state` | Shared mutable workflow state — the same object is passed to every task in the spec. |
| `ctx.deps` | Injected dependencies (storage clients, metrics sinks, …). Provided at execute time via the runtime. |
| `ctx.inputs` | Output of the upstream task. `None` for root tasks. Multi-parent fan-in surfaces as a dict keyed by upstream name. |
| `ctx.config` | Active `ProfileConfig` — a read-only mapping of the merged molcfg data. Defaults to an empty config when no profile is active. |

### Reading config safely

`ctx.config` implements `Mapping[str, Any]` and always exists — even when the user didn't pass `profile_config=`. Defensive access via `.get()`:

```python
class Train(Task):
    async def execute(self, ctx: TaskContext) -> dict:
        lr = ctx.config.get("lr", 1e-3)
        batch = ctx.config.get("batch", 32)
        return {"lr": lr, "batch": batch}
```

`ctx.config` is **immutable inside the task**. To change effective config for a downstream task, either mutate `ctx.state` (explicit) or plumb a new dep.

## Workspace-Side Helpers

These call into the attached `RunContext` when one exists; they return `None` (or no-op) otherwise, which is what makes the same task code runnable with or without a workspace.

| Helper | Behaviour |
|--------|-----------|
| `ctx.save_artifact(name, data)` | Writes `data` under `<run_dir>/artifacts/<name>`. Dicts → JSON, bytes → binary, Paths → copied, everything else → `str(...)`. Returns the written path. |
| `ctx.get_artifact_path(name)` | Returns the path (does not check existence). |
| `ctx.find_asset(name)` | Walks `experiment → project → workspace` AssetLibraries and returns the first match (an `Asset` object with `.path`, `.meta`, …). |
| `ctx.checkpoint(name=None)` | Snapshots the current execution state under `<run_dir>/.ckpt/<ckpt_id>.json`. Returns the checkpoint ID. |
| `ctx.set_result(key, value)` | Stores a named result in the RunContext (visible to downstream tasks via `ctx.get_result`). |
| `ctx.get_result(key)` | Retrieves a previously stored result. |

`ctx.run_context` exposes the raw `RunContext` for advanced use (access to `run.parameters`, work directory, the backing `Run`, etc.).

## ActorContext (Streaming)

`ActorContext` adds two coroutines on top:

```python
class MyActor(Actor):
    async def run(self, ctx: ActorContext):
        while True:
            msg = await ctx.receive()          # blocks on the default "input" channel
            processed = transform(msg)
            await ctx.send(processed)          # emits to the "output" channel
            yield processed
```

`receive()` / `send()` currently expect channels to be registered on the backing `RunContext` (`_channels`). If you call them without a `RunContext`, they raise `NotImplementedError`.

## Typing Tips

- Plain `TaskContext` defaults to `TaskContext[Any, Any, Any]` — fine for quick prototypes.
- When you want upstream/downstream types to line up, parameterize both the upstream and downstream `TaskContext`:
  ```python
  @wf.task
  async def fetch(ctx: TaskContext[State, Deps, None]) -> dict: ...

  @wf.task(depends_on=["fetch"])
  async def process(ctx: TaskContext[State, Deps, dict]) -> Summary: ...
  ```
- Inside a `Task` subclass, the generics carry `OutputT` as well:
  ```python
  class Process(Task[State, Deps, dict, Summary]):
      async def execute(
          self, ctx: TaskContext[State, Deps, dict]
      ) -> Summary: ...
  ```

## Standalone Execution

Without a workspace `Run`, workspace helpers silently degrade:

```python
result = await spec.execute()                    # no run attached
# ctx.save_artifact(...) returns None
# ctx.find_asset(...) returns None
# ctx.checkpoint() returns None
```

This is by design — the same task code is valid in both modes. Use it for quick local iteration before wiring up a workspace.
