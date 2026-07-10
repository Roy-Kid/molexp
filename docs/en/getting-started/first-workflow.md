# Your First Workflow

Before MolExp becomes a workspace or a CLI tool, it is a workflow system. A workflow is just a compiled graph of computation steps. You can build and run that graph without creating a workspace at all.

## Define Tasks

A task is an ordinary function. It declares the data it needs as named parameters. The engine binds values by name from upstream outputs.

```python
from molexp.workflow import WorkflowCompiler

wf = WorkflowCompiler(name="pipeline")

@wf.task
def fetch() -> dict:
    return {"value": 42}

@wf.task(depends_on=["fetch"])
def scale(value: int, factor: int = 2) -> int:
    return value * factor
```

`scale` depends on `fetch`, so `fetch` runs first. Its return value `{"value": 42}` is unpacked: the `value` key binds to `scale`'s `value` parameter. The `factor` parameter has no upstream match, so it uses its default of `2`.

## Sync and Async

Sync `def` tasks run in a worker thread — a blocking computation never stalls other tasks. Use `async def` when the body genuinely awaits something:

```python
@wf.task(depends_on=["scale"])
async def publish(value: int) -> str:
    return f"published {value}"
```

The two styles mix freely in one graph.

## Compile and Run (No Workspace)

`wf.compile()` freezes the definition into a `CompiledWorkflow`. You can run it purely in memory — no projects, experiments, or directories:

```python
import asyncio
from molexp.workflow import WorkflowRuntime

compiled = wf.compile()
result = asyncio.run(WorkflowRuntime().execute(compiled))
print(result.status, result.outputs)  # succeeded {'fetch': ..., 'scale': 84, 'publish': 'published 84'}
```

This is useful during early iteration — iterate on task boundaries and data flow without thinking about persistence. When you want a durable record, hand the same `compiled` to a tracked run.

## Task Classes (Reusable)

If you prefer reusable classes over decorators, subclass `Task`:

```python
from molexp.workflow import Task, TaskContext

class Fetch(Task):
    def execute(self, ctx: TaskContext) -> dict:
        return {"value": 42}

class Scale(Task):
    def execute(self, ctx: TaskContext, value: int, factor: int = 2) -> int:
        return value * factor

compiled = (
    WorkflowCompiler(name="pipeline-oop")
    .add(Fetch())
    .add(Scale(), depends_on=["fetch"])
    .compile()
)
```

The `ctx` parameter gives you a scratch directory (`ctx.workdir`) — omit it when you don't need one. Both styles produce the same kind of compiled workflow.

## The Engine's Contract

Three rules govern every task execution:

1. **Name binding.** A parameter receives its value from the upstream output key that shares its name.
2. **Defaults as fallback.** When no upstream provides a value, the parameter's default is used. Build-time configuration (`params`) can override that default.
3. **Thread isolation for sync.** Sync bodies run in `asyncio.to_thread`. They never block the event loop.

## Next Step

Once the workflow model is clear, the natural next question is: what happens when I want a durable record? That is where the workspace layer comes in. Continue with [Track a Run](tracked-runs.md).

## Runnable Example

`examples/getting_started/02_first_workflow.py` compiles and runs a workspace-less workflow.
