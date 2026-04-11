# Quick Start

Let's build a complete MolExp workflow in five minutes — from defining tasks to execution and workspace tracking.

## Step 1: Define Tasks

A **Task** is the basic computation unit. Subclass `Task` and implement `execute()`. The `TaskContext` gives you access to upstream outputs, shared state, and workspace capabilities.

```python
from molexp.workflow import Task, TaskContext

class SquareTask(Task):
    async def execute(self, ctx: TaskContext) -> float:
        value: float = ctx.inputs
        return value * value
```

For streaming tasks, use `Actor` with `run()` instead.

## Step 2: Compose a Workflow

Two equivalent styles are available.

**Functional DSL** (for inline definitions):

```python
from molexp.workflow import workflow, TaskContext

wf = workflow(name="pipeline")

@wf.task
async def square(ctx: TaskContext) -> float:
    return ctx.inputs ** 2

@wf.task(depends_on=["square"])
async def add_bias(ctx: TaskContext) -> float:
    return ctx.inputs + 10.0

spec = wf.build()
```

**OOP builder** (for reusable task classes):

```python
from molexp.workflow import Task, WorkflowBuilder, TaskContext

class SquareTask(Task):
    async def execute(self, ctx: TaskContext) -> float:
        return ctx.inputs ** 2

class AddBiasTask(Task):
    async def execute(self, ctx: TaskContext) -> float:
        return ctx.inputs + 10.0

spec = (
    WorkflowBuilder(name="pipeline")
    .add(SquareTask())
    .add(AddBiasTask(), depends_on=["square"])
    .build()
)
```

## Step 3: Execute

```python
import asyncio

async def main():
    result = await spec.execute()
    print(result)

asyncio.run(main())
```

Same-level tasks run in parallel automatically — no extra configuration needed.

## Step 4: Track with Workspace

In real scientific workflows you want reproducibility, artifact tracking, and parameter sweeps. MolExp's workspace layer handles all of that.

```python
from molexp.workspace import Workspace

# Create workspace (side-effect-free until materialize())
workspace = Workspace.from_path("./lab")

# Hierarchical API
project = workspace.create_project(name="My Research Project")
experiment = project.create_experiment(name="Experiment 1")
run = experiment.create_run(parameters={"lr": 0.01, "epochs": 10})

# Attach a run when executing so outputs are tracked automatically.
# Pass dry_run=True to expose ctx.dry_run inside tasks.
result = await spec.execute(run=run, dry_run=True)
```

Inside tasks, access workspace capabilities via `ctx`:

```python
class TrainTask(Task):
    async def execute(self, ctx: TaskContext) -> dict:
        if ctx.dry_run:
            return {"loss": None, "mode": "dry-run"}

        model = train(ctx.inputs, **ctx.state)

        # Save artifact to run directory
        ctx.save_artifact("model.pt", model)

        # Find a pre-registered asset up the scope hierarchy
        dataset = ctx.find_asset("training_data")

        return {"loss": 0.05}
```

## Complete Runnable Example

```python
import asyncio
from molexp.workflow import workflow, TaskContext
from molexp.workspace import Workspace

wf = workflow(name="demo")

@wf.task
async def fetch(ctx: TaskContext) -> list[float]:
    return [1.0, 4.0, 9.0]

@wf.task(depends_on=["fetch"])
async def compute(ctx: TaskContext) -> float:
    return sum(ctx.inputs)

spec = wf.build()

async def main():
    # Pure computation (no workspace)
    result = await spec.execute()
    print(f"Result: {result}")

    # With workspace tracking
    workspace = Workspace.from_path("./lab")
    project = workspace.create_project(name="Demo")
    experiment = project.create_experiment(name="Run 1")
    run = experiment.create_run(parameters={})

    result = await spec.execute(run=run, dry_run=True)
    print(f"Result (tracked): {result}")

asyncio.run(main())
```

## Next Steps

- [Task reference](../core/task.md) — full Task and Actor API
- [Workspace architecture](../workspace/) — projects, experiments, assets
- [Developer docs](../developer/) — internal compiler and IR details
