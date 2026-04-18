# Quick Start

Build a complete MolExp workflow in five minutes — from defining tasks to executing them under a tracked workspace run.

## Step 1: Define Tasks

A **Task** is the basic batch computation unit. Subclass `Task` and implement `execute()`. The `TaskContext` gives access to upstream outputs (`ctx.inputs`), shared state (`ctx.state`), dependencies (`ctx.deps`), the active profile config (`ctx.config`), and — when running under a workspace `Run` — artifact / asset helpers.

```python
from molexp.workflow import Task, TaskContext

class Square(Task):
    async def execute(self, ctx: TaskContext) -> float:
        value: float = ctx.inputs
        return value * value
```

For streaming workloads, subclass `Actor` with `run()` (an async generator).

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
from molexp.workflow import Task, TaskContext, WorkflowBuilder

class Square(Task):
    async def execute(self, ctx: TaskContext) -> float:
        return ctx.inputs ** 2

class AddBias(Task):
    async def execute(self, ctx: TaskContext) -> float:
        return ctx.inputs + 10.0

spec = (
    WorkflowBuilder(name="pipeline")
    .add(Square())
    .add(AddBias(), depends_on=["square"])
    .build()
)
```

Zero-import third-party integration also works: any object whose method signature matches `async def execute(self, ctx)` satisfies the `Runnable` protocol and can be passed straight to `.add(...)`.

## Step 3: Execute (pure computation)

```python
import asyncio

async def main():
    result = await spec.execute()
    print(result)

asyncio.run(main())
```

Tasks at the same dependency level run in parallel automatically — no extra configuration needed.

## Step 4: Track with a Workspace

In real scientific workflows you want reproducibility, artifact tracking, and parameter sweeps. MolExp's workspace layer handles all of that.

```python
import molexp as me

ws = me.Workspace("./lab")
project = ws.project("QM9")

# Bind the workflow to an experiment that carries parameters + replica count
exp = project.experiment(
    "baseline",
    params={"lr": 0.01, "epochs": 10},
    n_replicas=3,
    workflow_source="train.py",
)
exp.set_workflow(spec)

# Execute under a tracked Run so artifacts / results persist
run = exp.run(parameters={"lr": 0.01, "epochs": 10})
result = await spec.execute(run=run)
```

All factories (`ws.project(...)`, `project.experiment(...)`, `exp.run(...)`) are **idempotent**: call them again with the same name / ID and you get the same instance, loaded from disk if it already exists.

Inside tasks, access workspace capabilities via `ctx`:

```python
class Train(Task):
    async def execute(self, ctx: TaskContext) -> dict:
        # profile config is always available
        lr = ctx.config.get("lr", 1e-3)

        # workspace helpers return None when no run is attached
        dataset = ctx.find_asset("training_data")
        ctx.save_artifact("hparams.json", {"lr": lr})
        ctx.set_result("final_loss", 0.05)
        return {"loss": 0.05}
```

## Step 5: Run from the CLI

For longer-lived projects, put the workspace setup in a Python script and register it with `me.entry(ws)` so the CLI can discover it:

```python
# train.py
import molexp as me
from molexp.workflow import workflow, TaskContext

wf = workflow(name="demo")

@wf.task
async def fetch(ctx: TaskContext) -> list[float]:
    return [1.0, 4.0, 9.0]

@wf.task(depends_on=["fetch"])
async def compute(ctx: TaskContext) -> float:
    return sum(ctx.inputs)

spec = wf.build()

ws = me.Workspace("./lab")
project = ws.project("demo")
exp = project.experiment("sum", workflow_source="train.py")
exp.set_workflow(spec)

me.entry(ws)
```

```bash
# Local sequential execution
molexp run train.py

# Named molcfg profile with one-off override
molexp run train.py --profile smoke --override compute.scale=2

# HPC submission via molq
molexp run train.py --slurm --partition gpu --cpus 8 --time 02:00:00
```

## Step 6: Put Execution Variants in `molcfg.yaml`

Once the script itself is stable, the next source of change is usually execution shape: shorter runs for iteration, profile-specific parameter sets, or toggles that let tasks skip heavy work. In MolExp, that variability belongs in `molcfg`, not in ad-hoc command flags or duplicated scripts.

```yaml
version: 1

defaults:
  epochs: 100
  optimizer:
    lr: 0.001

profiles:
  smoke:
    epochs: 3

  dry-run:
    epochs: 1
    skip_heavy_compute: true
```

Task code reads the active profile through `ctx.config`:

```python
@wf.task(depends_on=["fetch"])
async def compute(ctx: TaskContext) -> float:
    if ctx.config.get("skip_heavy_compute"):
        return 0.0
    return sum(ctx.inputs) * ctx.config.get("optimizer", {}).get("lr", 1.0)
```

The CLI then selects or refines a profile at run time:

```bash
# Use defaults only
molexp run train.py

# Resolve one named profile
molexp run train.py --profile smoke

# Apply one-off changes after profile resolution
molexp run train.py --profile smoke --override optimizer.lr=0.0005

# Retry failed or cancelled runs for that profile
molexp run train.py --profile smoke --resume
```

This is also where reproducibility becomes tangible. The run stores the normalized profile name, the merged config payload, and a `config_hash`, so different profiles of the same experiment remain distinct on disk and in the UI.

## Complete Runnable Example

```python
import asyncio
import molexp as me
from molexp.workflow import workflow, TaskContext

wf = workflow(name="demo")

@wf.task
async def fetch(ctx: TaskContext) -> list[float]:
    return [1.0, 4.0, 9.0]

@wf.task(depends_on=["fetch"])
async def compute(ctx: TaskContext) -> float:
    return sum(ctx.inputs)

spec = wf.build()


async def main():
    # Pure computation
    result = await spec.execute()
    print(f"Result: {result.outputs}")

    # With workspace tracking
    ws = me.Workspace("./lab")
    project = ws.project("demo")
    exp = project.experiment("sum", workflow_source="inline")
    exp.set_workflow(spec)

    run = exp.run(parameters={})
    result = await spec.execute(run=run)
    print(f"Result (tracked): {result.outputs}, run={run.id}")


asyncio.run(main())
```

## Next Steps

- [Task reference](../guide/task-and-actor.md) — full Task / Actor / TaskContext API
- [Run Profiles and Reproducible CLI Execution](../guide/run-profiles.md) — `molcfg`, `ctx.config`, overrides, resume behavior
- [Workspace architecture](../guide/workspace-architecture.md) — projects, experiments, runs, assets
- [Developer docs](../development/ir-and-compiler.md) — internal compiler and graph IR
