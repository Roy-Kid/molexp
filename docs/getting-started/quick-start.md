# Quick Start

The fastest way to understand MolExp is to see one small script move through the whole lifecycle. The example below defines a workflow, binds it to a workspace experiment, executes one tracked run, and exposes the workspace to the CLI.

## One Small Script

```python
import asyncio
import molexp as me
from molexp.workflow import TaskContext, workflow

wf = workflow(name="demo")


@wf.task
async def fetch(ctx: TaskContext) -> list[float]:
    return [1.0, 4.0, 9.0]


@wf.task(depends_on=["fetch"])
async def summarize(ctx: TaskContext) -> float:
    scale = ctx.config.get("scale", 1.0)
    total = sum(ctx.inputs) * scale
    ctx.set_result("total", total)
    ctx.artifact.save("metrics.json", {"total": total})
    return total


spec = wf.build()

ws = me.Workspace("./lab")
project = ws.project("demo")
exp = project.experiment(
    "sum",
    params={"scale": 1.0},
    workflow_source="train.py",
)
exp.set_workflow(spec)

me.entry(ws)


async def main() -> None:
    run = exp.run(parameters={"scale": 1.0}, id="sum-default")
    result = await spec.execute(run=run)
    print(result.outputs)


if __name__ == "__main__":
    asyncio.run(main())
```

## What This Script Does

The workflow itself is only the graph created by `workflow(...)`, the two task definitions, and `spec = wf.build()`. Everything after that is about persistence. The workspace creates a durable root on disk. The project groups related work. The experiment binds the compiled workflow to one named research definition. The run gives one concrete execution attempt a stable directory and metadata record.

The explicit run id matters in direct Python usage. If you call `exp.run()` without an id, you get a fresh run. When you do want stable identity from Python, provide the id yourself. The CLI does that automatically when it derives deterministic ids from parameters and profile metadata.

Inside the task body, `ctx.config` exposes profile data when a profile is active. `ctx.set_result(...)` stores lightweight result values on the run record. `ctx.artifact.save(...)` writes a file under the run's artifact directory and registers it as an `ArtifactAsset` in the workspace catalog. The task code does not need to know where that directory lives, only that a run-backed context is attached. See [Unified Asset Model](../guide/assets.md) for the full shape of artifacts, logs, checkpoints, and data assets.

## Running the Script

If you execute the file directly with Python, the `main()` function above will create one tracked run and print the workflow outputs. If you want MolExp to discover the same workspace through the CLI, keep the `me.entry(ws)` call and run:

```bash
molexp run train.py
```

At that point the same script can be driven by the CLI rather than by the manual `asyncio.run(...)` path. That is usually where profiles, resume behavior, and scheduler-backed execution start to matter.

## After the First Run

The next useful page depends on what felt most mysterious. If the workflow definition itself was the unfamiliar part, continue with [Your First Workflow](first-workflow.md). If the new part was the workspace hierarchy and tracked run state, continue with [Track a Run](tracked-runs.md). If the script already makes sense and you want to move to `molexp run`, continue with [CLI and Profiles](cli-and-profiles.md).

## Runnable Example

`examples/getting_started/01_quick_start.py` is the same idea as a stand-alone script you can run with `python`.
