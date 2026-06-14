# Quick Start

The fastest way to understand MolExp is to see one small script move through the whole lifecycle. The example below defines a workflow, declares it on a workspace experiment, executes one tracked run, and reads the persisted result back.

## One Small Script

```python
import asyncio

import molexp as me
from molexp.workflow import TaskContext, WorkflowCompiler, WorkflowRuntime

wf = WorkflowCompiler(name="sum")


@wf.task
async def fetch(ctx: TaskContext) -> list[float]:
    return [1.0, 4.0, 9.0]


@wf.task(depends_on=["fetch"])
async def summarize(ctx: TaskContext) -> float:
    scale = ctx.config.get("scale", 1.0)
    return sum(ctx.inputs) * scale


compiled = wf.compile()

ws = me.Workspace("./lab", name="lab")
exp = ws.project("demo").experiment("sum").run(compiled, params={"scale": [1.0]})


async def main() -> None:
    run = exp.list_runs()[0]
    with run.start() as ctx:
        result = await WorkflowRuntime().execute(compiled, run_context=ctx)
        ctx.set_result("total", result.outputs["summarize"])
        ctx.artifact.save("metrics.json", {"total": result.outputs["summarize"]})
    print(run.status, run.get_result("total"))


if __name__ == "__main__":
    asyncio.run(main())
```

## What This Script Does

The workflow itself is only the graph created by `WorkflowCompiler(...)`, the two task definitions, and `compiled = wf.compile()`. Everything after that is about persistence. The workspace creates a durable root on disk. The project groups related work. The experiment binds the compiled workflow to one named research definition via `exp.run(compiled, params=...)` — `params` is the sweep, and MolExp materializes one content-addressed `Run` per parameter cell. Each run gives one concrete execution attempt a stable directory and metadata record.

Because run ids are derived from the run's parameters, re-declaring the same sweep is idempotent: repeated invocations rediscover the same runs instead of creating duplicates.

Inside a task body, `ctx.inputs` carries the data flowing in along the graph's edges (the upstream output, or — for a root task of a tracked run — the engine-injected `{"params": ..., "workdir": ...}` mapping) and `ctx.config` exposes profile data when a profile is active. Workspace helpers live on the driver-side `RunContext` opened by `run.start()`: `ctx.set_result(...)` stores lightweight result values on the run record (read them back with the public `run.get_result(key)`), and `ctx.artifact.save(...)` writes a file under the run's artifact directory and registers it as an `ArtifactAsset` in the workspace catalog. See [Unified Asset Model](../guide/assets.md) for the full shape of artifacts, logs, checkpoints, and data assets.

## Running the Script

If you execute the file directly with Python, the `main()` function above will execute one tracked run and print its status and result. The `exp.run(...)` declaration also registers the workspace for CLI discovery, so the same script can be driven by `molexp run` instead:

```bash
molexp run train.py
```

At that point the CLI owns run selection, deterministic run ids, and the `RunContext` lifecycle. That is usually where profiles, resume behavior, and scheduler-backed execution start to matter.

## After the First Run

The next useful page depends on what felt most mysterious. If the workflow definition itself was the unfamiliar part, continue with [Your First Workflow](first-workflow.md). If the new part was the workspace hierarchy and tracked run state, continue with [Track a Run](tracked-runs.md). If the script already makes sense and you want to move to `molexp run`, continue with [CLI and Profiles](cli-and-profiles.md).

## Runnable Example

`examples/getting_started/01_quick_start.py` is the same idea as a stand-alone script you can run with `python`.
