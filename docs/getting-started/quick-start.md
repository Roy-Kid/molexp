# Quick Start

The fastest way to understand MolExp is to see one small script move through the whole lifecycle: define a workflow, execute one tracked run, and read the result back.

## One Small Script

```python
import molexp as me
from molexp.workflow import WorkflowCompiler

wf = WorkflowCompiler(name="sum")


@wf.task
def fetch(scale: float = 1.0) -> dict:
    return {"values": [1.0, 4.0, 9.0], "scale": scale}


@wf.task(depends_on=["fetch"])
def summarize(values: list[float], scale: float = 1.0) -> float:
    return sum(values) * scale


ws = me.Workspace("./lab", name="lab")
run = ws.project("demo").experiment("sum").add_run(params={"scale": 2.0})

result = run.execute(wf)
print(run.status, result.outputs["summarize"])
```

Running the file prints `succeeded 28.0`. That is the whole loop: two plain functions become a dependency graph, one `Run` gives the execution a durable home on disk, and `run.execute(wf)` drives it end to end — compiling the workflow, opening the run's tracked lifecycle, executing the graph, and persisting every task's output — before handing back the result. `result.outputs` maps each task's name to what it returned.

## What This Script Does

Tasks are ordinary functions — synchronous `def` is fine (an `async def` task works exactly the same way when you need it). Data flows along the graph's edges and binds to each task's named parameters: the root task `fetch` receives the run's `params` by name, which is why declaring a `scale` parameter hands it `2.0`; because `fetch` returns a dictionary, `summarize`'s `values` and `scale` parameters are filled from that dictionary's keys. A parameter with no matching input falls back to its declared default.

The workspace half is the persistent hierarchy `Workspace -> Project -> Experiment -> Run`: the workspace is a durable root directory, the project groups related work, the experiment names one repeatable definition, and each run records one execution with its parameters, status, and outputs. Everything `run.execute` produced is still there after the process exits — `run.get_result("summarize")` reads the persisted value back in a later session.

`run.execute` is deliberately strict about state: a run that already succeeded refuses to silently re-execute (pass `fresh=True` for a new attempt), and a failed run raises instead of returning quietly — calling `run.execute(wf)` again on it resumes from where it stopped. [Track a Run](tracked-runs.md) covers those semantics.

## Running the Script

Execute the file directly with `python`. The same workflow can also be declared on the experiment for CLI-driven execution (`molexp run train.py`), which adds profiles, schedulers, and resume flags on top of the identical execution path — see [CLI and Profiles](cli-and-profiles.md).

## After the First Run

The next useful page depends on what felt most mysterious. If the workflow definition itself was the unfamiliar part, continue with [Your First Workflow](first-workflow.md). If the new part was the workspace hierarchy and tracked run state — or you want to fan one workflow out over a parameter sweep — continue with [Track a Run](tracked-runs.md). If the script already makes sense and you want to move to `molexp run`, continue with [CLI and Profiles](cli-and-profiles.md).

## Runnable Example

`examples/getting_started/01_quick_start.py` is the same idea as a stand-alone script you can run with `python`.
