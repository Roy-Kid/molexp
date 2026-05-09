# Track a Run

The workflow layer is enough to describe computation, but it is not enough to preserve a research record. As soon as you care about reruns, artifacts, profile metadata, or reusable assets, you need the workspace layer.

## The Persistent Hierarchy

MolExp stores persistent state as `Workspace -> Project -> Experiment -> Run`. The workspace is the root directory. The project groups related work. The experiment is the repeatable definition of one workflow plus one parameter set. The run is one concrete execution attempt under that definition.

That split is the heart of the model. An experiment is not the same thing as a run. The experiment says what should be repeatable. The run records what actually happened this time.

## Binding a Workflow to an Experiment

```python
import molexp as me
from molexp.workflow import Workflow

ws = me.Workspace("./lab")
project = ws.Project("qm9")
exp = project.Experiment(
    "baseline",
    params={"lr": 1e-3},
    workflow_source="train.py",
)
spec.bind_to(exp)
```

`set_workflow` lives in `molexp.workflow.bindings`. It records the
`Workflow` in a process-local registry keyed by `experiment.id` so
the CLI, server routes, and agent tools can later retrieve it via
`Workflow.for_experiment(experiment)`. The previous workspace-side
`exp.set_workflow(...)` method was removed when the layer direction was
rectified — workspace no longer carries any workflow-shaped types.

`Workspace(...)` itself is lightweight. The workspace record is materialized explicitly with `ws.materialize()` or implicitly when the first child object is created. In ordinary usage, calling `ws.Project(...)` is enough to create the real on-disk hierarchy.

## Creating a Stable Run

When you create runs directly from Python, the safest habit is to give them explicit ids:

```python
run = exp.Run(parameters={"lr": 1e-3}, id="baseline-default")
result = await spec.execute(run=run)
```

That explicit id makes the run directory stable and easy to reason about. If you omit the id, MolExp creates a fresh run identity. The CLI behaves differently: it derives deterministic run ids from the resolved parameters, replica index, and active profile metadata so that repeated `molexp run` invocations can find the same run again.

## Recording Results, Artifacts, and Assets

Once a workflow runs under a `Run`, the task context gains workspace-backed helpers:

```python
@wf.task(depends_on=["fetch"])
async def train(ctx: TaskContext) -> dict:
    dataset = ctx.find_asset("training-data")
    loss = 0.05
    ctx.set_result("final_loss", loss)
    ctx.artifact.save("metrics.json", {"loss": loss})
    ctx.log("train").append(f"loss={loss}")
    return {"loss": loss, "dataset": dataset.path if dataset else None}
```

`ctx.set_result(...)` stores lightweight structured values on the run record. `ctx.artifact.save(...)` writes a file under the run's artifact directory and registers an `ArtifactAsset` in the catalog. `ctx.log(name)` returns a bound log handle that appends lines into `logs/<name>.log`. `ctx.find_asset(...)` walks the unified asset hierarchy from run to experiment to project to workspace, which is what allows task code to ask for named resources without hard-coding one particular disk path.

If a run produces a file that should become reusable managed state, promote it into the experiment's data-asset library:

```python
with run.start() as ctx:
    ctx.run.experiment.data_assets.import_asset(
        "feature-cache", "/tmp/features.parquet", action="move"
    )
```

That is how a one-off execution output becomes a `DataAsset` later runs can find and reuse by name. See [Unified Asset Model](../guide/assets.md) for the full set of kinds (artifact, log, checkpoint, error trace, execution state, data) and how they are indexed by the workspace catalog.

## Why This Layer Matters

Without the workspace layer, a workflow run is just one process execution. With the workspace layer attached, the same run becomes a durable record with parameters, profile data, execution history, artifacts, and reusable assets. That is the point where MolExp stops being only a graph runtime and starts becoming a research execution system.

The next practical step is usually to let the CLI discover and drive that same workspace, which is what [CLI and Profiles](cli-and-profiles.md) covers.

## Runnable Example

`examples/getting_started/03_tracked_run.py` executes one tracked run and prints the resulting on-disk layout plus the saved `run.json` fields.
