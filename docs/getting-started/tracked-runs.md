# Track a Run

The workflow layer is enough to describe computation, but it is not enough to preserve a research record. As soon as you care about reruns, artifacts, profile metadata, or reusable assets, you need the workspace layer.

## The Persistent Hierarchy

MolExp stores persistent state as `Workspace -> Project -> Experiment -> Run`. The workspace is the root directory. The project groups related work. The experiment is the repeatable definition of one workflow plus one parameter sweep. The run is one concrete execution attempt under that definition.

That split is the heart of the model. An experiment is not the same thing as a run. The experiment says what should be repeatable. The run records what actually happened this time.

## Declaring a Workflow on an Experiment

```python
import molexp as me

ws = me.Workspace("./lab", name="lab")
exp = ws.project("qm9").experiment("baseline").run(compiled, params={"lr": [1e-3]})
```

`Experiment.run(workflow, params=...)` is the declaration step. It binds the compiled workflow to the experiment in `molexp.workflow.default_binding_registry` (an explicit, injectable `{experiment_id → CompiledWorkflow}` store keyed by `experiment.id`, so the CLI, server routes, and agent tools can retrieve it via `default_binding_registry.for_experiment(experiment)`), records the workflow's graph IR on the experiment, registers the workspace for CLI discovery, and materializes one content-addressed `Run` per cell of the `params` sweep. `params=None` seeds a single parameter-free run.

`Workspace(...)` itself is lightweight. The workspace record is materialized explicitly with `ws.materialize()` or implicitly when the first child object is created. In ordinary usage, calling `ws.project(...)` is enough to create the real on-disk hierarchy.

## Run Identity Is Content-Addressed

Runs seeded through `exp.run(..., params=...)` derive their ids from their parameters, so re-declaring the same sweep is idempotent — repeated invocations rediscover the same runs instead of creating duplicates. When you create runs directly, `exp.add_run(params)` generates a fresh id unless you pass `id=...` yourself:

```python
run = exp.add_run({"lr": 1e-3}, id="baseline-default")
```

The CLI builds on the same mechanism: `molexp run` folds resolved parameters, replica index, and active profile metadata into deterministic run ids so that repeated invocations can find the same run again.

## Executing and Recording Results

A tracked execution happens inside the run's `RunContext`, with the compiled workflow driven by `WorkflowRuntime`:

```python
from molexp.workflow import WorkflowRuntime

run = exp.list_runs()[0]
with run.start() as ctx:
    result = await WorkflowRuntime().execute(compiled, run_context=ctx)
    ctx.set_result("final_loss", result.outputs["train"])
    ctx.artifact.save("metrics.json", result.outputs["train"])
    ctx.log("train").append("done")

print(run.status, run.get_result("final_loss"))
```

Task bodies stay on the pure `{inputs, config}` contract — they cannot reach the run or the workspace. The workspace helpers live on the driver-side `RunContext`: `ctx.set_result(...)` stores lightweight structured values on the run record (read them back later with the public `run.get_result(key)`), `ctx.artifact.save(...)` writes a file under the run's artifact directory and registers an `ArtifactAsset` in the catalog, `ctx.log(name)` returns a bound log handle that appends lines into the run's logs, and `ctx.find_asset(...)` walks the unified asset hierarchy from run to experiment to project to workspace.

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

`examples/getting_started/03_tracked_run.py` executes one tracked run and prints the resulting on-disk layout plus the persisted run fields.
