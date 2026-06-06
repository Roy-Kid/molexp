# Workspace API

The `molexp.workspace` package is the persistence layer of Molexp. It does not describe computation itself; that job belongs to `molexp.workflow`. Instead, the workspace API answers a different set of questions: where a project lives on disk, how an experiment is recorded, how a concrete run is created, where artifacts are written, and which configuration was active when that run executed.

This page explains that API as a model of state, not as a flat list of methods. The main idea is simple. A workflow spec is reusable and in-memory. A workspace hierarchy is durable and on disk. When the two are bound together, Molexp can execute a workflow while preserving enough metadata to reproduce, inspect, and rerun it later.

## Public Entry Points

Most user code imports the workspace surface from the top-level package:

```python
import molexp as me

me.Workspace
me.Project
me.Experiment
me.Run
me.RunContext
me.ParamSpace
me.GridSpace
me.UniformSpace
me.entry
```

The same names are also available from `molexp.workspace`, together with lower-level metadata models such as `RunMetadata` and `ExecutionRecord`. For ordinary workflow authoring, the top-level imports are the intended entry point.

## The Hierarchy as a State Model

The workspace API organizes persistent state into four levels:

```text
Workspace
└── Project
    └── Experiment
        └── Run
```

Each layer has a distinct responsibility.

A `Workspace` is the root of a lab or repository-local working area. It owns the filesystem root, the top-level metadata file, and the collection of projects beneath it.

A `Project` groups related work. In practice this is where one usually draws the line around a model family, dataset campaign, paper track, or pipeline family.

An `Experiment` records a repeatable definition: the chosen workflow source, the concrete parameter dictionary attached to that definition, and the replica policy used to derive repeated runs with different seeds.

A `Run` is one realized execution attempt under an experiment. It stores status, parameters, profile metadata, execution history, structured failure information, and all per-run files.

That split matters because it preserves the difference between definition and outcome. Many runs can belong to the same experiment without collapsing them into a single mutable record.

## Creating and Materializing a Workspace

The constructor for `Workspace` is intentionally lightweight:

```python
ws = me.Workspace("./lab", name="Lab")
```

At this point you have a Python object, not yet a fully materialized directory tree. The constructor resolves the root path and prepares metadata, but it does not immediately write `workspace.json`. Disk state appears when you call `ws.materialize()` explicitly or when you create the first child project through `ws.Project(...)`.

That delayed materialization is important if you want to compose workspace objects without leaving empty directories behind. When you do want a strict load-from-disk path, use:

```python
ws = me.Workspace.load("./lab")
```

`Workspace.load()` raises if `workspace.json` is missing; use it when you want to fail loud rather than silently create a new workspace.

Once a workspace exists, its core properties are straightforward. `ws.id` and `ws.name` come from `workspace.json`. `ws.root` is the resolved filesystem root. `ws.assets` gives access to the workspace-level asset library. `ws.save()` persists metadata changes when you deliberately update the workspace record itself.

## Projects Are Stable Children of a Workspace

Projects are created through the workspace:

```python
project = ws.Project("QM9")
```

This operation is idempotent by project slug. If the corresponding directory already exists, Molexp loads it. If it does not exist, Molexp creates a new `Project`, writes `project.json`, and returns it. Repeated calls with the same project name inside one process return the same in-memory object, which lets later code keep bound state on child objects without accidentally duplicating handles.

Projects expose descriptive metadata such as `project.description`, `project.owner`, `project.tags`, and `project.config`. They also expose `project.project_dir`, which is the concrete directory under `<workspace>/projects/<project_id>/`, `project.assets`, which is a catalog view scoped to the project, and `project.data_assets`, which is the `DataAssetLibrary` for imported external data.

Project discovery works in two styles. `ws.project(name_or_id)` retrieves a single project by id or slugified name. `ws.list_projects()` scans the on-disk tree and merges that result with the in-memory cache. Deletion is explicit through `ws.delete_project(project_id)`, which removes the project directory from disk.

## Experiments Bind Workflow Identity to Parameters

Experiments are created through a project:

```python
exp = project.Experiment(
    "baseline",
    params={"lr": 1e-3},
    n_replicas=3,
    seeds=[42, 123, 456],
    workflow_source="train.py",
    workflow_type="python",
    git_commit="abc123",
)
```

This is the point where a reusable workflow begins to acquire persistent identity. The experiment metadata records the workflow source reference, the parameter dictionary, the replica configuration, and optional source-control provenance. In the API, `exp.parameter_space` and `exp.params` are the same concrete dictionary; despite the name, this is not an abstract search space object but the bound parameter payload for this one experiment.

Like projects, experiments are get-or-create objects. `project.Experiment(name, ...)` is idempotent by id. If you omit `id`, Molexp uses the slugified experiment name. If an experiment directory with that id already exists, it is loaded; otherwise it is created and materialized. `project.experiment(experiment_id)` and `project.list_experiments()` provide the expected lookup paths.

Experiments also own `exp.assets`, the experiment-scoped asset library, and `exp.experiment_dir`, the root directory for everything stored beneath that experiment.

## Binding a Workflow Is a Separate Step

Creating an experiment does not automatically attach executable workflow code. The binding lives at the workflow layer, not on the experiment object itself — workspace was reduced to pure storage, so it no longer carries any workflow-shaped types. Use `set_workflow` from `molexp.workflow.bindings`:

```python
from molexp.workflow import Workflow, WorkflowBuilder

spec.bind_to(exp)
```

The expected input is a compiled `Workflow`. For a bare callable that accepts `RunContext`, wrap it through `promote_callable`, which is the workflow-layer helper that replaced the old workspace-side `_promote_to_workflow` shortcut:

```python
from molexp.workflow import promote_callable, Workflow, WorkflowBuilder

def train(run_ctx: me.RunContext) -> None:
    ...

name="train").bind_to(exp, promote_callable(train)
```

`set_workflow` records the spec in a process-local registry keyed by `experiment.id`. Re-binding is allowed (later calls replace the earlier spec); use `Workflow.for_experiment(experiment)` to read it back from anywhere in the same process — typically from CLI / server / agent code that needs to drive execution after the user script has bound the spec.

Replica handling also lives on the experiment. `exp.n_replicas` and `exp.seeds` describe the declared policy, while `exp.get_seeds()` returns the effective seed list. If no explicit seeds were stored, Molexp expands a deterministic default seed sequence and truncates it to the requested replica count.

## Runs Record Concrete Executions

Runs are created through the experiment:

```python
run = exp.Run(parameters={"lr": 1e-3, "seed": 42})
```

This is the point where the distinction between the Python API and the CLI matters.

From Python, `exp.run()` usually creates a fresh run, because a new run id is generated when you do not provide one explicitly. The method only behaves as a get-or-load operation when the target id already exists. In other words, projects and experiments are naturally idempotent by name, but runs are only idempotent by explicit run id.

If you need a stable identity, pass `id=...` yourself:

```python
run = exp.run(
    parameters={"lr": 1e-3, "seed": 42},
    id="baseline-seed-42",
)
```

The CLI uses this mechanism to achieve repeatable runs. `molexp run` derives deterministic run ids from resolved parameters, replica index, and active profile metadata, so re-running the same script can discover and resume or skip existing runs.

Each `Run` stores its data in `<experiment>/runs/run-<run_id>/`. The most important fields live on `run.metadata`, a `RunMetadata` model. Direct convenience properties such as `run.id`, `run.parameters`, `run.status`, and `run.run_dir` simply expose the corresponding metadata fields in a more ergonomic form.

Two details are easy to miss. First, if the experiment metadata includes `workflow_source`, then `exp.run()` can carry a workflow-snapshot dict on the new run's `RunMetadata.workflow_snapshot`. The canonical type for that payload is `molexp.workflow.snapshot_ref.WorkflowSnapshotRef`, but workspace stores it as opaque JSON to keep the layer one-directional — it is the workflow layer that gives the dict its shape. Second, one logical run may be executed more than once. Molexp does not flatten those attempts; instead it appends `ExecutionRecord` entries to `run.metadata.execution_history`.

If a run should be marked dead without completing normally, `run.cancel()` transitions it to `cancelled` and writes the terminal timestamp.

## RunContext Governs the Execution Lifecycle

`RunContext` is the execution-time object that turns a persistent run record into a live working directory:

```python
with run.start(profile_config=cfg) as ctx:
    ...
```

Entering the context creates `artifacts/` and `logs/` if necessary, loads previously saved result values, stores the active profile metadata onto the run, stamps the run with temporary ownership labels, switches status to `running`, and appends a fresh `ExecutionRecord` for the current attempt.

Leaving the context closes the lifecycle. A normal exit marks the run as `succeeded`, unless the workflow context itself recorded a failed run status. An exception marks the run as `failed`, writes `ErrorInfo` into metadata, and stores a traceback under `execution/<execution_id>/error.txt`. In both cases the execution history entry is finalized with its end time and final status.

The most commonly used `RunContext` helpers fall into three groups.

The first group deals with results and metadata. `ctx.set_result(key, value)` and `ctx.get_result(key)` read and write the lightweight result map persisted into `run.json`. `ctx.run.set_workflow(payload)` stores a workflow-shaped dictionary onto the serialized run-context — the payload type is opaque to workspace; in practice it is a serialized `WorkflowSnapshotRef` from the workflow layer, but workspace just round-trips the JSON. This is distinct from the workflow-layer `Workflow.bind_to(experiment)` registry call, which records the live `Workflow` instance for downstream consumers within the same process.

The second group deals with files, through typed accessors. `ctx.artifact.save(name, data)` writes under `<run_dir>/artifacts/`, automatically choosing JSON, binary, file-copy, or string serialization, and registers an `ArtifactAsset` in the catalog with a populated `Producer`. `ctx.log(name)` returns a bound log handle whose `.append(line)` writes a line into `<run_dir>/logs/<name>.log` and keeps the `LogAsset` up to date. `ctx.checkpoint(name, data=...)` writes a `CheckpointAsset` under `<run_dir>/.ckpt/`, linearly chained to the previous checkpoint of the same run.

The third group deals with asset lookup across the unified catalog. `ctx.find_asset(name)` searches in run → experiment → project → workspace order and returns a typed `Asset` subclass. `ctx.get_data_dir(asset_name, fallback=...)` first tries that hierarchical lookup; if nothing is found and a fallback path is provided, it creates that directory under the workspace root and returns it. For typed queries against the catalog use `ws.catalog.query_assets(kind=..., producer_run=..., scope=...)`.

Finally, `RunContext.open(run_dir)` reconstructs a full workspace, project, experiment, and run chain from an existing run directory. This is what worker-style entry points use when they only know the run path on disk.

## Assets Are Unified, Typed, and Scoped

Every persistent byproduct of an experiment — imported datasets, task artifacts, logs, checkpoints, captured exceptions, workflow execution state — is a typed `Asset` subclass indexed by the workspace catalog. Each workspace, project, experiment, and run scope exposes an `assets` property that returns a read-only catalog view and a `data_assets` property for importing external data:

```python
ws.assets.list()                # every asset in the workspace
project.assets.list()           # assets attributable to this project
exp.assets.list()               # assets attributable to this experiment
ws.data_assets.import_asset("qm9", "/data/qm9.tar.bz2")
project.data_assets.import_asset("features", "/tmp/features", action="move")
```

`import_asset(name, src, action="copy", meta=None)` writes the payload under `<scope>/data_assets/<asset_id>/payload/` and registers a `DataAsset`. Supported actions are `copy`, `move`, `symlink`, and `hardlink`. The asset then appears in the same catalog as artifacts, logs, and checkpoints produced by runs.

Lookup during execution walks outward automatically. `ctx.find_asset(name)` searches run → experiment → project → workspace until it matches, so common datasets can live once at workspace scope while derived artifacts stay attached to the run that produced them. For the complete mental model — scopes, producer attribution, the catalog, and the per-kind subclasses — see the [Unified Asset Model](assets.md) guide.

## Parameter Spaces Help Build Experiments, Not Runs

`ParamSpace`, `GridSpace`, and `UniformSpace` belong to the workspace surface because parameter exploration is usually what leads to experiment creation:

```python
from molexp import GridSpace, UniformSpace

grid = GridSpace({"lr": [1e-3, 1e-4], "batch": [32, 64]})
random_search = UniformSpace({"lr": [1e-3, 5e-4, 1e-4]}, n_samples=10, seed=42)
```

These objects are iterable generators of parameter dictionaries. They do not cause Molexp to create experiments automatically. The intended pattern is that user code iterates the space and materializes one experiment per parameter combination:

```python
from molexp.workflow import Workflow, WorkflowBuilder

for params in grid:
    exp = project.Experiment(
        f"lr-{params['lr']}-batch-{params['batch']}",
        params=params,
        workflow_source="train.py",
    )
    spec.bind_to(exp)
```

That design keeps the library explicit. Molexp gives you the combinatorics, but the user still decides naming, grouping, and lifecycle boundaries.

## Script Registration for CLI Discovery

When workflows are launched from the CLI, the missing link is `me.entry(ws)`:

```python
import molexp as me
from molexp.workflow import Workflow, WorkflowBuilder

ws = me.Workspace("./lab")
project = ws.Project("demo")
exp = project.Experiment("baseline", params={"lr": 1e-3}, workflow_source="train.py")
spec.bind_to(exp)

me.entry(ws)
```

This call registers the workspace when the script is imported by `molexp run`. The CLI then loads the script, reads the registered workspaces, discovers projects and experiments beneath them, and resolves which workflow belongs to which persistent experiment via `Workflow.for_experiment(experiment)` from the same registry the script wrote to. Without that registration step, the CLI has no supported way to discover the workspace graph from an arbitrary Python module.

## A Minimal End-to-End Example

The following example shows the full path from workflow definition to persistent run:

```python
import molexp as me
from molexp.workflow import TaskContext, Workflow, Workflow, WorkflowBuilder

wf = WorkflowBuilder(name="demo")


@wf.task
async def compute(ctx: TaskContext) -> float:
    scale = ctx.config.get("scale", 1.0)
    value = ctx.run_context.params["lr"] * scale
    ctx.run_context.set_result("scaled_lr", value)
    ctx.run_context.artifact.save("metrics.json", {"scaled_lr": value})
    return value


spec = wf.build()

ws = me.Workspace("./lab")
project = ws.Project("demo")
exp = project.Experiment(
    "baseline",
    params={"lr": 1e-3},
    workflow_source="train.py",
)
spec.bind_to(exp)

run = exp.Run(parameters={"lr": 1e-3}, id="baseline-defaults")
result = await spec.execute(run=run)

print(run.status)
print(run.metadata.execution_history[-1].status)
print(result.outputs)
```

The important part is not the arithmetic. It is the shape of the interaction. A workflow spec remains reusable. The experiment gives that workflow a stable scientific identity. The run gives one execution attempt a stable filesystem location. `RunContext` bridges the two during execution so that results, artifacts, profile data, and failures are recorded as part of the same durable object graph.

## Runnable Example

`examples/workspace/workspace_api.py` walks the four-level hierarchy, demonstrates factory idempotence, and reopens the workspace with `Workspace.load()`.
