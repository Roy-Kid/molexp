# Workspace API

The `molexp.workspace` package is the persistence layer of Molexp. It does not describe computation itself; that job belongs to `molexp.workflow`. Instead, the workspace API answers a different set of questions: where a project lives on disk, how an experiment is recorded, how a concrete run is created, where artifacts are written, and which configuration was active when that run executed.

This page explains that API as a model of state, not as a flat list of methods. The main idea is simple. A compiled workflow is reusable and in-memory. A workspace hierarchy is durable and on disk. When the two are bound together, Molexp can execute a workflow while preserving enough metadata to reproduce, inspect, and rerun it later.

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

An `Experiment` records a repeatable definition: the declared workflow (graph IR plus source snapshot), the parameter sweep attached to that definition, and the replica policy used to derive repeated runs with different seeds.

A `Run` is one realized execution attempt under an experiment. It stores status, parameters, profile metadata, execution history, structured failure information, and all per-run files.

That split matters because it preserves the difference between definition and outcome. Many runs can belong to the same experiment without collapsing them into a single mutable record.

Every level is a `Folder`: generic CRUD (`add_folder` / `get_folder` / `has_folder` / `list_folders` / `remove_folder`) plus typed semantic sugar per subclass (`add_project` / `add_experiment` / `add_run` / ...) and fluent create-or-get aliases (`ws.project(...)` / `project.experiment(...)`).

## Creating and Materializing a Workspace

The constructor for `Workspace` is intentionally lightweight:

```python
ws = me.Workspace("./lab", name="Lab")
```

At this point you have a Python object, not yet a fully materialized directory tree. The constructor resolves the root path and prepares metadata, but it does not immediately write `workspace.json`. Disk state appears when you call `ws.materialize()` explicitly or when you create the first child project through `ws.project(...)`.

That delayed materialization is important if you want to compose workspace objects without leaving empty directories behind. When you do want a strict load-from-disk path, use:

```python
ws = me.Workspace.load("./lab")
```

`Workspace.load()` raises if `workspace.json` is missing; use it when you want to fail loud rather than silently create a new workspace.

Once a workspace exists, its core properties are straightforward. `ws.id` and `ws.name` come from `workspace.json`. `ws.root` is the resolved filesystem root. `ws.assets` gives access to the workspace-level asset catalog view. `ws.save()` persists metadata changes when you deliberately update the workspace record itself.

## Projects Are Stable Children of a Workspace

Projects are created through the workspace:

```python
project = ws.project("QM9")          # fluent create-or-get
project = ws.add_project("QM9")      # same operation, explicit spelling
```

This operation is idempotent by project slug. If the corresponding directory already exists, Molexp loads it. If it does not exist, Molexp creates a new `Project`, writes `project.json`, and returns it. Repeated calls with the same project name inside one process return the same in-memory object, which lets later code keep bound state on child objects without accidentally duplicating handles.

Projects expose descriptive metadata such as `project.description`, `project.owner`, `project.tags`, and `project.config`. They also expose `project.project_dir`, which is the concrete directory under `<workspace>/projects/<project_id>/`, `project.assets`, which is a catalog view scoped to the project, and `project.data_assets`, which is the `DataAssetLibrary` for imported external data.

Project lookup works in three styles. `ws.get_project(name)` is the strict getter (raises `ProjectNotFoundError` if absent), `ws.has_project(name)` tests existence, and `ws.list_projects()` scans the on-disk tree. Deletion is explicit through `ws.remove_project(name)`, which removes the project directory from disk and cascades the catalog rows.

## Experiments Bind Workflow Identity to Parameters

Experiments are created through a project:

```python
exp = project.experiment(            # fluent create-or-get (add_experiment also works)
    "baseline",
    params={"lr": 1e-3},
    n_replicas=3,
    seeds=[42, 123, 456],
)
```

This is the point where a reusable workflow begins to acquire persistent identity. The experiment metadata records the workflow declaration, the parameter dictionary, the replica configuration, and optional source-control provenance. In the API, `exp.parameter_space` and `exp.params` are the same concrete dictionary; despite the name, this is not an abstract search space object but the bound parameter payload for this one experiment.

Like projects, experiments are get-or-create objects, idempotent on the slugified name. `project.get_experiment(name)` and `project.list_experiments()` provide the expected lookup paths.

Experiments also own `exp.assets`, the experiment-scoped asset view, and `exp.experiment_dir`, the root directory for everything stored beneath that experiment.

## Declaring the Workflow Is One Fluent Call

Creating an experiment does not automatically attach executable workflow code. The sanctioned way to pair the two is `Experiment.run(workflow, params=...)`:

```python
from molexp.workflow import WorkflowCompiler

compiled = WorkflowCompiler(name="train").add(TrainTask()).compile()

exp = ws.project("demo").experiment("baseline").run(compiled, params={"lr": [1e-3, 1e-4]})
```

`params` is the per-run sweep — a plain `{axis: [values]}` grid (expanded as a Cartesian product) or any `ParamSpace`. The call materializes one content-addressed `Run` per cell (idempotent: re-declaring the same sweep adds no duplicates), binds the compiled workflow to the experiment in `molexp.workflow.default_binding_registry` (an explicit, injectable `{experiment_id → CompiledWorkflow}` store that replaced the old class-level registry), records the workflow's graph IR on the experiment for the server/UI, and registers the workspace for CLI discovery. Read the binding back with `default_binding_registry.for_experiment(experiment)` from anywhere in the same process. Re-declaring replaces the earlier binding; the registry is process-local, so cluster workers re-establish it by re-importing the user script.

The expected input is a `CompiledWorkflow` (the product of `WorkflowCompiler(...).compile()`). For a bare callable on the pure `fn(inputs, config)` contract, `molexp.workflow.promote_callable(fn, name=...)` wraps it into a single-task `CompiledWorkflow` that works in `Experiment.run` too: a module-level (importable) function is serialized into the experiment's graph IR as a `module:qualname` entrypoint ref and resolved back via importlib at execution time. Non-importable callables (lambdas, closures, functions defined in `__main__` / a REPL) cannot be serialized for tracked execution — `Experiment.run` raises a clear error for those; execute them in-memory through `WorkflowRuntime().execute(...)` instead.

Replica handling also lives on the experiment. `exp.n_replicas` and `exp.seeds` describe the declared policy, while `exp.get_seeds()` returns the effective seed list. If no explicit seeds were stored, Molexp expands a deterministic default seed sequence and truncates it to the requested replica count.

## Runs Record Concrete Executions

Runs seeded by `exp.run(..., params=...)` derive their ids from their parameters (content-addressed), which is what makes the declaration idempotent. You can also mount runs directly:

```python
run = exp.add_run({"lr": 1e-3, "seed": 42})
```

From Python, `exp.add_run()` creates a fresh run, because a new run id is generated when you do not provide one explicitly. If you need a stable identity, pass `id=...` yourself:

```python
run = exp.add_run({"lr": 1e-3, "seed": 42}, id="baseline-seed-42")
```

The CLI uses the content-addressing mechanism to achieve repeatable runs. `molexp run` derives deterministic run ids from resolved parameters, replica index, and active profile metadata, so re-running the same script can discover and resume or skip existing runs.

Each `Run` stores its data in `<experiment>/runs/run-<run_id>/`. The most important fields live on `run.metadata`, a `RunMetadata` model. Direct convenience properties such as `run.id`, `run.parameters`, `run.status`, and `run.run_dir` simply expose the corresponding metadata fields in a more ergonomic form, and `run.get_result(key)` reads back a result value persisted by `RunContext.set_result` — falling back, when no driver-side result exists, to the completed workflow node of that name in the run's most recent execution (so results of CLI-executed runs are readable through the same accessor).

One logical run may be executed more than once. Molexp does not flatten those attempts; instead it appends `ExecutionRecord` entries to `run.metadata.execution_history`, with per-attempt state under `executions/<exec_id>/`.

If a run should be marked dead without completing normally, `run.cancel()` transitions it to `cancelled` and writes the terminal timestamp.

## RunContext Governs the Execution Lifecycle

`RunContext` is the execution-time object that turns a persistent run record into a live working directory:

```python
from molexp.workflow import WorkflowRuntime

with run.start(profile_config=cfg) as ctx:
    result = await WorkflowRuntime().execute(compiled, run_context=ctx)
```

Entering the context creates the run directories if necessary, loads previously saved result values, stores the active profile metadata onto the run, stamps the run with temporary ownership labels, switches status to `running`, and appends a fresh `ExecutionRecord` for the current attempt.

Leaving the context closes the lifecycle. A normal exit marks the run as `succeeded`, unless the workflow execution recorded a failed run status. An exception marks the run as `failed`, writes `ErrorInfo` into metadata, and stores a traceback under `executions/<execution_id>/error.txt`. In both cases the execution history entry is finalized with its end time and final status.

The most commonly used `RunContext` helpers fall into three groups. Note that they are **driver-side**: task bodies receive only the pure `TaskContext` (`ctx.inputs` / `ctx.config` / `ctx.workdir`) and cannot reach the `RunContext`.

The first group deals with results and metadata. `ctx.set_result(key, value)` and `ctx.get_result(key)` read and write the lightweight result map persisted into `run.json`; `run.get_result(key)` is the public read-back on the entity itself; when the key was never `set_result`-persisted (typical for `molexp run` CLI executions, which have no driver script calling `set_result`), it falls back to the persisted node output of the same name from the run's latest execution. Node outputs whose original value was not JSON-serializable are persisted only as lossy observability renderings and are never returned as results — `get_result` logs a warning and returns `None` for those. `ctx.set_workflow(payload)` stores a workflow-shaped dictionary onto the serialized run-context — the payload type is opaque to workspace; the workflow layer gives the dict its shape. This is distinct from the binding-registry association made by `Experiment.run(...)`, which records the live `CompiledWorkflow` instance for downstream consumers within the same process.

The second group deals with files, through typed accessors. `ctx.artifact.save(name, data)` writes under `<run_dir>/artifacts/`, automatically choosing JSON, binary, file-copy, or string serialization, and registers an `ArtifactAsset` in the catalog with a populated `Producer`. `ctx.log(name)` returns a bound log handle whose `.append(line)` writes a line into the run's logs and keeps the `LogAsset` up to date. `ctx.checkpoint(name, data=...)` writes a `CheckpointAsset`, linearly chained to the previous checkpoint of the same run.

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

## Parameter Spaces Drive the Sweep

`ParamSpace`, `GridSpace`, and `UniformSpace` belong to the workspace surface because parameter exploration is what seeds runs:

```python
from molexp import GridSpace, UniformSpace

grid = GridSpace({"lr": [1e-3, 1e-4], "batch": [32, 64]})
random_search = UniformSpace({"lr": [1e-3, 5e-4, 1e-4]}, n_samples=10, seed=42)
```

These objects are iterable generators of parameter dictionaries, and `Experiment.run(workflow, params=...)` accepts any of them directly — one content-addressed `Run` per cell:

```python
exp = project.experiment("sweep").run(compiled, params=grid)
```

A plain `{axis: [values]}` mapping is shorthand for a `GridSpace`. Because the run ids derive from the cell parameters, re-materializing the same space is a no-op.

## Script Registration for CLI Discovery

When workflows are launched from the CLI, the fluent chain is the registration point:

```python
import molexp as me
from molexp.workflow import WorkflowCompiler

wf = WorkflowCompiler(name="train")
# ... @wf.task definitions ...

(
    me.Workspace("./lab", name="lab")
    .project("demo")
    .experiment("baseline")
    .run(wf.compile(), params={"lr": [1e-3]})
)
```

`Experiment.run(...)` registers the owning workspace when the script is imported by `molexp run` (internally through the low-level `me.entry(ws)` primitive). The CLI then loads the script, reads the registered workspaces, discovers projects and experiments beneath them, and resolves which workflow belongs to which persistent experiment via `default_binding_registry.for_experiment(experiment)` — the same registry the declaration wrote to. Without that declaration step, the CLI has no supported way to discover the workspace graph from an arbitrary Python module.

## A Minimal End-to-End Example

The following example shows the full path from workflow definition to persistent run:

```python
import molexp as me
from molexp.workflow import TaskContext, WorkflowCompiler, WorkflowRuntime

wf = WorkflowCompiler(name="demo")


@wf.task
async def compute(ctx: TaskContext) -> float:
    lr = ctx.inputs["params"]["lr"]          # engine-injected root inputs
    scale = ctx.config.get("scale", 1.0)     # resolved profile / config
    return lr * scale


compiled = wf.compile()

ws = me.Workspace("./lab", name="lab")
exp = ws.project("demo").experiment("baseline").run(compiled, params={"lr": [1e-3]})

run = exp.list_runs()[0]
with run.start() as ctx:
    result = await WorkflowRuntime().execute(compiled, run_context=ctx)
    ctx.set_result("scaled_lr", result.outputs["compute"])
    ctx.artifact.save("metrics.json", {"scaled_lr": result.outputs["compute"]})

print(run.status)
print(run.metadata.execution_history[-1].status)
print(run.get_result("scaled_lr"))
```

The important part is not the arithmetic. It is the shape of the interaction. A compiled workflow remains reusable. The experiment gives that workflow a stable scientific identity. The run gives one execution attempt a stable filesystem location. `RunContext` bridges the two during execution so that results, artifacts, profile data, and failures are recorded as part of the same durable object graph.

## Runnable Example

`examples/workspace/workspace_api.py` walks the four-level hierarchy, demonstrates factory idempotence, and reopens the workspace with `Workspace.load()`.
