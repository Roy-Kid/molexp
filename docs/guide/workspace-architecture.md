# Workspace Architecture

MolExp organizes experiments in a four-tier hierarchy:

```
Workspace
└── Project
    └── Experiment      (parameter-space container + replica config)
        └── Run          (one execution attempt; re-runs append ExecutionRecords)
```

Every persistent byproduct — imported data, task artifacts, logs, checkpoints, error traces, workflow execution state — is a typed `Asset` subclass recorded in a per-scope `assets.json` manifest and indexed by the workspace catalog at `.catalog/index.json`. Every metadata write is atomic (temp-file + `os.rename`), so a crash never leaves a half-written JSON file.

Workspace is the bottom of the molexp dependency DAG. It owns filesystem layout, atomic JSON, content-addressed assets, and generic per-kind subsystem storage — and **does not know about workflows, sessions, agents, or LLMs**. Upstream layers (workflow, agent) reach *down* into workspace's public surface; the inverse is forbidden by the import-guard test.

## Hierarchy Levels

| Level | Directory | Metadata file | Purpose |
|-------|-----------|---------------|---------|
| `Workspace` | `<root>/` | `workspace.json` | Top-level container. Materialized explicitly or when the first child is created. |
| `Project` | `<root>/projects/<project_id>/` | `project.json` | Research-area container (MD, ML training, data pipeline, …). |
| `Experiment` | `<project>/experiments/<exp_id>/` | `experiment.json` | Concrete parameter set + replica count. Workspace stores the parameter binding; pairing the experiment with a workflow is the *caller's* concern. |
| `Run` | `<exp>/runs/run-<run_id>/` | `run.json` | Single execution instance; re-runs append `ExecutionRecord` entries. |

Children are **not** stored as lists in the parent metadata — parents discover children by scanning the filesystem. This keeps writes local and avoids lock contention.

## Why This Design

Scientific workflows tend to run the same pipeline many times with different parameters, then compare outcomes. The `Experiment → Run` split reflects that:

- **Experiment** is the *definition* — parameters, replica count, seeds, optional advisory `workflow_source` / `workflow_type` strings used by the UI for grouping.
- **Run** is a *realization* — one execution, one set of concrete parameter values, one outcome.

Each `Run` captures reproducibility metadata: an opaque `workflow_snapshot` payload (the canonical typed shape lives in `molexp.workflow.WorkflowSnapshotRef`; workspace stores it as a JSON `dict`), the resolved molcfg profile, a `config_hash`, execution history, error info, and produced artifacts.

## Creating a Hierarchy

The hierarchy is created from the top down, but not every step has identical identity rules. `ws.Project(...)` and `project.Experiment(...)` are get-or-create operations keyed by slug or explicit id, so repeated calls can load existing objects from disk. `exp.run(...)` is different: it creates a fresh run unless you provide an explicit `id`, in which case it becomes a get-or-load operation for that concrete run directory.

```python
import molexp as me

ws = me.Workspace("./lab")                                # lightweight object; no files yet
project = ws.Project("MD Simulations")                    # materializes workspace.json and project.json
exp = project.Experiment(
    "temperature-300K",
    params={"T": 300, "pressure": 1.0},
    n_replicas=3,
    seeds=[42, 43, 44],
    workflow_source="workflows/md.py",                    # advisory free-form string
    git_commit="abc123",
)
run = exp.run(
    parameters={"T": 300, "pressure": 1.0},
    id="temperature-300K-seed-42",
)                                                         # materializes run.json
```

Re-calling the project or experiment factory with the same name or id returns the same in-memory object within the current process and loads from disk when needed. Runs only behave that way when the run id is stable.

## Parameter Combinations

`GridSpace` and `UniformSpace` generate parameter combinations. One combination = one `Experiment`:

```python
from molexp import GridSpace

grid = GridSpace({"T": [300, 310, 320], "force_field": ["amber", "charmm"]})

for params in grid:
    slug = f"T{params['T']}-{params['force_field']}"
    exp = project.Experiment(slug, params=params, n_replicas=3, workflow_source="md.py")
    for seed in exp.get_seeds():
        run = exp.Run(parameters={**params, "seed": seed})
        await spec.execute(run=run)
```

`UniformSpace(param_values, n_samples, seed=None)` samples `n_samples` combinations uniformly at random — handy for broader search spaces.

## Pairing an Experiment with a Workflow

Workspace does **not** track the experiment-to-workflow association. The `Experiment` is a parameter container; pairing it with a workflow happens at execution time in the caller's code:

```python
from molexp.workflow import Workflow, promote_callable, WorkflowBuilder

# Build (or load) a Workflow however you like — workspace doesn't care.
spec = WorkflowBuilder(name="train").add(TrainTask()).build()
# or, for a bare fn(RunContext):
spec = promote_callable(train_fn, name="train")

# Workspace just provides the Run that the workflow executes within.
run = exp.Run(parameters={"lr": 1e-3})
result = await spec.execute(run=run)
```

This intentional decoupling came out of the 2026-05-09 rectification: workspace stays a storage primitive, workflow stays a graph engine, and the *orchestration* layer (typically the agent layer or a user script) decides which workflow to run against which experiment.

## Executing a Run

```python
result = await spec.execute(run=run)
```

Under the hood, the workflow runtime opens a `RunContext` which:

1. Ensures `artifacts/`, `logs/`, and `.ckpt/` subdirectories exist.
2. Records temporary ownership metadata for the active execution.
3. Appends a new `ExecutionRecord` to `run.execution_history`.
4. Sets `run.status = "running"` and runs the workflow.
5. On success, writes `status="succeeded"` plus the final timestamp.
6. On failure, writes `status="failed"`, an `ErrorInfo`, and registers an `ErrorTraceAsset` pointing at `execution/<exec_id>/error.txt`.

Every attempt appears in `run.metadata.execution_history`, newest last — a run that was retried twice will have three records.

## Assets

Every scope exposes a typed `assets` view into the workspace catalog and a `data_assets` library for importing external data. Run-time writes (artifacts, logs, checkpoints) go through the `RunContext` accessors, which keep the on-disk manifest and the catalog in sync.

```python
ws.data_assets.import_asset("bert_model", "/models/bert.pt")
project.data_assets.import_asset("dataset", "/data/qm9.tar.bz2")
exp.data_assets.import_asset("features", "/data/features.h5")

# Inside a task
class Train(Task):
    async def execute(self, ctx: TaskContext) -> dict:
        dataset = ctx.find_asset("dataset")       # walks run → experiment → project → workspace
        features = ctx.find_asset("features")
        ctx.artifact.save("metrics.json", {"val_loss": 0.1})
        ctx.log("train").append("epoch 1")
        ...
```

`import_asset(name, src, action="copy", meta=None)` supports `"copy"`, `"move"`, `"symlink"`, and `"hardlink"` for ingestion. See [Unified Asset Model](assets.md) for the full list of asset kinds (artifact, log, checkpoint, error trace, execution state, output, data) and how the catalog can be rebuilt from disk when needed.

## CLI Surface

The same hierarchy is exposed through the CLI:

```bash
molexp project   create|list|info
molexp experiment create|list
molexp runs      create|list|info|cancel|prune
molexp asset     list
molexp info      # show workspace summary
```

`molexp runs prune` interactively walks the project → experiment → run → execution tree and lets you delete per-execution records (removes `execution/<exec_id>/` and rewrites `run.execution_history`).

## Directory Layout

```
./lab/
├── workspace.json
├── .catalog/index.json             # derived workspace-wide catalog
├── assets.json                     # workspace-scoped asset manifest
├── data_assets/<asset_id>/payload/ # imported DataAssets
└── projects/
    └── qm9/
        ├── project.json
        ├── assets.json             # project-scoped asset manifest
        ├── data_assets/            # project-scoped DataAssets
        └── experiments/
            └── baseline/
                ├── experiment.json
                ├── assets.json     # experiment-scoped asset manifest
                ├── data_assets/    # experiment-scoped DataAssets
                └── runs/
                    └── run-<id>/
                        ├── run.json
                        ├── assets.json     # run-scoped asset manifest
                        ├── artifacts/
                        ├── logs/
                        ├── .ckpt/
                        └── execution/<exec_id>/   # per-attempt workflow.json + error.txt
```

## Runnable Example

`examples/workspace/workspace_architecture.py` seeds a tracked run and then prints the full on-disk tree with file sizes so you can see each layer for yourself.
