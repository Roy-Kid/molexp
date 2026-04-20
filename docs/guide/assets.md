# Unified Asset Model

In MolExp every persistent byproduct of an experiment — a dataset imported from outside, a model checkpoint produced by training, a log file written by a running task, or the execution state of a workflow — is represented by a single construct called an **Asset**. Assets are typed, scoped, and indexed, but their bytes stay in the natural place the producer wrote them. The asset model exists so that you can ask "what did this run produce?" or "give me every failed run in this experiment" without having to walk the filesystem by hand, while still keeping every run directory self-contained enough to be moved, archived, or inspected on its own.

This guide explains the mental model: the kinds of asset you encounter, how they are scoped inside the workspace, how producers are attributed, and how the catalog keeps the view consistent without becoming the source of truth.

## Why a unified model

Older revisions of MolExp kept each category of output in its own API. `ctx.save_artifact()` wrote artifacts, ad-hoc `ctx.log(...)` wrote logs, workflow persistence wrote `workflow.json`, and imported datasets lived in an `AssetLibrary`. This was workable but brittle. Each surface had its own filesystem layout, its own metadata, and its own queries, so simple questions like "which task produced this file" or "what is the total disk footprint of this experiment" were disproportionately hard to answer.

The unified model keeps what was already good — the physical layout where producers naturally write — and adds two things on top: a shared typed record, and a workspace-wide index.

## The Asset class hierarchy

Every asset is a Pydantic model with a common shape (id, name, scope, path, timestamps, producer, tags) and a `kind` discriminator that selects the subclass:

- `DataAsset` — data imported from outside the workspace, with `source_path` and `import_action`.
- `ArtifactAsset` — a file written by a task, optionally with a `mime` and `size`.
- `LogAsset` — a structured line-oriented log, with `line_count`.
- `CheckpointAsset` — a workflow checkpoint, with a `ckpt_id` and an optional `parent_ckpt_id` that forms a linear chain.
- `ErrorTraceAsset` — a captured exception, with `exception_type`, `message`, and `execution_id`.
- `ExecutionStateAsset` — a snapshot of pydantic-graph persistence (`workflow.json`).
- `OutputAsset` — a named output value promoted from a run so downstream workflows can consume it.

All of these serialize as the same `AssetResponse` JSON at the API boundary — the discriminator is the `kind` field, and the subclass-specific fields land in `extra` so the frontend can render them without a schema per kind.

Serialization round-trips through `parse_asset(dict)`, backed by a Pydantic `TypeAdapter`. Reading an `assets.json` manifest or a catalog entry always returns the correct subclass — you never lose the type.

## Scopes

Every asset declares the level at which it is meaningful. MolExp uses four scopes, mirroring the workspace hierarchy:

```
Workspace → Project → Experiment → Run
```

A `DataAsset` imported into `ws.data_assets` has `scope.kind = "workspace"` and empty `scope.ids`. An artifact written by a task has `scope.kind = "run"` and `scope.ids = (project_id, experiment_id, run_id)`. Scopes carry enough information to reconstruct the on-disk directory for the asset's manifest without consulting any catalog.

Scope matters because it decides who can reuse the asset. A workspace-scoped dataset is visible to every project in the workspace. An experiment-scoped feature cache is visible to every run of that experiment. A run-scoped log is visible only within that run. `ctx.find_asset(name)` walks outward from run → experiment → project → workspace until it finds a match.

## Where the bytes live

The unified model does **not** centralise payloads. Files stay where their producer wrote them:

```
<workspace_root>/
├── .catalog/                       # derived index
│   └── index.json
├── assets.json                     # workspace-scoped manifest
├── data_assets/<asset_id>/payload/ # imported DataAssets
└── projects/<project_id>/
    ├── assets.json                 # project-scoped manifest
    └── experiments/<exp_id>/
        ├── assets.json             # experiment-scoped manifest
        └── runs/<run_id>/
            ├── assets.json         # run-scoped manifest
            ├── artifacts/          # ArtifactAsset payloads
            ├── logs/               # LogAsset payloads
            ├── .ckpt/              # CheckpointAsset payloads
            └── execution/<exec_id>/
                ├── workflow.json   # ExecutionStateAsset
                └── error.txt       # ErrorTraceAsset
```

Each scope directory owns an `assets.json` manifest that lists the typed `Asset` records for assets in that scope. Manifest `path` fields are relative to the scope directory, so a run directory stays portable — you can tar it, move it to another workspace, and rebuild will re-register every asset from the local manifest.

This layout is the single source of truth. If the catalog is lost, the filesystem is still complete.

## The workspace catalog

On top of the per-scope manifests MolExp maintains a derived index at `<root>/.catalog/index.json`. The catalog stores the full set of workspaces, projects, experiments, runs, executions, and assets as separate tables, together with a `consumes` table that records which run consumed which asset. Its role is to answer queries that would otherwise require walking the filesystem:

```python
catalog = ws.catalog
failed_runs = [r for r in catalog.query_runs(experiment_id="baseline") if r["status"] == "failed"]
error_traces = catalog.query_assets(kind="error_trace", producer_run=failed_runs[0]["run_id"])
```

Writes go through the catalog the moment a run, execution, or asset is registered. Reads use the catalog directly. Because the catalog is fully derivable, `catalog.rebuild()` walks the filesystem, replays every manifest, and regenerates the index without referring to the previous file. That is the invariant the system relies on: the catalog is convenient, but never load-bearing.

## Producers and attribution

Assets produced during a run carry a `Producer` record: `run_id`, `execution_id`, and optionally `task_id`. The producer is populated automatically by the typed accessors the `RunContext` exposes:

```python
with run.start() as ctx:
    ctx.set_active_task("train")
    asset = ctx.artifact.save("metrics.json", {"loss": 0.1})
    # asset.producer.run_id == run.id
    # asset.producer.task_id == "train"
    # asset.producer.execution_id == ctx._execution_id
    log = ctx.log("train")
    log.append("epoch 1")
    ckpt = ctx.checkpoint("epoch-1", data={"step": 1})
```

`ArtifactAccessor.save()`, `LogAccessor.__call__()`, and `CheckpointAccessor.__call__()` all write to the right natural directory, register the asset in the scope's manifest, and upsert it into the catalog in one atomic step. `set_active_task(task_id)` scopes subsequent writes to a specific task so that a run with many tasks still produces clearly-attributed assets.

Checkpoints additionally chain: the second checkpoint of a run has its `parent_ckpt_id` set to the first, so you can follow the lineage without a separate table.

## Concurrency and atomicity

Manifests are protected by a process-local `threading.Lock` so concurrent task writes inside one run can all land safely in `assets.json`. Catalog writes use the same atomic-rename pattern (`_atomic_write_json`) used elsewhere in the workspace. A parallel run that writes twenty artifacts from a thread pool ends up with twenty entries in the manifest and twenty entries in the catalog; nothing is lost and nothing is half-written.

Crashes are survivable. A run that exits mid-execution leaves the manifests it already wrote intact. When the workspace is reopened the catalog can be rebuilt from those manifests — or from a portion of them, if some have been moved or deleted — without manual recovery.

## Importing external data

Data that originates outside the workspace enters through a `DataAssetLibrary`. Each scope that owns managed data exposes one as `scope.data_assets`:

```python
dataset = ws.data_assets.import_asset("qm9", "/data/qm9.tar.bz2")
project_dataset = project.data_assets.import_asset("lig-library", "/tmp/ligands.csv")
```

The import stores the payload under `<scope>/data_assets/<asset_id>/payload/` and registers a `DataAsset` that remembers the action used (`copy` / `move` / `symlink` / `hardlink`) and the source path. Because `DataAsset` is just another subclass in the unified model, it shows up in the same catalog queries and the same UI as artifacts and logs.

## Querying from the UI

On the server side every `AssetResponse` exposes the same envelope — `id`, `name`, `kind`, `scope_kind`, `scope_ids`, `path`, `created_at`, `updated_at`, `producer`, `tags`, `extra` — so the frontend can use a single table widget filterable by kind, scope, producing run, or tag. The typed `AssetViewer` dispatches on `kind` to pick the right content preview: a log tail for `LogAsset`, a JSON tree for `CheckpointAsset` / `ExecutionStateAsset`, a stack-trace header for `ErrorTraceAsset`, and a file preview for anything with bytes.

## Limits

The asset model is narrower than a general data catalog and intentionally so.

- Asset identity is local to the workspace. There is no content-addressed deduplication across libraries and no global identifier scheme.
- `tags` and `extra` are free-form. MolExp stores them but does not impose a controlled vocabulary.
- The catalog is an index, not a database. It is regenerable and not authoritative.
- Runs are execution attempts. Assets they produce live in their run directory; experiment- or project-scoped promotion is an explicit import step.

These choices keep the implementation simple, keep individual run directories portable, and leave the door open for richer consumers — a FAIR-style publishing layer, an external search index, a content-addressed store — to be added on top without rewriting the core.

## Runnable Example

`examples/workspace/assets.py` imports a data asset, writes artifact/log/checkpoint assets from a tracked run, and then runs a workspace-wide catalog query.
