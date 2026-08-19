# Unified Asset Model

In MolExp every persistent byproduct of an experiment — a dataset imported from outside, a model checkpoint produced by training, a log file written by a running task, or the execution state of a workflow — is represented by a single construct called an **Asset**. Assets are typed, scoped, and recorded in per-scope manifests, but their bytes stay in the natural place the producer wrote them. The asset model exists so that you can ask "what did this run produce?" or "which task wrote this file?" without having to walk the filesystem by hand, while still keeping every run directory self-contained enough to be moved, archived, or inspected on its own.

This guide explains the mental model: the kinds of asset you encounter, how they are scoped inside the workspace, how producers are attributed, and how queries scan the authoritative manifests directly — there is no derived asset index to keep in sync or to distrust.

## Why a unified model

Older revisions of MolExp kept each category of output in its own API. `ctx.save_artifact()` wrote artifacts, ad-hoc `ctx.log(...)` wrote logs, workflow persistence wrote `workflow.json`, and imported datasets lived in an `AssetLibrary`. This was workable but brittle. Each surface had its own filesystem layout, its own metadata, and its own queries, so simple questions like "which task produced this file" or "what is the total disk footprint of this experiment" were disproportionately hard to answer.

The unified model keeps what was already good — the physical layout where producers naturally write — and adds two things on top: a shared typed record, and a workspace-wide query surface that scans those records.

## The Asset class hierarchy

Every asset is a Pydantic model with a common shape (id, name, scope, path, timestamps, producer, tags) and a `kind` discriminator that selects the subclass:

- `DataAsset` — data imported from outside the workspace, with `source_path` and `import_action`.
- `ArtifactAsset` — a file written by a task, optionally with a `mime` and `size`.
- `LogAsset` — a structured line-oriented log, with `line_count`.
- `CheckpointAsset` — a workflow checkpoint, with a `ckpt_id` and an optional `parent_ckpt_id` that forms a linear chain.
- `ErrorTraceAsset` — a captured exception, with `exception_type`, `message`, and `execution_id`.

All of these serialize as the same `AssetResponse` JSON at the API boundary — the discriminator is the `kind` field, and the subclass-specific fields land in `extra` so the frontend can render them without a schema per kind.

Serialization round-trips through `parse_asset(dict)`, backed by a Pydantic `TypeAdapter`. Reading an `assets.json` manifest always returns the correct subclass — you never lose the type.

## Scopes

Every asset declares the level at which it is meaningful. MolExp uses four scopes, mirroring the workspace hierarchy:

```
Workspace → Project → Experiment → Run
```

A `DataAsset` imported into `ws.data_assets` has `scope.kind = "workspace"` and empty `scope.ids`. An artifact written by a task has `scope.kind = "run"` and `scope.ids = (project_id, experiment_id, run_id)`. Scopes carry enough information to reconstruct the on-disk directory for the asset's manifest without consulting any index.

Scope matters because it decides who can reuse the asset. A workspace-scoped dataset is visible to every project in the workspace. An experiment-scoped feature cache is visible to every run of that experiment. A run-scoped log is visible only within that run. `ctx.find_asset(name)` walks outward from run → experiment → project → workspace until it finds a match.

## Where the bytes live

The unified model does **not** centralise payloads. Files stay where their producer wrote them:

```
<workspace_root>/
├── assets.json                     # workspace-scoped manifest
├── data_assets/<asset_id>/payload/ # imported DataAssets
└── projects/<project_id>/
    ├── assets.json                 # project-scoped manifest
    └── experiments/<exp_id>/
        ├── assets.json             # experiment-scoped manifest
        └── runs/run-<run_id>/
            ├── assets.json         # run-scoped manifest
            ├── artifacts/          # ArtifactAsset payloads
            ├── .ckpt/              # CheckpointAsset payloads
            └── executions/<exec_id>/
                ├── workflow.json   # workflow execution state (not an Asset)
                ├── logs/<name>.log # LogAsset payloads for this attempt
                └── error.txt       # ErrorTraceAsset
```

Each scope directory owns an `assets.json` manifest that lists the typed `Asset` records for assets in that scope. Manifest `path` fields are relative to the scope directory, so a run directory stays portable — you can tar it, move it to another workspace, and every asset re-registers from the local manifest.

This layout is the single — and only — source of truth. There is no derived asset index beside it (the former `catalog/index.sqlite` was removed): queries scan these manifests directly.

## Producers and attribution

Assets produced during a run carry a `Producer` record: `run_id`, `execution_id`, and optionally `task_id`. The producer is populated automatically by the typed accessors the `RunContext` exposes:

```python
import molexp as me

ws = me.Workspace("./lab", name="lab")
run = ws.project("demo").experiment("baseline").add_run({"lr": 1e-3})

with run.start() as ctx:
    ctx.set_active_task("train")
    asset = ctx.register_artifact({"loss": 0.1}, name="metrics.json")
    # asset.producer.run_id == run.id
    # asset.producer.task_id == "train"
    log = ctx.log("train")
    log.append("epoch 1")
    ckpt = ctx.checkpoint("epoch-1", data={"step": 1})
```

`ctx.register_artifact(...)`, `ctx.log(name).append(...)`, and `ctx.checkpoint(...)` all write to the right natural directory and register the asset in the scope's manifest in one atomic step. `set_active_task(task_id)` scopes subsequent writes to a specific task so that a run with many tasks still produces clearly-attributed assets. (When the workflow engine executes a task under a tracked run, it tags the active task for you.)

## Querying across scopes

Every scope exposes a read-only `assets` view (`ws.assets`, `project.assets`, `exp.assets`, `run.assets`) whose queries scan the authoritative `assets.json` manifests on demand — `list()` returns the scope's own assets, `query(...)` filters by kind, producer, or tag, and `recursive=True` widens the match to every sub-scope:

```python
run_artifacts = run.assets.query(kind="artifact")
trained = ws.assets.query(producer_task="train", recursive=True)
error_traces = ws.assets.query(kind="error_trace", recursive=True)
```

When all you hold is a workspace root — no live `Workspace` object — the module-level helpers in `molexp.workspace.assets.scan` do the same scan:

```python
from molexp.workspace.assets import scan

everything = scan.scan_assets(ws.root)
same_asset = scan.get_asset(ws.root, run_artifacts[0].asset_id)
```

`scan.find_by_content_hash(root, "sha256:…")` resolves a content hash to the earliest asset carrying those bytes — the lookup the workflow cache uses to re-register cache hits.

Checkpoints additionally chain: the second checkpoint of a run has its `parent_ckpt_id` set to the first, so you can follow the lineage without a separate table.

## Concurrency and atomicity

Manifests are protected by a process-local `threading.Lock` so concurrent task writes inside one run can all land safely in `assets.json`, and every manifest write uses the same atomic-rename pattern (temp file + `os.rename`) used elsewhere in the workspace. A parallel run that writes twenty artifacts from a thread pool ends up with twenty entries in the manifest; nothing is lost and nothing is half-written.

Crashes are survivable. A run that exits mid-execution leaves the manifests it already wrote intact, and because queries scan those manifests directly, whatever was registered before the crash is immediately visible when the workspace is reopened — no rebuild step, no manual recovery.

## Importing external data

Data that originates outside the workspace enters through a `DataAssetLibrary`. Each scope that owns managed data exposes one as `scope.data_assets`:

```python
from pathlib import Path

Path("ligands.csv").write_text("smiles\nCCO\n")

dataset = ws.data_assets.import_asset("lig-library", "ligands.csv")
project_dataset = ws.project("demo").data_assets.import_asset("lig-subset", "ligands.csv")
```

The import stores the payload under `<scope>/data_assets/<asset_id>/payload/` and registers a `DataAsset` that remembers the action used (`copy` / `move` / `symlink` / `hardlink`) and the source path. Because `DataAsset` is just another subclass in the unified model, it shows up in the same manifest-backed queries and the same UI as artifacts and logs.

## Querying from the UI

On the server side every `AssetResponse` exposes the same envelope — `id`, `name`, `kind`, `scope_kind`, `scope_ids`, `path`, `created_at`, `updated_at`, `producer`, `tags`, `extra` — so the frontend can use a single table widget filterable by kind, scope, producing run, or tag. The typed `AssetViewer` dispatches on `kind` to pick the right content preview: a log tail for `LogAsset`, a JSON tree for `CheckpointAsset`, a stack-trace header for `ErrorTraceAsset`, and a file preview for anything with bytes.

## Limits

The asset model is narrower than a general data catalog and intentionally so.

- Asset identity is local to the workspace. There is no content-addressed deduplication across libraries and no global identifier scheme.
- `tags` and `extra` are free-form. MolExp stores them but does not impose a controlled vocabulary.
- Queries are linear scans over the manifests. That is a deliberate trade: no derived index means no index to drift, at the cost of raw query speed on very large workspaces.
- Runs are execution attempts. Assets they produce live in their run directory; experiment- or project-scoped promotion is an explicit import step.

These choices keep the implementation simple, keep individual run directories portable, and leave the door open for richer consumers — a FAIR-style publishing layer, an external search index, a content-addressed store — to be added on top without rewriting the core.

## Runnable Example

`examples/workspace/assets.py` imports a data asset, writes artifact/log/checkpoint assets from a tracked run, and then runs workspace-wide queries with `scan.scan_assets` over the on-disk manifests.
