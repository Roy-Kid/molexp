# Asset Management

Assets are reusable, named artifacts stored in a per-scope `AssetLibrary`. Every level of the workspace hierarchy (Workspace, Project, Experiment, Run) owns its own library; libraries are **isolated** — assets in one do not leak into a parent or child.

```
./lab/                         ← ws.assets
├── assets/
│   └── <asset_id>/
│       ├── asset.json
│       └── payload/
└── projects/
    └── qm9/                   ← project.assets
        ├── assets/
        └── experiments/
            └── baseline/      ← exp.assets
                ├── assets/
                └── runs/
                    └── run-xyz/   ← run.assets (via RunContext)
```

## The Asset Data Model

```python
class Asset(BaseModel):
    asset_id: str              # auto-generated on import
    name: str                  # user-provided label (unique per library)
    library_root: Path
    created_at: datetime
    metadata: dict[str, Any]

    uri   →  "asset://<asset_id>"
    path  →  library_root / asset_id / "payload"
```

The managed layout `<library>/<asset_id>/{asset.json, payload/}` makes it easy to inspect or relocate a single asset without scanning the whole store.

## Importing Assets

Every level of the hierarchy exposes `.assets` (an `AssetLibrary`). Import content via `import_asset(name, src, action="copy", meta=None)`:

```python
ws.assets.import_asset("bert_model", "/models/bert.pt")
project.assets.import_asset("qm9", "/data/qm9.tar.bz2")
experiment.assets.import_asset("features", "/data/features.h5", meta={"split": "train"})

# Project shortcut
project.import_asset("dataset", "/data/raw.csv")
```

`action` controls how the source is materialized into the store:

| action | Behaviour |
|--------|-----------|
| `"copy"` (default) | `shutil.copy2` / `shutil.copytree` |
| `"move"` | `shutil.move` (removes the source) |
| `"symlink"` | `Path.symlink_to` (pointer, no data copied) |
| `"hardlink"` | `os.link` (zero-copy dedup); falls back to copy if the filesystem refuses |

`name` must be unique **within the library**. Re-importing with the same name raises `ValueError`.

## Reading Assets

```python
asset = experiment.assets.get_asset("features")
if asset:
    path = asset.path               # materialized content
    meta = asset.metadata

# List everything in a library
for a in project.assets.list_assets():
    print(a.name, a.uri, a.path)
```

Inside a workflow task, `ctx.find_asset(name)` walks the hierarchy automatically:

```python
class Train(Task):
    async def execute(self, ctx: TaskContext) -> dict:
        ds = ctx.find_asset("dataset")      # experiment → project → workspace
        if ds is None:
            raise FileNotFoundError("dataset not available in any scope")
        return train(ds.path)
```

## RunContext-Level Asset Helpers

`RunContext` exposes convenience methods for runs that interact with the experiment's asset library:

```python
with run.start() as ctx:
    ctx.register_asset("run_output", "/tmp/output.h5", action="move")
    ctx.get_asset("dataset", scope="project")     # "experiment" | "project" | "workspace"
    ctx.find_asset("dataset")                      # walk experiment → project → workspace
    data_dir = ctx.get_data_dir("cache", fallback="shared/cache")  # asset lookup + lazy mkdir
```

`register_asset` is a thin wrapper around `run.experiment.assets.import_asset(...)` — registering a produced artifact at the experiment scope so later replicas can share it.

## Asset Workflows

For repeatable ingestion pipelines (download → extract → register), `AssetLibrary.add_workflow(name, AssetWorkflow(steps))` lets you chain callables and invoke them via `run_workflow(name, **kwargs)`:

```python
from molexp.workspace import AssetWorkflow

def download(**kw):
    fetch(kw["url"], "/tmp/raw.tar")
    return {"asset_path": "/tmp/raw.tar"}

def extract(**kw):
    extract_tarball(kw["asset_path"], "/data/unpacked")
    return {"asset_name": "dataset", "asset_path": "/data/unpacked"}

ws.assets.add_workflow(
    "download_dataset",
    AssetWorkflow("download_dataset", [download, extract]),
)

asset = ws.assets.run_workflow("download_dataset", url="https://example.com/data.tar")
```

The final step must populate `asset_name` and `asset_path`; the resulting file or directory is imported into the library as a regular asset (action `"copy"`).

## What This Library Does *Not* Do

- **No content-hash deduplication across libraries.** Identical content stored in two libraries occupies disk twice. Use `action="hardlink"` or `"symlink"` if you want to avoid duplication.
- **No implicit "current run" asset repo.** Assets are always attached to an explicit library; there is no `register_asset(...)` global you can call from outside a `RunContext`.
- **No typed `AssetType` taxonomy.** Assets carry free-form `metadata`; use it however you like (`meta={"kind": "trajectory"}` is fine).

## CLI

```bash
molexp asset list                  # workspace-level assets
```

Project / experiment / run-level listing is currently done through the Python API — use `project.assets.list_assets()` etc.
