# Unified Asset Model

**Status**: Approved · **Author**: @RoyKid · **Date**: 2026-04-19

## 1. Motivation

Every output molexp persists is organized by folder convention today, with different lifecycles, different metadata (or none), and no cross-scope index:

```text
run_dir/
├── artifacts/             free-form files; filename is the only key
├── logs/                  run.log (append), error.txt (overwritten)
├── execution/<exec_id>/   workflow.json, error.txt
└── .ckpt/                 <ckpt_id>.json blobs

{workspace,project,experiment}/assets/   ← the only indexed layer
```

Consequences: "all failed-run stack traces in experiment X" is a filesystem walk; `save_artifact`, direct `logs_dir` writes, `checkpoint`, and `RunStorePersistence` use four incompatible code paths; lineage lives in filenames; UI needs separate renderers per kind.

**Goals**:

1. **Separate physical layout from logical catalog.** Assets stay where producers naturally write them; the catalog references them by path.
2. **FAIR + portable runs.** Every run directory is self-describing. Zip it, move it to another workspace, rebuild the catalog — it works. The workspace catalog is a pure derived view.
3. **Typed subclasses, no god class.** `LogAsset.tail()`, `CheckpointAsset.parent_ckpt_id`, etc. live on the typed subclass.

This is a refactor. **No backwards compatibility.** Old call sites break and are updated in-place.

## 2. Design Principles

1. **Filesystem is the source of truth.** Every fact is recoverable from on-disk JSON files. The workspace catalog is always regenerable.
2. **Each scope directory is self-contained.** A `run/` tarball dropped into another workspace is fully reproducible.
3. **JSON everywhere.** Per-scope `assets.json` manifest; workspace-level `.catalog/index.json`. No SQLite, no JSONL, no sidecar sprawl.
4. **One workspace catalog.** Indexes the whole hierarchy — workspaces, projects, experiments, runs, executions, assets — as sections of a single JSON file.
5. **Typed subclasses.** Pydantic discriminated union on `kind`.
6. **Atomic writes.** Temp-file + `os.rename` for every JSON mutation. Unchanged from existing policy.
7. **Paths are relative to scope root.** Survives directory moves.

## 3. Asset Class Hierarchy

```text
src/molexp/workspace/assets/
├── __init__.py        re-exports
├── base.py            Asset, Producer, AssetScope, AnyAsset union
├── data.py            DataAsset
├── artifact.py        ArtifactAsset
├── log.py             LogAsset
├── error.py           ErrorTraceAsset
├── checkpoint.py      CheckpointAsset
├── execution.py       ExecutionStateAsset
├── output.py          OutputAsset
├── manifest.py        AssetManifest (per-scope JSON I/O)
└── catalog.py         AssetCatalog (workspace JSON I/O)
```

### 3.1 Base

```python
class Producer(BaseModel, frozen=True):
    run_id: str | None = None
    execution_id: str | None = None
    task_id: str | None = None

class AssetScope(BaseModel, frozen=True):
    kind: Literal["workspace", "project", "experiment", "run"]
    ids: tuple[str, ...]          # e.g. ("proj_42", "exp_7", "run-abc") for run scope

class Asset(BaseModel, ABC):
    asset_id: str                  # UUID
    name: str                      # unique within (scope, kind)
    scope: AssetScope
    path: Path                     # RELATIVE to scope root
    created_at: datetime
    updated_at: datetime
    producer: Producer | None
    tags: dict[str, str] = {}

    kind: ClassVar[str]

    @property
    def uri(self) -> str:
        return f"asset://{self.scope.kind}/{'/'.join(self.scope.ids)}/{self.asset_id}"

    def absolute_path(self, scope_dir: Path) -> Path:
        return scope_dir / self.path
```

### 3.2 Subclasses

Each one file, each with the fields and methods only its kind needs:

- **`DataAsset`** (`kind="data"`): user-imported inputs. Fields: `source_path`, `import_action`, `content_hash`. Lives at `{scope}/assets/<asset_id>/payload/`.
- **`ArtifactAsset`** (`kind="artifact"`): structured run outputs. Fields: `mime`, `size`, `content_hash`. Lives at `run_dir/artifacts/<name>`.
- **`LogAsset`** (`kind="log"`): append-only text streams. Methods: `append(line)`, `tail(n)`, `stream()`. Lives at `run_dir/logs/<name>.log`.
- **`ErrorTraceAsset`** (`kind="error_trace"`): captured exceptions. Fields: `exception_type`, `message`, `execution_id`. Lives at `run_dir/execution/<exec_id>/error.txt`.
- **`CheckpointAsset`** (`kind="checkpoint"`): mid-run state. Fields: `ckpt_id`, `parent_ckpt_id`. Lives at `run_dir/.ckpt/<ckpt_id>.json`.
- **`ExecutionStateAsset`** (`kind="execution_state"`): workflow snapshot. Fields: `execution_id`, `workflow_id`, `status`. Lives at `run_dir/execution/<exec_id>/workflow.json`.
- **`OutputAsset`** (`kind="output"`): final bound result. Fields: `result_key`, `value_type`.

Discriminated union:

```python
AnyAsset = Annotated[
    DataAsset | ArtifactAsset | LogAsset | ErrorTraceAsset
    | CheckpointAsset | ExecutionStateAsset | OutputAsset,
    Field(discriminator="kind"),
]
```

## 4. Per-Scope Manifest: `assets.json`

Every scope directory owns one `assets.json` file — the portable, filesystem-local source of truth for that scope's assets.

```text
workspace_root/assets.json
workspace_root/projects/<id>/assets.json
workspace_root/projects/<id>/experiments/<id>/assets.json
workspace_root/projects/<id>/experiments/<id>/runs/<id>/assets.json
```

Format:

```json
{
  "schema_version": 1,
  "assets": {
    "a-abc123": {"kind": "artifact", "name": "metrics.json", "path": "artifacts/metrics.json", ...},
    "a-def456": {"kind": "log", "name": "run", "path": "logs/run.log", ...}
  }
}
```

Writes: load → mutate dict → atomic write via temp-file + rename. Process-local lock (threading or asyncio) around load-mutate-write. One run process at a time writes to its own manifest, so no cross-process locks needed.

The pair `{ run.json + assets.json + physical files }` is self-sufficient: a run directory can be tarred, moved to another workspace, and fully reopened.

## 5. Workspace Catalog: `.catalog/index.json`

One JSON file indexing the whole hierarchy. Derived view — rebuildable from scope manifests.

```text
workspace_root/.catalog/index.json
```

Structure:

```json
{
  "schema_version": 1,
  "workspaces": {"<id>": {...}},
  "projects": {"<id>": {"workspace_id": "...", "name": "...", "path": "...", ...}},
  "experiments": {"<id>": {"project_id": "...", ...}},
  "runs": {"<id>": {"experiment_id": "...", "status": "...", "path": "...", ...}},
  "executions": {"<id>": {"run_id": "...", "status": "...", ...}},
  "assets": {"<id>": {"kind": "...", "scope_kind": "...", "scope_id": "...", "path": "...", "producer": {...}, ...}},
  "consumes": [{"execution_id": "...", "task_id": "...", "asset_id": "..."}]
}
```

All dict sections are keyed by ID for O(1) lookup. Queries filter in memory:

```python
catalog.query_assets(kind="log", scope=run.scope)
catalog.query_runs(experiment_id=..., status="failed")
```

For reasonable workspaces (≤ 10k runs) the whole file fits in memory and loads in milliseconds. Scale beyond that is §8 future work.

### 5.1 API

```python
class AssetCatalog:
    def __init__(self, workspace_root: Path) -> None: ...

    def register(self, asset: Asset) -> None: ...
    def update(self, asset: Asset) -> None: ...
    def upsert_run(self, run: Run) -> None: ...
    def upsert_execution(self, record: ExecutionRecord) -> None: ...
    # ... same for workspace/project/experiment

    def get(self, asset_id: str) -> Asset | None: ...
    def resolve(self, uri: str) -> Asset | None: ...

    def query_assets(self, *, kind=None, scope=None,
                     producer_run=None, producer_task=None,
                     tag=None, limit=None) -> list[Asset]: ...
    def query_runs(self, *, experiment_id=None, status=None) -> list[RunMetadata]: ...

    def rebuild(self) -> RebuildReport:
        """Wipe and rewalk the workspace. Always safe."""
```

### 5.2 Always regenerable

| Catalog section | Source of truth |
|-|-|
| `workspaces` | `workspace.json` at root |
| `projects` | `projects/<id>/project.json` |
| `experiments` | `projects/<id>/experiments/<id>/experiment.json` |
| `runs` | `…/runs/<id>/run.json` |
| `executions` | `run.json:execution_history` + `execution/<exec_id>/workflow.json` |
| `assets` | every scope's `assets.json` |
| `consumes` | each execution's `ExecutionRecord.consumes` |

On schema change, bump `schema_version` and require rebuild. No migrations.

## 6. API Changes

### 6.1 `RunContext`

Old `save_artifact` / `logs_dir` / `checkpoint(name)` are **deleted**. Replacements:

```python
ctx.artifact.save("metrics.json", {"loss": 0.1})     # → ArtifactAsset
ctx.log("run").append("epoch 3 complete")            # → LogAsset
ctx.checkpoint("after_epoch_3", data=state)          # → CheckpointAsset
```

Each accessor writes the file, updates the scope `assets.json`, and upserts the catalog row. `Producer` populated automatically from active task context.

### 6.2 `DataAsset` imports

`AssetLibrary` renamed to `DataAssetLibrary`. `workspace.assets` becomes the catalog view; user imports go through `workspace.data_assets.import_asset(...)`.

### 6.3 Server routes

```text
GET  /api/assets?kind=&scope=&run_id=   Query catalog
GET  /api/assets/{asset_id}             Metadata (discriminated union)
GET  /api/assets/{asset_id}/content     File download
GET  /api/assets/{asset_id}/tail?n=     Last N lines (LogAsset)
GET  /api/assets/{asset_id}/stream      SSE tail (LogAsset)
POST /api/assets/data/import            Upload → DataAsset

GET  /api/runs?experiment_id=&status=   Indexed run query
GET  /api/executions?run_id=            Indexed execution query
```

### 6.4 UI

One `<AssetViewer>` dispatches on the TS discriminated union. Inspector panel collapses Artifacts / Logs / Checkpoints tabs into one filterable Assets tab.

## 7. Lineage

```python
@wf.task(consumes=["asset://project/qm9/<asset_id>"])
async def train(ctx): ...
```

At execution, URIs are resolved to `Asset` snapshots, stored on `ExecutionRecord.consumes`, and inserted into the catalog's `consumes` list. First supported lineage query: `catalog.query_runs(consumed_asset="asset://...")`.

## 8. Invariants and Tests

- **Regenerable catalog**: `rm -rf .catalog && molexp catalog rebuild` produces an equivalent index. Test: populate workspace, snapshot, wipe + rebuild, diff.
- **Run portability**: tar a run directory → extract into another workspace → rebuild → run is queryable and `RunContext.open(run_dir)` succeeds.
- **Manifest–disk consistency**: every manifest entry's path exists; every conventional file has a manifest entry after rebuild. Test: fuzz-create files, rebuild, assert no delta.
- **Subclass dispatch**: reading an asset returns the correct subclass. Test: write one of each kind, query back, assert `isinstance`.
- **Typed accessors**: `ctx.artifact.save` → catalog row exists, `Producer.task_id` set when called inside a task.
- **Concurrent writes within a run**: parallel tasks writing assets in one run process all land in `assets.json` without loss. Test: N parallel tasks × M asset writes, assert count.

## 9. Implementation Order

1. `assets/base.py` + all subclass files + discriminated union.
2. `AssetManifest` — per-scope `assets.json` load/mutate/atomic-write, with process-local lock.
3. `AssetCatalog` — `.catalog/index.json` load/mutate/atomic-write + rebuild walker.
4. Rewire `RunContext`: delete `save_artifact` / `logs_dir` / old `checkpoint`; add `ctx.artifact` / `ctx.log(name)` / `ctx.checkpoint(name, data=...)`.
5. Wire `Workspace` / `Project` / `Experiment` / `Run` to call catalog `upsert_*` on materialize.
6. Rewire `RunStorePersistence` to register `ExecutionStateAsset` on each write.
7. Rename `AssetLibrary` → `DataAssetLibrary`; `{scope}.assets` returns catalog view.
8. Tests for each of §8 invariants.
9. Server routes (§6.3) + OpenAPI regen + TS client regen.
10. MSW mock handlers.
11. UI `<AssetViewer>` + inspector panel rework.
12. Delete `ctx.save_artifact`, `ctx.logs_dir`, old checkpoint path from tests and examples; update docs.

## 10. Future Work (Not Blocking)

- Checkpoint branching UI (schema already supports it).
- Large binary backends (multi-GB artifacts via chunked storage / S3).
- Catalog sharding or remote Postgres for multi-user deployments at scale.
