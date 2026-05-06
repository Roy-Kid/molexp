"""Workspace-level JSON catalog.

One file: ``<workspace_root>/.catalog/index.json``.  Sections:

    workspaces  projects  experiments  runs  executions  assets  consumes

All sections are derived from filesystem state.  ``rebuild()`` wipes
and rewalks.  Mutations use load → edit → atomic-rename with a
process-local lock.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._adapter import ASSET_ADAPTER, parse_asset
from .base import Asset, AssetScope
from .manifest import MANIFEST_FILENAME, AssetManifest

CATALOG_SCHEMA_VERSION = 1
CATALOG_DIRNAME = ".catalog"
CATALOG_FILENAME = "index.json"

_EMPTY_CATALOG: dict[str, Any] = {
    "schema_version": CATALOG_SCHEMA_VERSION,
    "workspaces": {},
    "projects": {},
    "experiments": {},
    "runs": {},
    "executions": {},
    "assets": {},
    "consumes": [],
}


@dataclass
class RebuildReport:
    """Summary of a catalog rebuild."""

    workspaces: int = 0
    projects: int = 0
    experiments: int = 0
    runs: int = 0
    executions: int = 0
    assets: int = 0
    errors: list[str] = field(default_factory=list)


class AssetCatalog:
    """Workspace-wide JSON index."""

    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.dir = self.workspace_root / CATALOG_DIRNAME
        self.path = self.dir / CATALOG_FILENAME
        self._lock = threading.Lock()

    # ── Asset operations ─────────────────────────────────────────────────

    def register(self, asset: Asset) -> None:
        """Insert or overwrite an asset row."""
        with self._lock:
            data = self._load()
            data["assets"][asset.asset_id] = _dump_asset(asset)
            self._save(data)

    def update(self, asset: Asset) -> None:
        self.register(asset)

    def deregister_asset(self, asset_id: str) -> None:
        with self._lock:
            data = self._load()
            data["assets"].pop(asset_id, None)
            self._save(data)

    def get(self, asset_id: str) -> Asset | None:
        data = self._load()
        entry = data["assets"].get(asset_id)
        return parse_asset(entry) if entry else None

    def resolve(self, uri: str) -> Asset | None:
        """Resolve ``asset://.../<asset_id>`` to the stored asset."""
        if not uri.startswith("asset://"):
            return None
        asset_id = uri.rsplit("/", 1)[-1]
        return self.get(asset_id)

    def query_assets(
        self,
        *,
        kind: str | type[Asset] | None = None,
        scope: AssetScope | None = None,
        producer_run: str | None = None,
        producer_task: str | None = None,
        tag: tuple[str, str] | None = None,
        limit: int | None = None,
        recursive: bool = False,
    ) -> list[Asset]:
        """Query assets matching the given filters.

        When ``recursive`` is ``True`` and ``scope`` is given, the scope
        match also includes any sub-scope whose ids start with the given
        scope's ids — i.e. an experiment scope sees all assets in its
        runs, a project scope sees all assets in its experiments + runs.
        Default (``recursive=False``) preserves the historic exact-scope
        match.
        """
        data = self._load()
        kind_str = _kind_value(kind)
        out: list[Asset] = []
        for entry in data["assets"].values():
            if kind_str and entry.get("kind") != kind_str:
                continue
            if scope is not None and not _scope_matches(entry, scope, recursive):
                continue
            if producer_run:
                producer = entry.get("producer") or {}
                if producer.get("run_id") != producer_run:
                    continue
            if producer_task:
                producer = entry.get("producer") or {}
                if producer.get("task_id") != producer_task:
                    continue
            if tag is not None:
                tk, tv = tag
                if entry.get("tags", {}).get(tk) != tv:
                    continue
            out.append(parse_asset(entry))
            if limit is not None and len(out) >= limit:
                break
        return out

    # ── Scope upserts ────────────────────────────────────────────────────

    def upsert_workspace(self, record: dict) -> None:
        self._upsert("workspaces", record["workspace_id"], record)

    def upsert_project(self, record: dict) -> None:
        self._upsert("projects", record["project_id"], record)

    def upsert_experiment(self, record: dict) -> None:
        self._upsert("experiments", record["experiment_id"], record)

    def upsert_run(self, record: dict) -> None:
        self._upsert("runs", record["run_id"], record)

    def upsert_execution(self, record: dict) -> None:
        self._upsert("executions", record["execution_id"], record)

    def _upsert(self, section: str, key: str, record: dict) -> None:
        with self._lock:
            data = self._load()
            data[section][key] = record
            self._save(data)

    # ── Scope removals (cascade) ─────────────────────────────────────────

    def remove_project(self, project_id: str) -> None:
        """Drop a project and everything scoped under it."""
        with self._lock:
            data = self._load()
            data["projects"].pop(project_id, None)
            exp_ids = {
                eid for eid, e in data["experiments"].items() if e.get("project_id") == project_id
            }
            for eid in exp_ids:
                data["experiments"].pop(eid, None)
            run_ids = {rid for rid, r in data["runs"].items() if r.get("experiment_id") in exp_ids}
            for rid in run_ids:
                data["runs"].pop(rid, None)
            data["executions"] = {
                xid: x for xid, x in data["executions"].items() if x.get("run_id") not in run_ids
            }
            data["consumes"] = [
                edge for edge in data["consumes"] if edge.get("execution_id") in data["executions"]
            ]
            self._save(data)

    def remove_experiment(self, experiment_id: str) -> None:
        """Drop an experiment and its runs / executions."""
        with self._lock:
            data = self._load()
            data["experiments"].pop(experiment_id, None)
            run_ids = {
                rid for rid, r in data["runs"].items() if r.get("experiment_id") == experiment_id
            }
            for rid in run_ids:
                data["runs"].pop(rid, None)
            data["executions"] = {
                xid: x for xid, x in data["executions"].items() if x.get("run_id") not in run_ids
            }
            data["consumes"] = [
                edge for edge in data["consumes"] if edge.get("execution_id") in data["executions"]
            ]
            self._save(data)

    def remove_run(self, run_id: str) -> None:
        """Drop a run and its executions."""
        with self._lock:
            data = self._load()
            data["runs"].pop(run_id, None)
            data["executions"] = {
                xid: x for xid, x in data["executions"].items() if x.get("run_id") != run_id
            }
            data["consumes"] = [
                edge for edge in data["consumes"] if edge.get("execution_id") in data["executions"]
            ]
            self._save(data)

    def remove_execution(self, execution_id: str) -> None:
        """Drop a single execution row."""
        with self._lock:
            data = self._load()
            data["executions"].pop(execution_id, None)
            data["consumes"] = [
                edge for edge in data["consumes"] if edge.get("execution_id") != execution_id
            ]
            self._save(data)

    # ── Scope queries ────────────────────────────────────────────────────

    def query_runs(
        self,
        *,
        experiment_id: str | None = None,
        status: str | None = None,
        limit: int | None = None,
    ) -> list[dict]:
        data = self._load()
        out: list[dict] = []
        for entry in data["runs"].values():
            if experiment_id and entry.get("experiment_id") != experiment_id:
                continue
            if status and entry.get("status") != status:
                continue
            out.append(entry)
            if limit is not None and len(out) >= limit:
                break
        return out

    def query_executions(
        self,
        *,
        run_id: str | None = None,
        limit: int | None = None,
    ) -> list[dict]:
        data = self._load()
        out: list[dict] = []
        for entry in data["executions"].values():
            if run_id and entry.get("run_id") != run_id:
                continue
            out.append(entry)
            if limit is not None and len(out) >= limit:
                break
        return out

    # ── Lineage ──────────────────────────────────────────────────────────

    def record_consumes(self, execution_id: str, task_id: str, asset_id: str) -> None:
        with self._lock:
            data = self._load()
            edge = {
                "execution_id": execution_id,
                "task_id": task_id,
                "asset_id": asset_id,
            }
            if edge not in data["consumes"]:
                data["consumes"].append(edge)
            self._save(data)

    # ── Rebuild ──────────────────────────────────────────────────────────

    def rebuild(self) -> RebuildReport:
        """Drop the index and rewalk the workspace from on-disk truth."""
        report = RebuildReport()
        fresh = json.loads(json.dumps(_EMPTY_CATALOG))  # deep copy

        # Workspace
        ws_json = self.workspace_root / "workspace.json"
        if ws_json.exists():
            raw = _read_json(ws_json) or {}
            wid = raw.get("id")
            if wid:
                fresh["workspaces"][wid] = {
                    "workspace_id": wid,
                    "root_path": str(self.workspace_root),
                    "name": raw.get("name", wid),
                    "created_at": raw.get("created_at"),
                    "updated_at": raw.get("created_at"),
                }
                report.workspaces += 1

        # Projects, experiments, runs
        projects_dir = self.workspace_root / "projects"
        if projects_dir.exists():
            for proj_dir in projects_dir.iterdir():
                if not proj_dir.is_dir():
                    continue
                proj_record = _read_json(proj_dir / "project.json")
                if proj_record is None:
                    continue
                pid = proj_record.get("id")
                if not pid:
                    continue
                fresh["projects"][pid] = {
                    "project_id": pid,
                    "workspace_id": fresh["workspaces"] and next(iter(fresh["workspaces"])),
                    "path": str(proj_dir.relative_to(self.workspace_root)),
                    **proj_record,
                }
                report.projects += 1

                experiments_dir = proj_dir / "experiments"
                if experiments_dir.exists():
                    for exp_dir in experiments_dir.iterdir():
                        if not exp_dir.is_dir():
                            continue
                        exp_record = _read_json(exp_dir / "experiment.json")
                        if exp_record is None:
                            continue
                        eid = exp_record.get("id")
                        if not eid:
                            continue
                        fresh["experiments"][eid] = {
                            "experiment_id": eid,
                            "project_id": pid,
                            "path": str(exp_dir.relative_to(self.workspace_root)),
                            **exp_record,
                        }
                        report.experiments += 1

                        runs_dir = exp_dir / "runs"
                        if runs_dir.exists():
                            for run_dir in runs_dir.iterdir():
                                if not run_dir.is_dir():
                                    continue
                                run_record = _read_json(run_dir / "run.json")
                                if run_record is None:
                                    continue
                                rid = run_record.get("id")
                                if not rid:
                                    continue
                                fresh["runs"][rid] = {
                                    "run_id": rid,
                                    "experiment_id": eid,
                                    "path": str(run_dir.relative_to(self.workspace_root)),
                                    **{k: v for k, v in run_record.items() if k != "context"},
                                }
                                report.runs += 1

                                # Executions from history
                                for exec_record in run_record.get("execution_history", []):
                                    xid = exec_record.get("execution_id")
                                    if not xid:
                                        continue
                                    fresh["executions"][xid] = {
                                        "execution_id": xid,
                                        "run_id": rid,
                                        **exec_record,
                                    }
                                    report.executions += 1

        # Assets — scan every scope's manifest that exists
        for manifest_path in _iter_manifest_paths(self.workspace_root):
            if not manifest_path.exists():
                continue
            try:
                with open(manifest_path) as fh:
                    data = json.load(fh)
                for entry in (data.get("assets") or {}).values():
                    asset = parse_asset(entry)
                    fresh["assets"][asset.asset_id] = _dump_asset(asset)
                    report.assets += 1
            except (json.JSONDecodeError, OSError, ValueError) as exc:
                report.errors.append(f"{manifest_path}: {exc}")

        with self._lock:
            self._save(fresh)

        return report

    # ── Internal I/O ─────────────────────────────────────────────────────

    def _load(self) -> dict:
        if not self.path.exists():
            return json.loads(json.dumps(_EMPTY_CATALOG))
        with open(self.path) as fh:
            data = json.load(fh)
        if data.get("schema_version") != CATALOG_SCHEMA_VERSION:
            # Schema mismatch → rebuild is the answer; return empty for now,
            # callers should invoke rebuild() explicitly.
            return json.loads(json.dumps(_EMPTY_CATALOG))
        # Ensure all expected sections exist (for older files)
        for key, default in _EMPTY_CATALOG.items():
            if key == "schema_version":
                continue
            data.setdefault(key, json.loads(json.dumps(default)))
        return data

    def _save(self, data: dict) -> None:
        from ..base import _atomic_write_json

        self.dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(self.path, data)


# ── Helpers ──────────────────────────────────────────────────────────────


def _dump_asset(asset: Asset) -> dict:
    return ASSET_ADAPTER.dump_python(asset, mode="json")


_SCOPE_KIND_RANK: dict[str, int] = {
    "workspace": 0,
    "project": 1,
    "experiment": 2,
    "run": 3,
}


def _scope_matches(entry: dict, scope: AssetScope, recursive: bool) -> bool:
    """Return True if *entry*'s recorded scope satisfies the query scope.

    Default (``recursive=False``) is the historic exact-scope behaviour.
    With ``recursive=True``, an entry whose scope is *underneath* the
    queried scope also matches — i.e. its ids start with the queried
    ids and its kind is at the same or deeper level in the hierarchy.
    """
    entry_scope = entry.get("scope") or {}
    entry_kind = entry_scope.get("kind")
    entry_ids = tuple(entry_scope.get("ids", ()))

    if not recursive:
        return entry_kind == scope.kind and entry_ids == scope.ids

    if entry_kind not in _SCOPE_KIND_RANK or scope.kind not in _SCOPE_KIND_RANK:
        return False
    if _SCOPE_KIND_RANK[entry_kind] < _SCOPE_KIND_RANK[scope.kind]:
        return False
    if len(entry_ids) < len(scope.ids):
        return False
    return entry_ids[: len(scope.ids)] == scope.ids


def _kind_value(kind: str | type[Asset] | None) -> str | None:
    if kind is None:
        return None
    if isinstance(kind, str):
        return kind
    # It's a subclass of Asset; pull the Literal default
    try:
        default = kind.model_fields["kind"].default  # type: ignore[attr-defined]
    except (AttributeError, KeyError):
        return None
    return default


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None


def _iter_manifest_paths(workspace_root: Path):
    """Yield every scope's assets.json path (workspace, projects, experiments, runs)."""
    yield workspace_root / MANIFEST_FILENAME
    projects_dir = workspace_root / "projects"
    if not projects_dir.exists():
        return
    for proj_dir in projects_dir.iterdir():
        if not proj_dir.is_dir():
            continue
        yield proj_dir / MANIFEST_FILENAME
        experiments_dir = proj_dir / "experiments"
        if not experiments_dir.exists():
            continue
        for exp_dir in experiments_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            yield exp_dir / MANIFEST_FILENAME
            runs_dir = exp_dir / "runs"
            if not runs_dir.exists():
                continue
            for run_dir in runs_dir.iterdir():
                if not run_dir.is_dir():
                    continue
                yield run_dir / MANIFEST_FILENAME


# Also expose AssetManifest here for convenience when scope entities
# want to construct their own local manifest:
__all__ = ["AssetCatalog", "AssetManifest", "CATALOG_FILENAME", "RebuildReport"]
