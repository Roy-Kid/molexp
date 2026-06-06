"""Workspace-level derived asset catalog (SQLite backend).

One database: ``<workspace_root>/catalog/index.sqlite`` with six tables —
``workspaces  projects  experiments  runs  executions  assets``.

Every section is **derived** from entity ``*.json`` + asset manifests, which
remain the single source of truth; :meth:`AssetCatalog.rebuild` wipes the DB
and rewalks the tree. Mutations are row-level ``INSERT OR REPLACE`` / ``DELETE``
under WAL, so concurrent multi-process writers do not lose rows (the legacy
load-mutate-atomic-rename ``index.json`` did) and each write is ~O(log A)
instead of an O(A) whole-file rewrite. See :mod:`._sqlite` for the rationale.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ..assets._adapter import ASSET_ADAPTER, parse_asset
from ..assets.base import Asset, AssetScope, Producer
from ..assets.manifest import MANIFEST_FILENAME, AssetManifest
from ..utils import generate_asset_id
from ._sqlite import CATALOG_DB_FILENAME, open_catalog_db

CATALOG_SCHEMA_VERSION = 3
CATALOG_DIRNAME = "catalog"
CATALOG_FILENAME = CATALOG_DB_FILENAME

_SCOPE_KIND_RANK: dict[str, int] = {
    "workspace": 0,
    "project": 1,
    "experiment": 2,
    "run": 3,
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
    """Workspace-wide derived index, backed by SQLite."""

    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.dir = self.workspace_root / CATALOG_DIRNAME
        self.path = self.dir / CATALOG_DB_FILENAME
        self._connection: sqlite3.Connection | None = None
        self._db_lock: threading.Lock | None = None

    # ── Connection (lazy; no I/O in __init__) ────────────────────────────

    def _conn(self) -> tuple[sqlite3.Connection, threading.Lock]:
        if self._connection is None or self._db_lock is None:
            self._connection, self._db_lock = open_catalog_db(self.path)
        return self._connection, self._db_lock

    def _write(self, sql: str, params: tuple) -> None:
        conn, lock = self._conn()
        with lock:
            conn.execute(sql, params)

    def _read(self, sql: str, params: tuple = ()) -> list[tuple]:
        conn, lock = self._conn()
        with lock:
            return conn.execute(sql, params).fetchall()

    @contextmanager
    def _txn(self) -> Iterator[sqlite3.Connection]:
        conn, lock = self._conn()
        with lock:
            conn.execute("BEGIN")
            try:
                yield conn
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise

    # ── Asset operations ─────────────────────────────────────────────────

    def register(self, asset: Asset) -> None:
        """Insert or overwrite an asset row."""
        self._write(_ASSET_UPSERT_SQL, _asset_row(_dump_asset(asset)))

    def update(self, asset: Asset) -> None:
        self.register(asset)

    def find_by_content_hash(self, content_hash: str) -> Asset | None:
        """Return any catalogued asset whose ``content_hash`` matches, else ``None``.

        Content-addressed lookup used by the workflow cache's re-registration
        path: the bytes are guaranteed present iff some asset row already
        points at them (no row → the store does not hold the artifact, so the
        caller skips it gracefully).
        """
        if not content_hash:
            return None
        rows = self._read(
            "SELECT json FROM assets WHERE content_hash = ? ORDER BY rowid LIMIT 1",
            (content_hash,),
        )
        return parse_asset(json.loads(rows[0][0])) if rows else None

    def reregister_artifact(
        self,
        *,
        name: str,
        content_hash: str,
        target_scope: AssetScope,
        producer_task: str | None = None,
    ) -> Asset | None:
        """Idempotently re-register a content-addressed artifact into a new scope.

        Looks the artifact up by ``content_hash`` (the bytes must already be
        catalogued by an earlier run); if found, inserts a fresh artifact row
        bound to *target_scope* that points at the SAME on-disk path and the
        SAME ``content_hash`` — no recompute, no byte recopy. Returns the new
        asset, or ``None`` when the bytes are absent (different / fresh
        workspace) so the caller can skip gracefully.

        Idempotent: when *target_scope* already holds an artifact with this
        ``(name, content_hash)`` the existing row is returned unchanged.
        """
        source = self.find_by_content_hash(content_hash)
        if source is None:
            return None

        scope_ids = "/".join(target_scope.ids)
        existing = self._read(
            "SELECT json FROM assets WHERE kind = 'artifact' AND content_hash = ? "
            "AND scope_kind = ? AND scope_ids = ?",
            (content_hash, target_scope.kind, scope_ids),
        )
        for (raw,) in existing:
            if json.loads(raw).get("name") == name:
                return parse_asset(json.loads(raw))

        now = datetime.now()
        clone = source.model_copy(
            update={
                "asset_id": generate_asset_id(),
                "name": name,
                "scope": target_scope,
                "created_at": now,
                "updated_at": now,
                "producer": Producer(
                    run_id=target_scope.ids[-1] if target_scope.ids else None,
                    task_id=producer_task,
                ),
            }
        )
        self._write(_ASSET_UPSERT_SQL, _asset_row(_dump_asset(clone)))
        return clone

    def deregister_asset(self, asset_id: str) -> None:
        self._write("DELETE FROM assets WHERE asset_id = ?", (asset_id,))

    def get(self, asset_id: str) -> Asset | None:
        rows = self._read("SELECT json FROM assets WHERE asset_id = ?", (asset_id,))
        return parse_asset(json.loads(rows[0][0])) if rows else None

    def resolve(self, uri: str) -> Asset | None:
        """Resolve ``asset://.../<asset_id>`` to the stored asset."""
        if not uri.startswith("asset://"):
            return None
        return self.get(uri.rsplit("/", 1)[-1])

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

        When ``recursive`` is ``True`` and ``scope`` is given, the scope match
        also includes any sub-scope whose ids start with the given scope's ids
        — an experiment scope sees all assets in its runs, a project scope sees
        all assets in its experiments + runs. Default (``recursive=False``)
        preserves the historic exact-scope match.
        """
        clauses: list[str] = []
        params: list[object] = []

        kind_str = _kind_value(kind)
        if kind_str:
            clauses.append("kind = ?")
            params.append(kind_str)
        if scope is not None:
            _append_scope_clause(clauses, params, scope, recursive)
        if producer_run:
            clauses.append("producer_run = ?")
            params.append(producer_run)
        if producer_task:
            clauses.append("producer_task = ?")
            params.append(producer_task)

        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = self._read(f"SELECT json FROM assets{where} ORDER BY rowid", tuple(params))

        out: list[Asset] = []
        for (raw,) in rows:
            entry = json.loads(raw)
            if tag is not None:
                tk, tv = tag
                if (entry.get("tags") or {}).get(tk) != tv:
                    continue
            out.append(parse_asset(entry))
            if limit is not None and len(out) >= limit:
                break
        return out

    # ── Scope upserts ────────────────────────────────────────────────────

    def upsert_workspace(self, record: dict) -> None:
        self._write(
            "INSERT OR REPLACE INTO workspaces (workspace_id, json) VALUES (?, ?)",
            (record["workspace_id"], _dumps(record)),
        )

    def upsert_project(self, record: dict) -> None:
        self._write(
            "INSERT OR REPLACE INTO projects (project_id, workspace_id, json) VALUES (?, ?, ?)",
            (record["project_id"], record.get("workspace_id"), _dumps(record)),
        )

    def upsert_experiment(self, record: dict) -> None:
        self._write(
            "INSERT OR REPLACE INTO experiments (experiment_id, project_id, json) VALUES (?, ?, ?)",
            (record["experiment_id"], record.get("project_id"), _dumps(record)),
        )

    def upsert_run(self, record: dict) -> None:
        self._write(_RUN_UPSERT_SQL, _run_row(record))

    def upsert_execution(self, record: dict) -> None:
        self._write(_EXEC_UPSERT_SQL, _exec_row(record))

    def upsert_run_with_executions(self, run_record: dict, execution_records: list[dict]) -> None:
        """Upsert a run row plus all its executions in one transaction.

        Batches what was N+1 separate whole-file rewrites in the legacy
        backend into a single SQLite transaction — the run row and only the
        executions handed in (the caller passes the current history).
        """
        with self._txn() as conn:
            conn.execute(_RUN_UPSERT_SQL, _run_row(run_record))
            for rec in execution_records:
                conn.execute(_EXEC_UPSERT_SQL, _exec_row(rec))

    # ── Scope removals (cascade) ─────────────────────────────────────────

    def remove_project(self, project_id: str) -> None:
        """Drop a project and everything scoped under it."""
        with self._txn() as conn:
            conn.execute(
                "DELETE FROM executions WHERE run_id IN ("
                "  SELECT run_id FROM runs WHERE experiment_id IN ("
                "    SELECT experiment_id FROM experiments WHERE project_id = ?))",
                (project_id,),
            )
            conn.execute(
                "DELETE FROM runs WHERE experiment_id IN ("
                "  SELECT experiment_id FROM experiments WHERE project_id = ?)",
                (project_id,),
            )
            conn.execute("DELETE FROM experiments WHERE project_id = ?", (project_id,))
            conn.execute("DELETE FROM projects WHERE project_id = ?", (project_id,))

    def remove_experiment(self, experiment_id: str) -> None:
        """Drop an experiment and its runs / executions."""
        with self._txn() as conn:
            conn.execute(
                "DELETE FROM executions WHERE run_id IN ("
                "  SELECT run_id FROM runs WHERE experiment_id = ?)",
                (experiment_id,),
            )
            conn.execute("DELETE FROM runs WHERE experiment_id = ?", (experiment_id,))
            conn.execute("DELETE FROM experiments WHERE experiment_id = ?", (experiment_id,))

    def remove_run(self, run_id: str) -> None:
        """Drop a run and its executions."""
        with self._txn() as conn:
            conn.execute("DELETE FROM executions WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))

    def remove_execution(self, execution_id: str) -> None:
        """Drop a single execution row."""
        self._write("DELETE FROM executions WHERE execution_id = ?", (execution_id,))

    # ── Scope queries ────────────────────────────────────────────────────

    def query_runs(
        self,
        *,
        experiment_id: str | None = None,
        status: str | None = None,
        limit: int | None = None,
    ) -> list[dict]:
        clauses: list[str] = []
        params: list[object] = []
        if experiment_id:
            clauses.append("experiment_id = ?")
            params.append(experiment_id)
        if status:
            clauses.append("status = ?")
            params.append(status)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT json FROM runs{where} ORDER BY rowid"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        return [json.loads(raw) for (raw,) in self._read(sql, tuple(params))]

    def query_executions(
        self,
        *,
        run_id: str | None = None,
        limit: int | None = None,
    ) -> list[dict]:
        clauses: list[str] = []
        params: list[object] = []
        if run_id:
            clauses.append("run_id = ?")
            params.append(run_id)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT json FROM executions{where} ORDER BY rowid"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        return [json.loads(raw) for (raw,) in self._read(sql, tuple(params))]

    # ── Rebuild ──────────────────────────────────────────────────────────

    def rebuild(self) -> RebuildReport:
        """Drop the index and rewalk the workspace from on-disk truth.

        Asset rows are copied straight from each manifest's validated dict —
        no ``parse_asset → _dump_asset`` round-trip.
        """
        report = RebuildReport()
        with self._txn() as conn:
            for table in ("workspaces", "projects", "experiments", "runs", "executions", "assets"):
                conn.execute(f"DELETE FROM {table}")

            ws_id = self._rebuild_workspace(conn, report)
            self._rebuild_tree(conn, report, ws_id)
            self._rebuild_assets(conn, report)

        # Drop a stale legacy single-file index left by an older backend.
        legacy = self.dir / "index.json"
        if legacy.exists():
            legacy.unlink()
        return report

    def _rebuild_workspace(self, conn: sqlite3.Connection, report: RebuildReport) -> str | None:
        ws_json = self.workspace_root / "workspace.json"
        raw = _read_json(ws_json) or {}
        wid = raw.get("id")
        if not wid:
            return None
        record = {
            "workspace_id": wid,
            "root_path": str(self.workspace_root),
            "name": raw.get("name", wid),
            "created_at": raw.get("created_at"),
            "updated_at": raw.get("created_at"),
        }
        conn.execute(
            "INSERT OR REPLACE INTO workspaces (workspace_id, json) VALUES (?, ?)",
            (wid, _dumps(record)),
        )
        report.workspaces += 1
        return wid

    def _rebuild_tree(
        self, conn: sqlite3.Connection, report: RebuildReport, ws_id: str | None
    ) -> None:
        projects_dir = self.workspace_root / "projects"
        if not projects_dir.exists():
            return
        for proj_dir in projects_dir.iterdir():
            if not proj_dir.is_dir():
                continue
            proj_record = _read_json(proj_dir / "project.json")
            pid = proj_record.get("id") if proj_record else None
            if proj_record is None or not pid:
                continue
            conn.execute(
                "INSERT OR REPLACE INTO projects (project_id, workspace_id, json) VALUES (?, ?, ?)",
                (
                    pid,
                    ws_id,
                    _dumps(
                        {
                            "project_id": pid,
                            "workspace_id": ws_id,
                            "path": str(proj_dir.relative_to(self.workspace_root)),
                            **proj_record,
                        }
                    ),
                ),
            )
            report.projects += 1
            self._rebuild_experiments(conn, report, proj_dir, pid)

    def _rebuild_experiments(
        self, conn: sqlite3.Connection, report: RebuildReport, proj_dir: Path, pid: str
    ) -> None:
        experiments_dir = proj_dir / "experiments"
        if not experiments_dir.exists():
            return
        for exp_dir in experiments_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            exp_record = _read_json(exp_dir / "experiment.json")
            eid = exp_record.get("id") if exp_record else None
            if exp_record is None or not eid:
                continue
            conn.execute(
                "INSERT OR REPLACE INTO experiments "
                "(experiment_id, project_id, json) VALUES (?, ?, ?)",
                (
                    eid,
                    pid,
                    _dumps(
                        {
                            "experiment_id": eid,
                            "project_id": pid,
                            "path": str(exp_dir.relative_to(self.workspace_root)),
                            **exp_record,
                        }
                    ),
                ),
            )
            report.experiments += 1
            self._rebuild_runs(conn, report, exp_dir, eid)

    def _rebuild_runs(
        self, conn: sqlite3.Connection, report: RebuildReport, exp_dir: Path, eid: str
    ) -> None:
        runs_dir = exp_dir / "runs"
        if not runs_dir.exists():
            return
        for run_dir in runs_dir.iterdir():
            if not run_dir.is_dir():
                continue
            run_record = _read_json(run_dir / "run.json")
            rid = run_record.get("id") if run_record else None
            if run_record is None or not rid:
                continue
            row = {
                "run_id": rid,
                "experiment_id": eid,
                "path": str(run_dir.relative_to(self.workspace_root)),
                **{k: v for k, v in run_record.items() if k != "context"},
            }
            conn.execute(_RUN_UPSERT_SQL, _run_row(row))
            report.runs += 1
            for exec_record in run_record.get("execution_history", []):
                xid = exec_record.get("execution_id")
                if not xid:
                    continue
                conn.execute(_EXEC_UPSERT_SQL, _exec_row({"run_id": rid, **exec_record}))
                report.executions += 1

    def _rebuild_assets(self, conn: sqlite3.Connection, report: RebuildReport) -> None:
        for manifest_path in _iter_manifest_paths(self.workspace_root):
            data = _read_json(manifest_path)
            if data is None:
                continue
            try:
                for entry in (data.get("assets") or {}).values():
                    conn.execute(_ASSET_UPSERT_SQL, _asset_row(entry))
                    report.assets += 1
            except (KeyError, TypeError, ValueError) as exc:
                report.errors.append(f"{manifest_path}: {exc}")


# ── Row builders / SQL ─────────────────────────────────────────────────────

_ASSET_UPSERT_SQL = (
    "INSERT OR REPLACE INTO assets "
    "(asset_id, kind, content_hash, scope_kind, scope_rank, scope_ids, "
    " producer_run, producer_task, json) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
)
_RUN_UPSERT_SQL = (
    "INSERT OR REPLACE INTO runs (run_id, experiment_id, status, json) VALUES (?, ?, ?, ?)"
)
_EXEC_UPSERT_SQL = "INSERT OR REPLACE INTO executions (execution_id, run_id, json) VALUES (?, ?, ?)"


def _dumps(obj: object) -> str:
    """JSON-encode a catalog record, stringifying stragglers (``Path``, …).

    Mirrors the legacy ``atomic_write_json(default=str)`` tolerance: run /
    workspace records may still carry a ``Path`` value.
    """
    return json.dumps(obj, default=str)


def _asset_row(entry: dict) -> tuple:
    """Extract indexed columns + JSON blob from a validated asset dict.

    Used by both ``register`` (asset → ``_dump_asset`` dict) and ``rebuild``
    (manifest entry dict, no pydantic round-trip).
    """
    scope = entry.get("scope") or {}
    scope_kind = scope.get("kind")
    ids = tuple(scope.get("ids", ()))
    producer = entry.get("producer") or {}
    return (
        entry["asset_id"],
        entry.get("kind"),
        entry.get("content_hash"),
        scope_kind,
        _SCOPE_KIND_RANK.get(scope_kind) if scope_kind else None,
        "/".join(ids),
        producer.get("run_id"),
        producer.get("task_id"),
        _dumps(entry),
    )


def _run_row(record: dict) -> tuple:
    return (record["run_id"], record.get("experiment_id"), record.get("status"), _dumps(record))


def _exec_row(record: dict) -> tuple:
    return (record["execution_id"], record.get("run_id"), _dumps(record))


def _append_scope_clause(
    clauses: list[str], params: list[object], scope: AssetScope, recursive: bool
) -> None:
    if not recursive:
        clauses.append("scope_kind = ? AND scope_ids = ?")
        params.extend([scope.kind, "/".join(scope.ids)])
        return
    # Recursive: this scope and any deeper scope whose ids extend these ids.
    rank = _SCOPE_KIND_RANK.get(scope.kind)
    clauses.append("scope_rank IS NOT NULL AND scope_rank >= ?")
    params.append(rank if rank is not None else 0)
    if scope.ids:
        prefix = "/".join(scope.ids)
        clauses.append("(scope_ids = ? OR scope_ids LIKE ?)")
        params.extend([prefix, prefix + "/%"])


def _dump_asset(asset: Asset) -> dict:
    return ASSET_ADAPTER.dump_python(asset, mode="json")


def _kind_value(kind: str | type[Asset] | None) -> str | None:
    if kind is None:
        return None
    if isinstance(kind, str):
        return kind
    try:
        return kind.model_fields["kind"].default  # type: ignore[attr-defined]
    except (AttributeError, KeyError):
        return None


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path) as fh:  # noqa: PTH123
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None


def _iter_manifest_paths(workspace_root: Path) -> Iterator[Path]:
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


# Also expose AssetManifest here for convenience when scope entities want to
# construct their own local manifest:
__all__ = [
    "CATALOG_DIRNAME",
    "CATALOG_FILENAME",
    "CATALOG_SCHEMA_VERSION",
    "AssetCatalog",
    "AssetManifest",
    "RebuildReport",
]
