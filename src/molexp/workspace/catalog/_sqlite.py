"""SQLite bootstrap for the workspace-derived asset catalog.

Private to ``molexp.workspace.catalog`` — not part of the public surface.
Mirrors the *paradigm* of ``harness.store._sqlite`` (WAL, ``busy_timeout``,
a path-keyed connection lock) but stays inside the workspace layer: the
workspace MUST NOT import anything from ``harness``.

Why SQLite, not a single ``index.json``:
- The catalog is **derived** and written from many call sites — every
  ``register`` / ``upsert_*`` / ``remove_*``. The old design loaded the
  whole file, mutated a dict, and atomic-renamed it back. Two OS processes
  doing that concurrently both read the same base and the last writer wins,
  silently dropping the other's rows.
- A row-level ``INSERT OR REPLACE`` under WAL serializes writers at the DB
  level *across processes*, so concurrent registration loses no rows, and
  each write is ~O(log A) instead of O(A) whole-file rewrite.

Pragmas (same rationale as the harness store):
- ``journal_mode=WAL`` — concurrent readers while a writer appends.
- ``synchronous=NORMAL`` — safe under WAL; drops the per-commit fsync stall.
- ``busy_timeout=5000`` — wait up to 5 s for a competing writer instead of
  raising ``SQLITE_BUSY`` immediately.

Thread-safety. ``ws.catalog`` is a per-process singleton holding one
connection, but the same DB file may also be reached from a freshly
constructed ``AssetCatalog`` (e.g. a subprocess, or a second instance).
:func:`open_catalog_db` opens with ``check_same_thread=False`` and returns a
``threading.Lock`` keyed on the resolved DB path, so every connection to the
same file serializes through one lock — the cross-instance contract the
caller MUST honour by guarding all connection access behind that lock.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

__all__ = ["CATALOG_DB_FILENAME", "SCHEMA_VERSION", "open_catalog_db"]

SCHEMA_VERSION = 1
CATALOG_DB_FILENAME = "index.sqlite"

# Path-keyed registry of per-DB-file locks: two ``AssetCatalog`` instances
# pointing at the same file must serialize through the SAME lock even though
# they hold independent connections, so the lock cannot live on a connection.
_locks: dict[str, threading.Lock] = {}
_registry_guard = threading.Lock()


def _lock_for(path: Path) -> threading.Lock:
    key = str(path.resolve())
    with _registry_guard:
        lock = _locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _locks[key] = lock
        return lock


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY);

CREATE TABLE IF NOT EXISTS workspaces (
    workspace_id TEXT PRIMARY KEY,
    json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS projects (
    project_id TEXT PRIMARY KEY,
    workspace_id TEXT,
    json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    project_id TEXT,
    json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    experiment_id TEXT,
    status TEXT,
    json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS executions (
    execution_id TEXT PRIMARY KEY,
    run_id TEXT,
    json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS assets (
    asset_id TEXT PRIMARY KEY,
    kind TEXT,
    content_hash TEXT,
    scope_kind TEXT,
    scope_rank INTEGER,
    scope_ids TEXT,
    producer_run TEXT,
    producer_task TEXT,
    json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_runs_experiment ON runs(experiment_id);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
CREATE INDEX IF NOT EXISTS idx_exec_run ON executions(run_id);
CREATE INDEX IF NOT EXISTS idx_assets_content ON assets(content_hash);
CREATE INDEX IF NOT EXISTS idx_assets_kind ON assets(kind);
CREATE INDEX IF NOT EXISTS idx_assets_scope ON assets(scope_kind, scope_ids);
CREATE INDEX IF NOT EXISTS idx_assets_producer_run ON assets(producer_run);
"""


def open_catalog_db(path: Path) -> tuple[sqlite3.Connection, threading.Lock]:
    """Open or create the catalog SQLite DB at ``path``.

    Creates the parent directory; enables WAL + ``busy_timeout``; bootstraps
    the schema and records :data:`SCHEMA_VERSION` on first open.

    Returns a ``(connection, lock)`` pair; the lock is the shared per-file
    lock from :func:`_lock_for` and MUST guard every use of the connection.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        path,
        isolation_level=None,  # autocommit; we BEGIN explicitly for batches
        check_same_thread=False,  # serialized by the shared lock
    )
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.executescript(_SCHEMA_SQL)
    conn.execute("INSERT OR IGNORE INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
    return conn, _lock_for(path)
