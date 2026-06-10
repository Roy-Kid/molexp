"""Shared SQLite bootstrap for ``SQLiteEventLog`` and ``SQLiteArtifactLineageStore``.

Private to ``molexp.harness.store`` — not part of the public surface. Both
SQLite-backed stores share the same database file per run so a single
``open_db`` call yields a connection ready to use for either table set.

Pragmas:
- ``journal_mode=WAL`` — concurrent reads while a writer is appending.
- ``synchronous=NORMAL`` — safe under WAL; drops the per-commit fsync
  stall that otherwise serializes every event-log append.
- ``busy_timeout=5000`` — wait up to 5 s for a competing writer instead
  of raising ``SQLITE_BUSY`` immediately.
- ``foreign_keys=ON`` — needed for the ``artifact_edges`` cross-reference.

Thread-safety (design A — see spec ``perf-hardening-01-async-io-offload``).
``StageRunner`` offloads the stores' blocking writes onto worker threads via
:func:`asyncio.to_thread`, so the same :class:`sqlite3.Connection` can be
touched from several threads. To make that safe, :func:`open_db` opens the
connection with ``check_same_thread=False`` and returns a
:class:`threading.Lock` alongside it; every store serializes all connection
access behind that lock. ``SQLiteEventLog`` and ``SQLiteArtifactLineageStore``
share one DB file per run — they each call :func:`open_db` separately, so the
lock is keyed on the **resolved DB path** in a module-level registry: two
``open_db`` calls on the same file return the *same* lock instance, which is
what serializes the cross-store ``events`` / ``artifact_edges`` writes.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

__all__ = ["SCHEMA_VERSION", "open_db"]


SCHEMA_VERSION = 2
"""Current schema version.

History:

- v1 — ``events`` + bare ``artifact_edges`` (parent/child/relation/created_at).
- v2 — ``artifact_edges`` gains nullable ``stage`` + ``run_id`` columns so a
  lineage edge records which pipeline stage of which run derived the child.
  v1 databases are migrated in place by :func:`_migrate_artifact_edges`
  (pre-existing rows read back with ``NULL`` in the new columns).

``schema_version`` keeps one row per version ever applied (``INSERT OR
IGNORE``); the effective version is ``MAX(version)``.
"""

# Path-keyed registry of per-DB-file locks. Two stores opening the same file
# (the event log + lineage store of one run) must serialize through the
# SAME lock, but they construct independent connections — so the lock cannot
# live on the connection; it is shared by resolved path here. ``_registry_guard``
# protects insertion into ``_locks`` itself.
_locks: dict[str, threading.Lock] = {}
_registry_guard = threading.Lock()


def _lock_for(path: Path) -> threading.Lock:
    """Return the shared :class:`threading.Lock` for ``path`` (one per DB file)."""
    key = str(path.resolve())
    with _registry_guard:
        lock = _locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _locks[key] = lock
        return lock


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    type TEXT NOT NULL,
    actor TEXT NOT NULL,
    created_at TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    artifact_ids_json TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_events_run_seq ON events(run_id, seq);

CREATE TABLE IF NOT EXISTS artifact_edges (
    parent_id TEXT NOT NULL,
    child_id TEXT NOT NULL,
    relation TEXT NOT NULL,
    created_at TEXT NOT NULL,
    stage TEXT,
    run_id TEXT,
    PRIMARY KEY (parent_id, child_id, relation)
);

CREATE INDEX IF NOT EXISTS idx_edges_parent ON artifact_edges(parent_id);
CREATE INDEX IF NOT EXISTS idx_edges_child ON artifact_edges(child_id);
"""


def open_db(path: Path) -> tuple[sqlite3.Connection, threading.Lock]:
    """Open or create the harness's SQLite database at ``path``.

    Creates the parent directory if needed; enables WAL + foreign-keys;
    bootstraps the schema and records :data:`SCHEMA_VERSION` on first open.

    Args:
        path: The SQLite database file path.

    Returns:
        A ``(connection, lock)`` pair. The connection is opened with
        ``check_same_thread=False`` so it can be used from
        :func:`asyncio.to_thread` workers; the lock is the shared per-file
        lock from :func:`_lock_for` and MUST guard every use of the
        connection (see the module docstring's thread-safety contract).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        path,
        isolation_level=None,  # autocommit; we BEGIN explicitly
        check_same_thread=False,  # connection is serialized by the shared lock
    )
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(_SCHEMA_SQL)
    _migrate_artifact_edges(conn)
    # Record the schema version on open. INSERT OR IGNORE avoids a
    # PRIMARY KEY race when two processes open a fresh DB concurrently.
    conn.execute("INSERT OR IGNORE INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
    return conn, _lock_for(path)


def _migrate_artifact_edges(conn: sqlite3.Connection) -> None:
    """Bring a v1 ``artifact_edges`` table up to the v2 column set.

    ``CREATE TABLE IF NOT EXISTS`` is a no-op on an existing v1 table, so the
    ``stage`` / ``run_id`` columns are added here with ``ALTER TABLE``.
    Idempotent: columns already present are left untouched.
    """
    existing = {row[1] for row in conn.execute("PRAGMA table_info(artifact_edges)")}
    for column in ("stage", "run_id"):
        if column not in existing:
            conn.execute(f"ALTER TABLE artifact_edges ADD COLUMN {column} TEXT")
