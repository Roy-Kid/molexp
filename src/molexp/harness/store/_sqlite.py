"""Harness-specific SQLite bootstrap for ``SQLiteEventLog`` + ``SQLiteArtifactLineageStore``.

Private to ``molexp.harness.store``. The generic connection infrastructure — WAL
pragmas + the path-keyed thread-lock registry + ``check_same_thread=False`` — now
lives in the Layer-0 :mod:`molexp.sqlitelog` primitive
(:func:`~molexp.sqlitelog.open_wal_connection`), which this module delegates to so
``harness.store`` and ``molexp.workspace.events`` share one implementation
(integration.md §2.1). :func:`open_db` layers only the harness-specific
``artifact_edges`` lineage table + schema versioning on top; the ``events`` seq-log
table is created by the :class:`~molexp.sqlitelog.SeqEventStore` inside
``SQLiteEventLog``.

Both SQLite-backed stores (event log + lineage store) share one DB file per run, so
a single :func:`open_db` call yields a connection ready for either table set; the
shared per-file lock (from ``open_wal_connection``) serializes the cross-store
``events`` / ``artifact_edges`` writes.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from threading import Lock

from molexp.sqlitelog import open_wal_connection

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

# The ``events`` seq-log table is owned by ``SeqEventStore`` (Layer-0); this
# module only bootstraps the harness-specific lineage table + schema version.
_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);

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


def open_db(path: Path) -> tuple[sqlite3.Connection, Lock]:
    """Open or create the harness's SQLite database at ``path``.

    Delegates the WAL connection + path-keyed shared lock to
    :func:`molexp.sqlitelog.open_wal_connection`, then bootstraps the
    harness-specific ``artifact_edges`` table + records :data:`SCHEMA_VERSION`.
    The ``events`` table is created by ``SQLiteEventLog``'s ``SeqEventStore``.

    Args:
        path: The SQLite database file path.

    Returns:
        A ``(connection, lock)`` pair; the lock is the shared per-file lock and
        MUST guard every use of the connection (the thread-safety contract).
    """
    conn, lock = open_wal_connection(path)
    conn.executescript(_SCHEMA_SQL)
    _migrate_artifact_edges(conn)
    # INSERT OR IGNORE avoids a PRIMARY KEY race when two processes open a fresh
    # DB concurrently.
    conn.execute("INSERT OR IGNORE INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
    return conn, lock


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
