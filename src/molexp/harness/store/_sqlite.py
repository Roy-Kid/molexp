"""Shared SQLite bootstrap for ``SQLiteEventLog`` and ``SQLiteProvenanceStore``.

Private to ``molexp.harness.store`` — not part of the public surface. Both
SQLite-backed stores share the same database file per run so a single
``open_db`` call yields a connection ready to use for either table set.

Pragmas:
- ``journal_mode=WAL`` — concurrent reads while a writer is appending.
- ``foreign_keys=ON`` — needed for the ``artifact_edges`` cross-reference.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

__all__ = ["SCHEMA_VERSION", "open_db"]


SCHEMA_VERSION = 1


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
    PRIMARY KEY (parent_id, child_id, relation)
);

CREATE INDEX IF NOT EXISTS idx_edges_parent ON artifact_edges(parent_id);
CREATE INDEX IF NOT EXISTS idx_edges_child ON artifact_edges(child_id);
"""


def open_db(path: Path) -> sqlite3.Connection:
    """Open or create the harness's SQLite database at ``path``.

    Creates the parent directory if needed; enables WAL + foreign-keys;
    bootstraps the schema and records :data:`SCHEMA_VERSION` on first open.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, isolation_level=None)  # autocommit; we BEGIN explicitly
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(_SCHEMA_SQL)
    # Record the schema version on first open. INSERT OR IGNORE avoids a
    # PRIMARY KEY race when two processes open a fresh DB concurrently.
    conn.execute("INSERT OR IGNORE INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
    return conn
