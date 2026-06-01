"""SQLite implementation of :class:`EventLog`.

Schema lives in :mod:`molexp.harness.store._sqlite`. Per-``run_id`` ``seq``
is assigned inside the same transaction as the insert:

    SELECT COALESCE(MAX(seq), 0) + 1 FROM events WHERE run_id = ?

paired with the ``UNIQUE(run_id, seq)`` index. A duplicate ``(run_id, seq)``
from any source raises :class:`molexp.harness.errors.EventSeqConflictError`
chaining the underlying :class:`sqlite3.IntegrityError`.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from molexp.harness.errors import EventSeqConflictError
from molexp.harness.schemas import EventType, HarnessEvent
from molexp.harness.store._sqlite import open_db

__all__ = ["SQLiteEventLog"]


class SQLiteEventLog:
    """SQLite-backed append-only event log."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        # ``_lock`` is shared per DB file (see ``store._sqlite``); the
        # provenance store on the same path holds the same lock instance so
        # the cross-table writes serialize. All connection access goes
        # through it because ``StageRunner`` calls us from worker threads.
        self._conn, self._lock = open_db(self._path)

    def append(
        self,
        run_id: str,
        type: EventType,
        actor: str,
        payload: dict[str, Any] | None = None,
        artifact_ids: list[str] | None = None,
    ) -> HarnessEvent:
        return self._insert(
            event_id=uuid.uuid4().hex,
            run_id=run_id,
            seq=None,  # autoincrement
            type_=type,
            actor=actor,
            payload=payload or {},
            artifact_ids=list(artifact_ids or []),
        )

    def list_events(self, run_id: str) -> list[HarnessEvent]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, run_id, seq, type, actor, created_at, payload_json, "
                "artifact_ids_json FROM events WHERE run_id = ? ORDER BY seq",
                (run_id,),
            ).fetchall()
        return [self._row_to_event(row) for row in rows]

    def get_timeline(self, run_id: str) -> list[HarnessEvent]:
        # Alias by contract: get_timeline == list_events for a single run.
        return self.list_events(run_id)

    # ----------------------------------------------------------- internals

    def _append_with_explicit_seq(
        self,
        *,
        run_id: str,
        seq: int,
        type: EventType,
        actor: str,
        payload: dict[str, Any] | None = None,
        artifact_ids: list[str] | None = None,
    ) -> HarnessEvent:
        """Test hook: force-insert a row at an explicit ``seq``.

        Production code paths always use ``append()`` (autoincremented
        ``seq``). This entrypoint exists so the test suite can verify the
        ``IntegrityError → EventSeqConflictError`` mapping without
        coordinating two threads.
        """
        return self._insert(
            event_id=uuid.uuid4().hex,
            run_id=run_id,
            seq=seq,
            type_=type,
            actor=actor,
            payload=payload or {},
            artifact_ids=list(artifact_ids or []),
        )

    def _insert(
        self,
        *,
        event_id: str,
        run_id: str,
        seq: int | None,
        type_: EventType,
        actor: str,
        payload: dict[str, Any],
        artifact_ids: list[str],
    ) -> HarnessEvent:
        created_at = datetime.now(tz=UTC)
        payload_json = json.dumps(payload, default=str)
        artifact_ids_json = json.dumps(artifact_ids)
        # Hold the shared lock across the whole SELECT MAX(seq) → INSERT
        # read-modify-write so concurrent appends from to_thread workers
        # cannot interleave and assign the same (run_id, seq).
        with self._lock:
            try:
                self._conn.execute("BEGIN")
                if seq is None:
                    row = self._conn.execute(
                        "SELECT COALESCE(MAX(seq), 0) + 1 FROM events WHERE run_id = ?",
                        (run_id,),
                    ).fetchone()
                    assigned_seq = int(row[0])
                else:
                    assigned_seq = seq
                self._conn.execute(
                    "INSERT INTO events (id, run_id, seq, type, actor, created_at, "
                    "payload_json, artifact_ids_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        event_id,
                        run_id,
                        assigned_seq,
                        type_,
                        actor,
                        created_at.isoformat(),
                        payload_json,
                        artifact_ids_json,
                    ),
                )
                self._conn.execute("COMMIT")
            except sqlite3.IntegrityError as exc:
                self._conn.execute("ROLLBACK")
                raise EventSeqConflictError(
                    f"duplicate (run_id={run_id!r}, seq={seq}) in event log"
                ) from exc

        return HarnessEvent(
            id=event_id,
            run_id=run_id,
            seq=assigned_seq,
            type=type_,
            actor=actor,
            created_at=created_at,
            payload=payload,
            artifact_ids=artifact_ids,
        )

    @staticmethod
    def _row_to_event(row: tuple) -> HarnessEvent:
        (
            event_id,
            run_id,
            seq,
            type_,
            actor,
            created_at_iso,
            payload_json,
            artifact_ids_json,
        ) = row
        return HarnessEvent(
            id=event_id,
            run_id=run_id,
            seq=seq,
            type=type_,
            actor=actor,
            created_at=datetime.fromisoformat(created_at_iso),
            payload=json.loads(payload_json),
            artifact_ids=json.loads(artifact_ids_json),
        )
