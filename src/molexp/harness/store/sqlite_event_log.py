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
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from molexp.harness.errors import EventSeqConflictError
from molexp.harness.schemas import EventType, HarnessEvent
from molexp.harness.store._sqlite import open_db
from molexp.sqlitelog import SeqConflictError, SeqEventStore

__all__ = ["SQLiteEventLog"]


class SQLiteEventLog:
    """SQLite-backed append-only event log."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        # ``_lock`` is shared per DB file (see ``store._sqlite``); the
        # lineage store on the same path holds the same lock instance so
        # the cross-table writes serialize. All connection access goes
        # through it because ``StageRunner`` calls us from worker threads.
        self._conn, self._lock = open_db(self._path)
        # The append-only ``events`` seq-log engine (Layer-0 primitive).
        # ``refs_column="artifact_ids_json"`` reproduces the existing on-disk
        # harness ``events`` schema exactly (so pre-existing DBs are unchanged).
        self._store = SeqEventStore(
            self._conn,
            self._lock,
            table="events",
            scope_column="run_id",
            refs_column="artifact_ids_json",
        )
        self._store.ensure_schema()

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
        return [self._row_to_event(row) for row in self._store.list_rows(run_id)]

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
        # Delegate the seq-append (BEGIN → MAX(seq)+1 → INSERT, under the shared
        # lock) to the Layer-0 SeqEventStore; wrap its SeqConflictError in the
        # harness-typed EventSeqConflictError so callers see the same contract.
        try:
            assigned_seq = self._store.append(
                event_id=event_id,
                scope_id=run_id,
                type=type_,
                actor=actor,
                created_at_iso=created_at.isoformat(),
                payload_json=json.dumps(payload, default=str),
                refs_json=json.dumps(artifact_ids),
                seq=seq,
            )
        except SeqConflictError as exc:
            # Preserve the harness contract that EventSeqConflictError chains the
            # underlying sqlite3.IntegrityError directly (SeqEventStore raised the
            # SeqConflictError ``from`` that IntegrityError).
            raise EventSeqConflictError(
                f"duplicate (run_id={run_id!r}, seq={seq}) in event log"
            ) from exc.__cause__

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
