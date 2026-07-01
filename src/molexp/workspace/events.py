"""``WorkspaceEventLog`` — the append-only cross-object event spine at workspace scope.

integration.md §2. A single append-only timeline of "what happened across objects"
in a workspace — `run.*`, `asset.added`, `knowledge.created`, … — persisted to one
``workspace.events.sqlite`` per root. Built on the Layer-0
:class:`molexp.sqlitelog.SeqEventStore` (**not** a copy of the harness event store —
invariant #1), so the two logs share one implementation.

Scope split: the per-run *deep* audit stays in the harness ``harness.sqlite``; this
log is the cross-object coordination view, each event carrying a ``run_id`` /
``asset_id`` / content-hash pointer in ``refs`` to drill down. The DB is a **derived,
rebuildable** operational sidecar — never an authoritative entity file.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from os import PathLike
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from molexp._typing import JSONValue
from molexp.sqlitelog import SeqEventStore, open_wal_connection

__all__ = [
    "WORKSPACE_EVENTS_DB",
    "WorkspaceEvent",
    "WorkspaceEventLog",
    "WorkspaceEventType",
]

WORKSPACE_EVENTS_DB = "workspace.events.sqlite"
"""Filename of the workspace event-spine DB, at the workspace root."""

# One monotonic-seq timeline for the whole workspace (a single scope key).
_WORKSPACE_SCOPE = "workspace"

WorkspaceEventType = Literal[
    "run.created",
    "run.started",
    "run.failed",
    "run.completed",
    "asset.added",
    "knowledge.created",
    "workflow.created",
    "experiment.created",
]
"""The cross-object coordination events a workspace records."""


class WorkspaceEvent(BaseModel):
    """One row of the workspace event timeline."""

    model_config = ConfigDict(frozen=True)

    id: str
    seq: int
    type: WorkspaceEventType
    actor: str
    created_at: datetime
    payload: dict[str, JSONValue] = Field(default_factory=dict)
    refs: list[str] = Field(default_factory=list)


class WorkspaceEventLog:
    """Append-only, workspace-scope timeline of cross-object events.

    One monotonic ``seq`` across the whole workspace (the "what happened" order).
    Backed by the Layer-0 :class:`~molexp.sqlitelog.SeqEventStore` on
    ``<root>/workspace.events.sqlite``.
    """

    def __init__(self, root: str | PathLike[str]) -> None:
        self._path = Path(root) / WORKSPACE_EVENTS_DB
        conn, lock = open_wal_connection(self._path)
        self._store = SeqEventStore(
            conn,
            lock,
            table="workspace_events",
            scope_column="scope",
            refs_column="refs_json",
        )
        self._store.ensure_schema()

    def append(
        self,
        type: WorkspaceEventType,
        actor: str,
        *,
        payload: dict[str, JSONValue] | None = None,
        refs: list[str] | None = None,
    ) -> WorkspaceEvent:
        """Append one event to the workspace timeline and return it.

        Args:
            type: The coordination event type.
            actor: Who emitted it (``"run-lifecycle"`` / ``"agent:<name>"`` / …).
            payload: Optional structured detail.
            refs: Related object ids (``run_id`` / ``asset_id`` / ``"sha256:…"`` / path).

        Returns:
            The persisted :class:`WorkspaceEvent` with its assigned ``seq``.
        """
        event_id = uuid.uuid4().hex
        created_at = datetime.now(tz=UTC)
        payload = dict(payload or {})
        refs = list(refs or [])
        seq = self._store.append(
            event_id=event_id,
            scope_id=_WORKSPACE_SCOPE,
            type=type,
            actor=actor,
            created_at_iso=created_at.isoformat(),
            payload_json=json.dumps(payload, default=str),
            refs_json=json.dumps(refs),
        )
        return WorkspaceEvent(
            id=event_id,
            seq=seq,
            type=type,
            actor=actor,
            created_at=created_at,
            payload=payload,
            refs=refs,
        )

    def list_events(self) -> list[WorkspaceEvent]:
        """Return the whole-workspace timeline, ordered by ``seq``."""
        return [self._row_to_event(row) for row in self._store.list_rows(_WORKSPACE_SCOPE)]

    @staticmethod
    def _row_to_event(row: tuple) -> WorkspaceEvent:
        (event_id, _scope, seq, type_, actor, created_at_iso, payload_json, refs_json) = row
        return WorkspaceEvent(
            id=event_id,
            seq=seq,
            type=type_,
            actor=actor,
            created_at=datetime.fromisoformat(created_at_iso),
            payload=json.loads(payload_json),
            refs=json.loads(refs_json),
        )
