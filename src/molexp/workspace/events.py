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
    "emit_workspace_event",
    "read_workspace_events",
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

    def list_events(
        self,
        *,
        type: WorkspaceEventType | None = None,
        ref: str | None = None,
        newest_first: bool = False,
        limit: int | None = None,
    ) -> list[WorkspaceEvent]:
        """Return the workspace timeline, optionally filtered.

        The no-argument call keeps the original contract (every event, ordered
        by ``seq`` ascending). Filters run before ordering/limiting:

        Args:
            type: Keep only events of this coordination type.
            ref: Keep only events whose ``refs`` contain this object id
                (``run_id`` / ``asset_id`` / content hash / path).
            newest_first: Order by ``seq`` descending (most recent first).
            limit: Truncate to at most this many events *after* ordering, so
                ``newest_first=True, limit=N`` yields the N most recent.
        """
        events = [self._row_to_event(row) for row in self._store.list_rows(_WORKSPACE_SCOPE)]
        if type is not None:
            events = [e for e in events if e.type == type]
        if ref is not None:
            events = [e for e in events if ref in e.refs]
        if newest_first:
            events = list(reversed(events))
        if limit is not None:
            events = events[:limit]
        return events

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


def emit_workspace_event(
    root: str | PathLike[str],
    type: WorkspaceEventType,
    actor: str,
    *,
    payload: dict[str, JSONValue] | None = None,
    refs: list[str] | None = None,
) -> WorkspaceEvent | None:
    """Best-effort append to a workspace's event spine — **default-on + non-fatal**.

    The workspace timeline is a *derived, rebuildable* sidecar, so emission never
    participates in the correctness of the operation that triggered it:

    - **Default-on.** The first emit creates ``<root>/workspace.events.sqlite``;
      no opt-in step exists. Wired emit sites: the run-lifecycle milestones
      (``run.created`` / ``run.started`` / ``run.completed`` / ``run.failed``
      — one row per status change), ``asset.added`` from ``register_artifact``
      and the ``DataAssetLibrary`` import/register verbs (log-line appends and
      checkpoints deliberately stay silent — that is the frequency budget), and
      ``knowledge.created`` from the Bundle create verbs and the plan-record
      KnowledgeItem writers (newly created Concepts only — idempotent repeat
      calls are not creations).
    - **Non-fatal.** Any exception is swallowed and ``None`` is returned; an
      event-log failure must never break the run or asset op that emitted.

    Args:
        root: The workspace root directory (callers resolve it from context they hold).
        type: The coordination event type.
        actor: Who emitted it (``"run-lifecycle"`` / ``"agent:<name>"`` / …).
        payload: Optional structured detail.
        refs: Related object ids (``run_id`` / ``asset_id`` / ``"sha256:…"`` / path).

    Returns:
        The appended :class:`WorkspaceEvent`, or ``None`` on failure.
    """
    try:
        return WorkspaceEventLog(root).append(type, actor, payload=payload, refs=refs)
    except Exception:
        return None


def read_workspace_events(
    root: str | PathLike[str],
    *,
    type: WorkspaceEventType | None = None,
    ref: str | None = None,
    limit: int | None = None,
) -> list[WorkspaceEvent]:
    """Read a workspace's event timeline, most recent first.

    The read counterpart of :func:`emit_workspace_event` — the consumer side
    of the event spine (``molexp runs info`` and the server's run-events
    endpoint both call this, one shared code path). A workspace with no
    timeline yet (no ``workspace.events.sqlite`` at *root* — nothing has
    emitted) has an empty timeline, so ``[]`` is returned without creating
    the DB: reading is always side-effect free.

    Args:
        root: The workspace root directory.
        type: Keep only events of this coordination type.
        ref: Keep only events whose ``refs`` contain this object id
            (typically a ``run_id``).
        limit: Return at most this many (most recent) events.

    Returns:
        Matching :class:`WorkspaceEvent` rows, newest first.
    """
    if not (Path(root) / WORKSPACE_EVENTS_DB).exists():
        return []
    return WorkspaceEventLog(root).list_events(type=type, ref=ref, newest_first=True, limit=limit)
