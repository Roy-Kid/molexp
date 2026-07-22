"""``ApprovalStore`` — persisted approval decisions in the run's ``harness.sqlite``.

The store is what turns approval from a per-process callback into a durable
resource: a gate consults it **before** asking anyone (store-first), a suspended
pipeline records its pending requests here so the server inbox can list them,
and a decision written here lets a ledger-resumed re-entry pass the gate.

Replay law (the store's one non-obvious rule):

* **Grants replay.** A stored grant is durable consent — every later re-entry
  of the same request id passes on it without re-asking.
* **Rejections do not replay.** A rejection fails the *current* attempt and is
  kept in the row (plus the event log) as history, but a later
  :meth:`ApprovalStore.record_pending` for the same request id re-opens it as
  pending — "not now" must never deadlock re-entry forever.
* A **granted** row is immutable to ``record_pending`` (never downgraded), so
  a decision racing a re-entry cannot be erased.

Mirrors the ``EventLog`` / ``SQLiteEventLog`` split: a Protocol contract plus
one SQLite implementation sharing the run's existing ``harness.sqlite``
(same ``db_path`` as ``SQLiteEventLog`` / ``SQLiteArtifactLineageStore``).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

from molexp.harness.schemas import ApprovalDecision, ApprovalRequest
from molexp.harness.store._sqlite import open_db

__all__ = ["ApprovalStore", "SQLiteApprovalStore"]


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS approvals (
    request_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    intent TEXT NOT NULL,
    reason TEXT NOT NULL,
    triggered_by_policy TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    requested_at TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('pending', 'granted', 'rejected')),
    decided_by TEXT,
    decided_at TEXT,
    decision_reason TEXT,
    scope TEXT DEFAULT 'approval_gate',
    target_agent_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_approvals_run ON approvals(run_id);
"""


@runtime_checkable
class ApprovalStore(Protocol):
    """Structural type for any persisted-approval backend."""

    def record_pending(self, run_id: str, request: ApprovalRequest) -> None: ...

    def record_decision(self, decision: ApprovalDecision) -> None: ...

    def granted_decision_for(self, request_id: str) -> ApprovalDecision | None: ...

    def pending(self, run_id: str) -> list[ApprovalRequest]: ...


class SQLiteApprovalStore:
    """SQLite-backed :class:`ApprovalStore` sharing the run's ``harness.sqlite``."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        # Shared per-file connection + lock (see ``store._sqlite``): the event
        # log and lineage store on the same path hold the same lock instance,
        # so cross-table writes serialize.
        self._conn, self._lock = open_db(self._path)
        with self._lock:
            self._conn.executescript(_SCHEMA_SQL)
            self._migrate_scope_columns()

    def _migrate_scope_columns(self) -> None:
        """Bring a pre-plan-emergent-07 ``approvals`` table up to the scope set.

        ``CREATE TABLE IF NOT EXISTS`` is a no-op on a pre-existing legacy
        table, so the ``scope`` / ``target_agent_id`` columns are added here
        with ``ALTER TABLE``. Idempotent (columns already present are left
        untouched); legacy rows read back ``scope == 'approval_gate'`` via the
        column default. Mirrors ``store._sqlite._migrate_artifact_edges``.
        """
        existing = {row[1] for row in self._conn.execute("PRAGMA table_info(approvals)")}
        if "scope" not in existing:
            self._conn.execute(
                "ALTER TABLE approvals ADD COLUMN scope TEXT DEFAULT 'approval_gate'"
            )
        if "target_agent_id" not in existing:
            self._conn.execute("ALTER TABLE approvals ADD COLUMN target_agent_id TEXT")

    def record_pending(self, run_id: str, request: ApprovalRequest) -> None:
        """Open (or re-open) *request* as pending.

        Idempotent on a pending row. A **granted** row is never downgraded
        (durable consent survives the race with a re-entering gate). A
        **rejected** row is re-opened as pending — re-entry re-asks (see the
        module's replay law).
        """
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO approvals (
                    request_id, run_id, intent, reason, triggered_by_policy,
                    metadata_json, requested_at, state, scope, target_agent_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)
                ON CONFLICT(request_id) DO UPDATE SET
                    state = 'pending',
                    decided_by = NULL,
                    decided_at = NULL,
                    decision_reason = NULL,
                    scope = excluded.scope,
                    target_agent_id = excluded.target_agent_id
                WHERE approvals.state != 'granted'
                """,
                (
                    request.id,
                    run_id,
                    request.intent,
                    request.reason,
                    request.triggered_by_policy,
                    json.dumps(request.metadata, default=str),
                    request.created_at.isoformat(),
                    request.scope,
                    request.target_agent_id,
                ),
            )

    def record_decision(self, decision: ApprovalDecision) -> None:
        """Persist *decision* onto its pending request.

        Raises:
            ValueError: ``decision.request_id`` has no row in this store —
                a decision for a request that was never asked is a defect,
                never silently inserted.
        """
        state = "granted" if decision.granted else "rejected"
        with self._lock, self._conn:
            cursor = self._conn.execute(
                """
                UPDATE approvals SET
                    state = ?, decided_by = ?, decided_at = ?, decision_reason = ?
                WHERE request_id = ?
                """,
                (
                    state,
                    decision.decided_by,
                    decision.decided_at.isoformat(),
                    decision.reason,
                    decision.request_id,
                ),
            )
            if cursor.rowcount == 0:
                raise ValueError(
                    f"approval decision for unknown request_id {decision.request_id!r} "
                    "— no pending request was ever recorded for it"
                )

    def granted_decision_for(self, request_id: str) -> ApprovalDecision | None:
        """Return the stored **grant** for *request_id*, or ``None``.

        Rejected and pending rows both return ``None`` — only a grant replays.
        """
        with self._lock:
            row = self._conn.execute(
                """
                SELECT decided_by, decided_at, decision_reason FROM approvals
                WHERE request_id = ? AND state = 'granted'
                """,
                (request_id,),
            ).fetchone()
        if row is None:
            return None
        decided_by, decided_at_iso, decision_reason = row
        return ApprovalDecision(
            request_id=request_id,
            granted=True,
            decided_by=decided_by,
            decided_at=datetime.fromisoformat(decided_at_iso),
            reason=decision_reason,
        )

    def pending(self, run_id: str) -> list[ApprovalRequest]:
        """Return every request of *run_id* still awaiting a decision."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT request_id, intent, reason, triggered_by_policy,
                       metadata_json, requested_at, scope, target_agent_id
                FROM approvals
                WHERE run_id = ? AND state = 'pending'
                ORDER BY requested_at
                """,
                (run_id,),
            ).fetchall()
        return [
            ApprovalRequest(
                id=request_id,
                intent=intent,
                reason=reason,
                triggered_by_policy=triggered_by_policy,
                metadata=json.loads(metadata_json),
                created_at=datetime.fromisoformat(requested_at_iso),
                scope=scope or "approval_gate",
                target_agent_id=target_agent_id,
            )
            for (
                request_id,
                intent,
                reason,
                triggered_by_policy,
                metadata_json,
                requested_at_iso,
                scope,
                target_agent_id,
            ) in rows
        ]
