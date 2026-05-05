"""One-shot migrator for legacy ``<workspace>/sessions/`` directories.

Older workspaces wrote ``sessions/<id>/metadata.json`` from the
PydanticAI plugin runtime (deleted in phase 3). After the cutover the
canonical layout is ``<workspace>/.molexp-agent/sessions/<id>/...``;
this helper reads the legacy metadata and lays down a tombstone
``session.json`` with ``status="legacy"`` so the UI lists the
session read-only.

Messages are intentionally *not* carried over: the legacy
``history.json`` shape diverges enough from
:mod:`molexp.agent.state.sessions` that copying it would risk
mis-rendering. Replay across the schema change isn't supported.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from molexp.agent.state.sessions import SessionMetadata, SessionStore
from molexp.agent.types import Goal, SessionStatus, utc_now

LEGACY_SESSIONS_DIRNAME = "sessions"
LEGACY_METADATA_FILENAME = "metadata.json"


@dataclass(frozen=True)
class MigrationResult:
    migrated: tuple[str, ...]
    skipped: tuple[str, ...]


def migrate_legacy_sessions(
    workspace_root: str | Path,
    sessions_store: SessionStore,
) -> MigrationResult:
    """Walk ``<root>/sessions/`` and write tombstones via ``sessions_store``.

    ``sessions_store`` is the live ``.molexp-agent/sessions`` store the
    caller already constructed. Returns the per-session ids that were
    migrated and the ones that were skipped (already present in the
    new layout, no metadata, or unparseable).
    """

    root = Path(workspace_root) / LEGACY_SESSIONS_DIRNAME
    if not root.exists():
        return MigrationResult(migrated=(), skipped=())

    migrated: list[str] = []
    skipped: list[str] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        sid = entry.name
        meta_path = entry / LEGACY_METADATA_FILENAME
        if not meta_path.exists():
            skipped.append(sid)
            continue
        if sessions_store.read_metadata(sid) is not None:
            skipped.append(sid)
            continue
        legacy = _load_legacy_metadata(meta_path)
        if legacy is None:
            skipped.append(sid)
            continue
        sessions_store.write_metadata(_to_tombstone(sid, legacy))
        migrated.append(sid)
    return MigrationResult(migrated=tuple(migrated), skipped=tuple(skipped))


def _load_legacy_metadata(path: Path) -> dict | None:
    raw = path.read_text()
    if not raw.strip():
        return None
    payload = json.loads(raw)
    return payload if isinstance(payload, dict) else None


def _to_tombstone(session_id: str, legacy: dict) -> SessionMetadata:
    """Translate a legacy metadata blob into a ``status="legacy"`` tombstone."""

    goal_blob = legacy.get("goal") if isinstance(legacy.get("goal"), dict) else {}
    description = str(goal_blob.get("description") or "")
    constraints = (
        goal_blob.get("constraints") if isinstance(goal_blob.get("constraints"), dict) else {}
    )
    success_criteria = goal_blob.get("success_criteria") or []
    instructions_override = (
        goal_blob.get("instructions_override")
        if isinstance(goal_blob.get("instructions_override"), str)
        else None
    )
    skill_id = goal_blob.get("skill_id") if isinstance(goal_blob.get("skill_id"), str) else None
    goal = Goal(
        description=description,
        constraints=dict(constraints),
        success_criteria=list(success_criteria),
        instructions_override=instructions_override,
        skill_id=skill_id,
    )
    summary_parts: list[str] = ["legacy session — read-only tombstone"]
    if status := legacy.get("status"):
        summary_parts.append(f"prior status: {status}")
    return SessionMetadata(
        session_id=session_id,
        goal=goal,
        status=SessionStatus.LEGACY,
        updated_at=utc_now(),
        summary=" — ".join(summary_parts),
    )


__all__ = ["MigrationResult", "migrate_legacy_sessions"]
