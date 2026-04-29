"""On-disk persistence + listing for agent sessions.

The runtime writes one ``sessions/<id>/metadata.json`` per session at start
and re-writes it on completion. This module owns the read side: a single
walk that produces summaries the server can merge with the in-memory
``_sessions`` dict so historical sessions survive a server restart.

Format mirrors what :class:`PydanticAIRuntime` writes — keep that file in
sync if you change the schema here.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SESSIONS_DIR_NAME = "sessions"
METADATA_FILE = "metadata.json"


@dataclass(frozen=True)
class PersistedSessionSummary:
    """Lightweight view of an on-disk session — enough to render a list row."""

    session_id: str
    status: str
    goal_description: str
    created_at: str | None
    completed_at: str | None
    plan_mode: bool = False
    skill_id: str | None = None


def sessions_dir(workspace_root: str | Path) -> Path:
    """Return the workspace's session directory, ensuring it exists."""
    path = Path(workspace_root) / SESSIONS_DIR_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_persisted_sessions(workspace_root: str | Path) -> list[PersistedSessionSummary]:
    """Walk ``<root>/sessions/*/metadata.json`` and yield summaries.

    Sorted by ``created_at`` descending (newest first); files that fail
    to parse are silently skipped — a single corrupt session must not
    take down the listing for everything else.
    """
    root = sessions_dir(workspace_root)
    rows: list[PersistedSessionSummary] = []
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        meta_path = entry / METADATA_FILE
        if not meta_path.exists():
            continue
        summary = _read_summary(meta_path)
        if summary is not None:
            rows.append(summary)
    rows.sort(key=lambda r: r.created_at or "", reverse=True)
    return rows


def get_persisted_session(
    workspace_root: str | Path, session_id: str
) -> PersistedSessionSummary | None:
    """Look up one session by id without scanning the whole directory."""
    meta_path = Path(workspace_root) / SESSIONS_DIR_NAME / session_id / METADATA_FILE
    if not meta_path.exists():
        return None
    return _read_summary(meta_path)


def write_session_metadata(
    workspace_root: str | Path,
    session_id: str,
    *,
    status: str,
    goal_description: str,
    constraints: dict[str, Any] | None = None,
    success_criteria: list[str] | None = None,
    created_at: str | None = None,
    completed_at: str | None = None,
    plan_mode: bool = False,
    instructions_override: str | None = None,
    skill_id: str | None = None,
    skill_instructions: str = "",
) -> None:
    """Atomically write the metadata file for one session.

    Called twice in a normal lifecycle: once at session start (status
    ``running``, no ``completed_at``) and once at termination with the
    final status. The atomic rename guards against torn writes when the
    server is killed mid-flush.

    The ``plan_mode``, ``instructions_override``, ``skill_id`` and
    ``skill_instructions`` fields persist the goal-shape extras so a
    server restart preserves enough state to re-render the right UI
    (chat vs. plan view) and to resume the session faithfully.
    """
    target_dir = Path(workspace_root) / SESSIONS_DIR_NAME / session_id
    target_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "session_id": session_id,
        "status": status,
        "goal": {
            "description": goal_description,
            "constraints": constraints or {},
            "success_criteria": list(success_criteria or []),
            "plan_mode": bool(plan_mode),
            "instructions_override": instructions_override,
            "skill_id": skill_id,
            "skill_instructions": skill_instructions,
        },
        "created_at": created_at or _now_iso(),
    }
    if completed_at is not None:
        payload["completed_at"] = completed_at
    path = target_dir / METADATA_FILE
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    os.replace(tmp, path)


def _read_summary(meta_path: Path) -> PersistedSessionSummary | None:
    try:
        raw = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    goal = raw.get("goal") if isinstance(raw.get("goal"), dict) else {}
    return PersistedSessionSummary(
        session_id=str(raw.get("session_id") or meta_path.parent.name),
        status=str(raw.get("status") or "unknown"),
        goal_description=str(goal.get("description") or ""),
        created_at=raw.get("created_at"),
        completed_at=raw.get("completed_at"),
        plan_mode=bool(goal.get("plan_mode", False)),
        skill_id=goal.get("skill_id") if isinstance(goal.get("skill_id"), str) else None,
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
