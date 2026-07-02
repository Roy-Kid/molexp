"""Lightweight on-disk metadata for user-facing agent tasks."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

AGENT_TASKS_DIR_NAME = "agent_tasks"
METADATA_FILE = "metadata.json"
EVENTS_FILE = "events.json"


@dataclass(frozen=True)
class PersistedAgentTask:
    task_id: str
    session_id: str
    title: str
    goal: str
    status: str
    created_at: str
    updated_at: str | None = None
    plan_mode: bool = False
    skill_id: str | None = None


def agent_tasks_dir(workspace_root: str | Path) -> Path:
    path = Path(workspace_root) / AGENT_TASKS_DIR_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_agent_task_metadata(workspace_root: str | Path) -> list[PersistedAgentTask]:
    root = agent_tasks_dir(workspace_root)
    rows: list[PersistedAgentTask] = []
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        meta_path = entry / METADATA_FILE
        if not meta_path.exists():
            continue
        task = read_agent_task_metadata(workspace_root, entry.name)
        if task is not None:
            rows.append(task)
    rows.sort(key=lambda r: r.updated_at or r.created_at, reverse=True)
    return rows


def read_agent_task_metadata(
    workspace_root: str | Path,
    task_id: str,
) -> PersistedAgentTask | None:
    meta_path = Path(workspace_root) / AGENT_TASKS_DIR_NAME / task_id / METADATA_FILE
    if not meta_path.exists():
        return None
    try:
        raw = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    return PersistedAgentTask(
        task_id=str(raw.get("task_id") or task_id),
        session_id=str(raw.get("session_id") or task_id),
        title=str(raw.get("title") or "Untitled agent task"),
        goal=str(raw.get("goal") or ""),
        status=str(raw.get("status") or "unknown"),
        created_at=str(raw.get("created_at") or _now_iso()),
        updated_at=raw.get("updated_at") if isinstance(raw.get("updated_at"), str) else None,
        plan_mode=bool(raw.get("plan_mode", False)),
        skill_id=raw.get("skill_id") if isinstance(raw.get("skill_id"), str) else None,
    )


def write_agent_task_metadata(
    workspace_root: str | Path,
    task: PersistedAgentTask,
) -> None:
    target_dir = Path(workspace_root) / AGENT_TASKS_DIR_NAME / task.task_id
    target_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "task_id": task.task_id,
        "session_id": task.session_id,
        "title": task.title,
        "goal": task.goal,
        "status": task.status,
        "created_at": task.created_at,
        "updated_at": task.updated_at or _now_iso(),
        "plan_mode": task.plan_mode,
        "skill_id": task.skill_id,
    }
    path = target_dir / METADATA_FILE
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    os.replace(tmp, path)  # noqa: PTH105


def write_agent_task_events(
    workspace_root: str | Path,
    task_id: str,
    events: list[dict[str, Any]],
) -> None:
    """Persist a task's session events (``{type, ts, payload}`` records).

    Used to record a synthesized transcript (e.g. a PlanMode run) so the session
    view shows the whole flow even though no live runtime session exists.
    """
    target_dir = Path(workspace_root) / AGENT_TASKS_DIR_NAME / task_id
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / EVENTS_FILE
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(events, indent=2, ensure_ascii=False))
    os.replace(tmp, path)  # noqa: PTH105


def read_agent_task_events(
    workspace_root: str | Path,
    task_id: str,
) -> list[dict[str, Any]]:
    """Read a task's persisted session events, or ``[]`` when none."""
    path = Path(workspace_root) / AGENT_TASKS_DIR_NAME / task_id / EVENTS_FILE
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    return data if isinstance(data, list) else []


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()
