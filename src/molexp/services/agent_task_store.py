"""Lightweight on-disk metadata for user-facing agent tasks.

Layout (under the product agent home)::

    <workspace>/agent/
        _tasks/<task_id>/metadata.json   # this module
        _tasks/<task_id>/events.json     # optional transcript
        <session_id>/                    # AgentSession (runner)
        .scratch/                        # LocalExecutionEnv
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

# Co-located with InteractiveLoop.name ("agent") — single product agent home.
AGENT_HOME_NAME = "agent"
TASKS_SUBDIR = "_tasks"
METADATA_FILE = "metadata.json"
EVENTS_FILE = "events.json"


#: The two agent-task modes. Persisted values outside this vocabulary are
#: coerced on read, so consumers never have to re-widen the field.
AgentTaskMode = Literal["chat", "plan"]


def _coerce_mode(raw: object, *, plan_mode: bool) -> AgentTaskMode:
    """Read a persisted ``active_mode``, falling back to the plan_mode flag."""
    if raw == "chat":
        return "chat"
    if raw == "plan":
        return "plan"
    return "plan" if plan_mode else "chat"


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
    active_mode: AgentTaskMode = "chat"
    active_turn_id: str | None = None
    active_plan_task_id: str | None = None
    pending_plan_draft: str | None = None
    skill_id: str | None = None
    # Mount scope (vision-loop-11) — rebuilt verbatim on re-attach.
    project_id: str | None = None
    experiment_id: str | None = None
    run_id: str | None = None


def agent_home_dir(workspace_root: str | Path) -> Path:
    """Return ``<workspace>/agent`` (the agent concept home)."""
    return Path(workspace_root) / AGENT_HOME_NAME


def agent_tasks_dir(workspace_root: str | Path) -> Path:
    """Task-metadata root: ``agent/_tasks/`` (created on use)."""
    path = agent_home_dir(workspace_root) / TASKS_SUBDIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def _validate_task_id(task_id: str) -> str:
    """Reject path segments that could escape ``agent/_tasks/``.

    Task ids are opaque client-facing tokens (often ``task-`` + hex). Allow a
    conservative character set and forbid ``..`` / separators so
    :func:`delete_agent_task` cannot ``rmtree`` outside the tasks root.
    """
    if not task_id or len(task_id) > 128:
        raise ValueError(f"invalid agent task id: {task_id!r}")
    if task_id in {".", ".."} or "/" in task_id or "\\" in task_id:
        raise ValueError(f"invalid agent task id: {task_id!r}")
    # Keep alnum + common separators; reject anything else (incl. null bytes).
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    if any(ch not in allowed for ch in task_id):
        raise ValueError(f"invalid agent task id: {task_id!r}")
    return task_id


def _task_dir(workspace_root: str | Path, task_id: str) -> Path:
    safe_id = _validate_task_id(task_id)
    path = (agent_tasks_dir(workspace_root) / safe_id).resolve()
    root = agent_tasks_dir(workspace_root).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"agent task path escapes tasks root: {task_id!r}")
    return path


def _read_meta_file(meta_path: Path, task_id: str) -> PersistedAgentTask | None:
    if not meta_path.exists():
        return None
    try:
        raw = json.loads(meta_path.read_text())
    except OSError, json.JSONDecodeError:
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
        active_mode=_coerce_mode(
            raw.get("active_mode"), plan_mode=bool(raw.get("plan_mode", False))
        ),
        active_turn_id=(
            raw.get("active_turn_id") if isinstance(raw.get("active_turn_id"), str) else None
        ),
        active_plan_task_id=(
            raw.get("active_plan_task_id")
            if isinstance(raw.get("active_plan_task_id"), str)
            else None
        ),
        pending_plan_draft=(
            raw.get("pending_plan_draft")
            if isinstance(raw.get("pending_plan_draft"), str)
            else None
        ),
        skill_id=raw.get("skill_id") if isinstance(raw.get("skill_id"), str) else None,
        project_id=raw.get("project_id") if isinstance(raw.get("project_id"), str) else None,
        experiment_id=(
            raw.get("experiment_id") if isinstance(raw.get("experiment_id"), str) else None
        ),
        run_id=raw.get("run_id") if isinstance(raw.get("run_id"), str) else None,
    )


def list_agent_task_metadata(workspace_root: str | Path) -> list[PersistedAgentTask]:
    root = agent_tasks_dir(workspace_root)
    rows: list[PersistedAgentTask] = []
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        task = _read_meta_file(entry / METADATA_FILE, entry.name)
        if task is not None:
            rows.append(task)
    rows.sort(key=lambda r: r.updated_at or r.created_at, reverse=True)
    return rows


def read_agent_task_metadata(
    workspace_root: str | Path,
    task_id: str,
) -> PersistedAgentTask | None:
    try:
        return _read_meta_file(_task_dir(workspace_root, task_id) / METADATA_FILE, task_id)
    except ValueError:
        return None


def write_agent_task_metadata(
    workspace_root: str | Path,
    task: PersistedAgentTask,
) -> None:
    target_dir = _task_dir(workspace_root, task.task_id)
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
        "active_mode": task.active_mode,
        "active_turn_id": task.active_turn_id,
        "active_plan_task_id": task.active_plan_task_id,
        "pending_plan_draft": task.pending_plan_draft,
        "skill_id": task.skill_id,
        "project_id": task.project_id,
        "experiment_id": task.experiment_id,
        "run_id": task.run_id,
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
    target_dir = _task_dir(workspace_root, task_id)
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
    try:
        path = _task_dir(workspace_root, task_id) / EVENTS_FILE
    except ValueError:
        return []
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except OSError, json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def append_agent_task_events(
    workspace_root: str | Path,
    task_id: str,
    events: list[dict[str, Any]],
) -> None:
    """Append transcript events while preserving prior chat/plan turns."""
    current = read_agent_task_events(workspace_root, task_id)
    write_agent_task_events(workspace_root, task_id, [*current, *events])


def _event_dedupe_key(event: dict[str, Any]) -> tuple[Any, ...]:
    return (event.get("type"), event.get("ts"), repr(event.get("payload")))


def merge_agent_task_events(
    workspace_root: str | Path,
    task_id: str,
    events: list[dict[str, Any]],
) -> int:
    """Merge *events* into the on-disk transcript without duplicates.

    Used to flush the in-memory live turn onto disk so chat history survives
    ``molexp serve`` restarts. Returns the number of newly written events.
    """
    if not events:
        return 0
    current = read_agent_task_events(workspace_root, task_id)
    seen = {_event_dedupe_key(event) for event in current if isinstance(event, dict)}
    extra = [
        event
        for event in events
        if isinstance(event, dict) and _event_dedupe_key(event) not in seen
    ]
    if not extra:
        return 0
    write_agent_task_events(workspace_root, task_id, [*current, *extra])
    return len(extra)


def delete_agent_task(workspace_root: str | Path, task_id: str) -> bool:
    """Remove task metadata from disk.

    Returns ``True`` if the task directory was removed.
    """
    try:
        path = _task_dir(workspace_root, task_id)
    except ValueError:
        return False
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
        return True
    return False


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()
