"""Lightweight on-disk metadata for user-facing agent tasks.

Layout (under the product agent home)::

    <workspace>/agent/
        _tasks/<task_id>/task.json       # this module (canonical)
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

# On-disk agent folder name — single product agent home.
AGENT_HOME_NAME = "agent"
TASKS_SUBDIR = "_tasks"
TASK_FILE = "task.json"
EVENTS_FILE = "events.json"


#: The two agent-task modes. On-disk ``active_mode`` must be one of these.
AgentTaskMode = Literal["chat", "plan"]


def _parse_mode(raw: object) -> AgentTaskMode | None:
    if raw == "chat":
        return "chat"
    if raw == "plan":
        return "plan"
    return None


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


def agent_tasks_dir(workspace_root: str | Path, *, create: bool = False) -> Path:
    """Task-metadata root: ``agent/_tasks/``.

    *create* is **off by default**. Listing must not mkdir — remote workspace
    roots (``Arrhenius:/home/...``) resolve to absolute POSIX paths that are
    not on the local disk; ``Path.mkdir`` would try to create ``/home/...`` on
    the laptop and fail with ``OSError: Operation not supported``.

    Pass ``create=True`` only on write paths that already know the root is a
    local filesystem. Remote agent-task I/O should go through ``workspace.fs``
    (not yet wired here) — until then writes on remote roots raise ``OSError``.
    """
    path = agent_home_dir(workspace_root) / TASKS_SUBDIR
    if create:
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


def _task_dir(workspace_root: str | Path, task_id: str, *, create: bool = False) -> Path:
    safe_id = _validate_task_id(task_id)
    root = agent_tasks_dir(workspace_root, create=create)
    path = root / safe_id
    # resolve only when the path is local and parents exist; remote absolute
    # roots must not force resolve (would follow a non-existent /home/...).
    try:
        path_r = path.resolve()
        root_r = root.resolve()
    except OSError:
        path_r, root_r = path, root
    if path_r != root_r and root_r not in path_r.parents:
        raise ValueError(f"agent task path escapes tasks root: {task_id!r}")
    return path


def _required_str(raw: dict[str, Any], key: str) -> str | None:
    value = raw.get(key)
    if isinstance(value, str) and value:
        return value
    return None


def _optional_str(raw: dict[str, Any], key: str) -> str | None:
    value = raw.get(key)
    return value if isinstance(value, str) else None


def _parse_task_json(path: Path, task_id: str) -> PersistedAgentTask | None:
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    mode = _parse_mode(raw.get("active_mode"))
    session_id = _required_str(raw, "session_id")
    title = _required_str(raw, "title")
    status = _required_str(raw, "status")
    created_at = _required_str(raw, "created_at")
    if mode is None or session_id is None or title is None or status is None or created_at is None:
        return None
    goal = raw.get("goal")
    if not isinstance(goal, str):
        return None
    persisted_id = _required_str(raw, "task_id")
    if persisted_id is None or persisted_id != task_id:
        return None
    return PersistedAgentTask(
        task_id=persisted_id,
        session_id=session_id,
        title=title,
        goal=goal,
        status=status,
        created_at=created_at,
        updated_at=_optional_str(raw, "updated_at"),
        plan_mode=mode == "plan",
        active_mode=mode,
        active_turn_id=_optional_str(raw, "active_turn_id"),
        active_plan_task_id=_optional_str(raw, "active_plan_task_id"),
        pending_plan_draft=_optional_str(raw, "pending_plan_draft"),
        skill_id=_optional_str(raw, "skill_id"),
        project_id=_optional_str(raw, "project_id"),
        experiment_id=_optional_str(raw, "experiment_id"),
        run_id=_optional_str(raw, "run_id"),
    )


def list_agent_task_metadata(workspace_root: str | Path) -> list[PersistedAgentTask]:
    """List on-disk agent tasks. Missing / non-local roots → empty list (no mkdir)."""
    root = agent_tasks_dir(workspace_root, create=False)
    try:
        if not root.is_dir():
            return []
        entries = list(root.iterdir())
    except OSError:
        # Remote path or unreadable root — treat as no tasks, never 500.
        return []
    rows: list[PersistedAgentTask] = []
    for entry in entries:
        if not entry.is_dir():
            continue
        task = _read_task_file(entry, entry.name)
        if task is not None:
            rows.append(task)
    rows.sort(key=lambda r: r.updated_at or r.created_at, reverse=True)
    return rows


def _read_task_file(task_dir: Path, task_id: str) -> PersistedAgentTask | None:
    """Read ``task.json`` only."""
    path = task_dir / TASK_FILE
    if not path.is_file():
        return None
    return _parse_task_json(path, task_id)


def read_agent_task_metadata(
    workspace_root: str | Path,
    task_id: str,
) -> PersistedAgentTask | None:
    try:
        return _read_task_file(_task_dir(workspace_root, task_id), task_id)
    except ValueError:
        return None


def write_agent_task_metadata(
    workspace_root: str | Path,
    task: PersistedAgentTask,
) -> None:
    target_dir = _task_dir(workspace_root, task.task_id, create=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "task_id": task.task_id,
        "session_id": task.session_id,
        "title": task.title,
        "goal": task.goal,
        "status": task.status,
        "created_at": task.created_at,
        "updated_at": task.updated_at or _now_iso(),
        "plan_mode": task.active_mode == "plan",
        "active_mode": task.active_mode,
        "active_turn_id": task.active_turn_id,
        "active_plan_task_id": task.active_plan_task_id,
        "pending_plan_draft": task.pending_plan_draft,
        "skill_id": task.skill_id,
        "project_id": task.project_id,
        "experiment_id": task.experiment_id,
        "run_id": task.run_id,
    }
    path = target_dir / TASK_FILE
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
    target_dir = _task_dir(workspace_root, task_id, create=True)
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
    except (OSError, json.JSONDecodeError):
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
