"""Plan/chat LLM-call cache projected onto agent-task sessions.

Authoritative audit still lives on the run's artifact store (``prompt`` +
``log``). This module is a **cache** for the Agents chat view:

* full prompt / raw body under ``agent/_tasks/<task_id>/llm_cache/<id>.json``
* a slim ``llm_call`` event in ``events.json`` (previews + cache_id + artifact
  refs) so the conversation timeline lists every gateway call without bloating
  the event log with multi-KB bodies

Retention is opportunistic — prune runs on every write and when a plan gateway
is built. Defaults: 7-day age, 100 calls per task, 64 MiB per workspace.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from mollog import get_logger

from molexp.services.agent_task_store import (
    agent_tasks_dir,
    append_agent_task_events,
    read_agent_task_events,
    write_agent_task_events,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from molexp.harness.gateways.llm_trace import LlmCallTrace

__all__ = [
    "DEFAULT_MAX_AGE_DAYS",
    "DEFAULT_MAX_BYTES_PER_WORKSPACE",
    "DEFAULT_MAX_CALLS_PER_TASK",
    "LLM_CACHE_SUBDIR",
    "PREVIEW_CHARS",
    "LlmCachePruneStats",
    "ensure_plan_session_started",
    "make_llm_call_observer",
    "prune_agent_task_llm_cache",
    "record_llm_call",
]

_LOG = get_logger(__name__)

LLM_CACHE_SUBDIR = "llm_cache"
PREVIEW_CHARS = 2000
DEFAULT_MAX_AGE_DAYS = 7
DEFAULT_MAX_CALLS_PER_TASK = 100
DEFAULT_MAX_BYTES_PER_WORKSPACE = 64 * 1024 * 1024
EVENT_TYPE = "llm_call"


@dataclass(frozen=True, slots=True)
class LlmCachePruneStats:
    """What one prune pass removed."""

    removed_files: int = 0
    removed_events: int = 0
    freed_bytes: int = 0


def ensure_plan_session_started(
    workspace_root: str | Path,
    task_id: str,
    *,
    draft: str,
    turn_id: str | None = None,
) -> None:
    """Emit a single ``loop_started`` so mid-plan ``llm_call`` events group correctly.

    Idempotent per ``turn_id`` (or once when ``turn_id`` is None). Best-effort.
    """
    try:
        existing = read_agent_task_events(workspace_root, task_id)
        for event in existing:
            if event.get("type") != "loop_started":
                continue
            payload = event.get("payload")
            if not isinstance(payload, dict):
                continue
            if turn_id is None or payload.get("turn_id") == turn_id:
                return
        append_agent_task_events(
            workspace_root,
            task_id,
            [
                {
                    "type": "loop_started",
                    "ts": _now_iso(),
                    "payload": {
                        "user_input": draft,
                        "turn_id": turn_id,
                        "mode": "plan",
                    },
                }
            ],
        )
    except Exception as exc:  # cache view — never break plan launch
        _LOG.warning(f"[llm-cache {task_id}] loop_started write failed: {exc!r}")


def record_llm_call(
    workspace_root: str | Path,
    task_id: str,
    *,
    agent_name: str,
    model: str,
    prompt: str,
    raw: str,
    prompt_artifact_id: str,
    raw_artifact_id: str,
    turn_id: str | None = None,
) -> str:
    """Persist full bodies to the cache dir and append a slim ``llm_call`` event.

    Returns the ``cache_id``. Raises on hard path errors (caller may swallow).
    """
    cache_id = uuid4().hex[:12]
    ts = _now_iso()
    body = {
        "cache_id": cache_id,
        "ts": ts,
        "agent_name": agent_name,
        "model": model,
        "prompt": prompt,
        "raw": raw,
        "prompt_artifact_id": prompt_artifact_id,
        "raw_artifact_id": raw_artifact_id,
        "turn_id": turn_id,
    }
    cache_dir = _cache_dir(workspace_root, task_id)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{cache_id}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)  # noqa: PTH105

    append_agent_task_events(
        workspace_root,
        task_id,
        [
            {
                "type": EVENT_TYPE,
                "ts": ts,
                "payload": {
                    "agent_name": agent_name,
                    "model": model,
                    "cache_id": cache_id,
                    "prompt_preview": _preview(prompt),
                    "raw_preview": _preview(raw),
                    "prompt_chars": len(prompt),
                    "raw_chars": len(raw),
                    "prompt_artifact_id": prompt_artifact_id,
                    "raw_artifact_id": raw_artifact_id,
                    "turn_id": turn_id,
                    "mode": "plan",
                    "cache": True,
                },
            }
        ],
    )
    # Opportunistic retention after every write.
    try:
        prune_agent_task_llm_cache(workspace_root)
    except Exception as exc:  # prune is best-effort
        _LOG.warning(f"[llm-cache] prune after write failed: {exc!r}")
    return cache_id


def make_llm_call_observer(
    workspace_root: str | Path,
    task_id: str,
    *,
    turn_id: str | None = None,
) -> Callable[[LlmCallTrace], None]:
    """Return a gateway ``on_llm_call`` sink bound to one agent-task session."""

    root = str(workspace_root)

    def _observe(trace: LlmCallTrace) -> None:
        record_llm_call(
            root,
            task_id,
            agent_name=trace.agent_name,
            model=trace.model,
            prompt=trace.prompt,
            raw=trace.raw,
            prompt_artifact_id=trace.prompt_artifact_id,
            raw_artifact_id=trace.raw_artifact_id,
            turn_id=turn_id,
        )

    return _observe


def prune_agent_task_llm_cache(
    workspace_root: str | Path,
    *,
    max_age_days: float = DEFAULT_MAX_AGE_DAYS,
    max_calls_per_task: int = DEFAULT_MAX_CALLS_PER_TASK,
    max_bytes_per_workspace: int = DEFAULT_MAX_BYTES_PER_WORKSPACE,
) -> LlmCachePruneStats:
    """Drop expired / over-budget LLM cache files and their ``llm_call`` events.

    Order of cuts (per task, then workspace):

    1. files older than ``max_age_days``
    2. keep only the newest ``max_calls_per_task`` files per task
    3. if total workspace cache bytes still exceed the cap, drop oldest files
       across tasks until under budget
    """
    from molexp.services.agent_task_store import agent_tasks_dir

    root = Path(workspace_root)
    tasks_root = agent_tasks_dir(root)
    if not tasks_root.is_dir():
        return LlmCachePruneStats()

    cutoff = time.time() - max(0.0, max_age_days) * 86400.0
    removed_files = 0
    freed = 0
    # (mtime, size, path, task_id, cache_id)
    survivors: list[tuple[float, int, Path, str, str]] = []

    for task_dir in tasks_root.iterdir():
        if not task_dir.is_dir():
            continue
        cache_dir = task_dir / LLM_CACHE_SUBDIR
        if not cache_dir.is_dir():
            continue
        task_id = task_dir.name
        files: list[tuple[float, int, Path, str]] = []
        for path in cache_dir.glob("*.json"):
            try:
                st = path.stat()
            except OSError:
                continue
            cache_id = path.stem
            if st.st_mtime < cutoff:
                freed += _unlink(path)
                removed_files += 1
                _drop_events_for_cache_ids(root, task_id, {cache_id})
                continue
            files.append((st.st_mtime, st.st_size, path, cache_id))
        files.sort(key=lambda row: row[0], reverse=True)  # newest first
        keep = files[: max(0, max_calls_per_task)]
        drop = files[max(0, max_calls_per_task) :]
        if drop:
            drop_ids = {row[3] for row in drop}
            for _mtime, size, path, _cid in drop:
                if path.exists():
                    freed += size
                    path.unlink(missing_ok=True)
                    removed_files += 1
            _drop_events_for_cache_ids(root, task_id, drop_ids)
        for mtime, size, path, cache_id in keep:
            survivors.append((mtime, size, path, task_id, cache_id))

    total = sum(row[1] for row in survivors)
    if total > max_bytes_per_workspace and survivors:
        survivors.sort(key=lambda row: row[0])  # oldest first
        by_task: dict[str, set[str]] = {}
        for _mtime, size, path, task_id, cache_id in survivors:
            if total <= max_bytes_per_workspace:
                break
            if not path.exists():
                continue
            path.unlink(missing_ok=True)
            removed_files += 1
            freed += size
            total -= size
            by_task.setdefault(task_id, set()).add(cache_id)
        for task_id, ids in by_task.items():
            _drop_events_for_cache_ids(root, task_id, ids)

    return LlmCachePruneStats(
        removed_files=removed_files,
        removed_events=0,  # counted per-file above via event filter; not tracked finely
        freed_bytes=freed,
    )


# ── internals ────────────────────────────────────────────────────────────────


def _cache_dir(workspace_root: str | Path, task_id: str) -> Path:
    """``agent/_tasks/<task_id>/llm_cache`` — rejects path-escape task ids."""
    if not task_id or len(task_id) > 128 or task_id in {".", ".."}:
        raise ValueError(f"invalid agent task id: {task_id!r}")
    if "/" in task_id or "\\" in task_id or ".." in task_id:
        raise ValueError(f"invalid agent task id: {task_id!r}")
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    if any(ch not in allowed for ch in task_id):
        raise ValueError(f"invalid agent task id: {task_id!r}")
    root = agent_tasks_dir(workspace_root).resolve()
    path = (root / task_id / LLM_CACHE_SUBDIR).resolve()
    if root not in path.parents:
        raise ValueError(f"agent task path escapes tasks root: {task_id!r}")
    return path


def _preview(text: str) -> str:
    if len(text) <= PREVIEW_CHARS:
        return text
    return text[:PREVIEW_CHARS] + f"\n… [{len(text) - PREVIEW_CHARS} more chars]"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _unlink(path: Path) -> int:
    try:
        size = path.stat().st_size
    except OSError:
        size = 0
    path.unlink(missing_ok=True)
    return size


def _drop_events_for_cache_ids(
    workspace_root: str | Path,
    task_id: str,
    cache_ids: set[str],
) -> int:
    if not cache_ids:
        return 0
    events = read_agent_task_events(workspace_root, task_id)
    if not events:
        return 0
    kept: list[dict[str, Any]] = []
    removed = 0
    for event in events:
        if event.get("type") == EVENT_TYPE:
            payload = event.get("payload")
            if isinstance(payload, dict) and payload.get("cache_id") in cache_ids:
                removed += 1
                continue
        kept.append(event)
    if removed:
        write_agent_task_events(workspace_root, task_id, kept)
    return removed
