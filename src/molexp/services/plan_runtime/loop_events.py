"""Project plan ReAct AgentEvents into agent-task ``events.json``.

The planning loop previously discarded its sink stream, so the Agents UI sat on
``stage_started: Drafting...`` with no thinking / tool animation until the whole
LLM pass finished (or forever, when turn grouping hid those stages).

This module is services-layer only (writes agent-task store) — harness stays
free of services imports via the injected ``on_loop_event`` observer.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from mollog import get_logger

__all__ = ["make_plan_loop_event_observer"]

_LOG = get_logger(__name__)

# Terminal planning-loop events must not close the agent-task turn — Phase 1
# still has the hard review gate after the board loop finishes.
_SKIP_KINDS = frozenset({"loop_started", "loop_completed", "loop_suspended"})

# Coalesce high-frequency stream deltas so we don't rewrite events.json per token.
_DELTA_KINDS = frozenset({"thinking_delta", "token_delta"})
_FLUSH_INTERVAL_S = 0.2
_FLUSH_CHARS = 48


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def make_plan_loop_event_observer(
    workspace_root: str,
    task_id: str,
    *,
    turn_id: str | None = None,
) -> Callable[[Any], Awaitable[None]]:
    """Return an async ``on_event`` that appends selected loop events to the task.

    The parameter stays ``Any``: the observer is handed to the harness through an
    injection seam, and the concrete value is an agent-layer ``AgentEvent`` that
    services reads attributes off directly.
    """

    buffer_kind: str | None = None
    buffer_text = ""
    last_flush = 0.0

    def _flush_buffer(*, force: bool = False) -> None:
        nonlocal buffer_kind, buffer_text, last_flush
        if not buffer_kind or not buffer_text:
            buffer_kind = None
            buffer_text = ""
            return
        now = time.monotonic()
        if not force and len(buffer_text) < _FLUSH_CHARS and (now - last_flush) < _FLUSH_INTERVAL_S:
            return
        _append(
            workspace_root,
            task_id,
            buffer_kind,
            {"text": buffer_text, "turn_id": turn_id, "mode": "plan"},
        )
        buffer_kind = None
        buffer_text = ""
        last_flush = now

    async def _observe(event: Any) -> None:  # noqa: ANN401 — AgentEvent
        nonlocal buffer_kind, buffer_text, last_flush
        try:
            kind = getattr(event, "kind", None) or ""
            if kind in _SKIP_KINDS:
                # Still flush any coalesced thinking so the last tokens land.
                _flush_buffer(force=True)
                return
            if kind in _DELTA_KINDS:
                text = str(getattr(event, "text", "") or "")
                if not text:
                    return
                if buffer_kind and buffer_kind != kind:
                    _flush_buffer(force=True)
                buffer_kind = kind
                buffer_text += text
                _flush_buffer(force=False)
                return
            # Non-delta: flush any pending stream first so order stays readable.
            _flush_buffer(force=True)
            payload = _payload_for(event, turn_id=turn_id)
            if payload is None:
                return
            _append(workspace_root, task_id, kind, payload)
        except Exception as exc:
            _LOG.debug(f"[plan-loop-events] observe failed: {exc!r}")

    return _observe


def _payload_for(event: Any, *, turn_id: str | None) -> dict[str, Any] | None:  # noqa: ANN401
    kind = getattr(event, "kind", None)
    if not kind:
        return None
    base: dict[str, Any] = {"turn_id": turn_id, "mode": "plan", "kind": kind}
    if kind == "thinking_delta" or kind == "token_delta":
        base["text"] = str(getattr(event, "text", "") or "")
        return base
    if kind == "tool_call_started":
        base["tool_name"] = str(getattr(event, "tool_name", "") or "")
        base["args_summary"] = str(getattr(event, "args_summary", "") or "")
        return base
    if kind == "tool_call_completed":
        base["tool_name"] = str(getattr(event, "tool_name", "") or "")
        base["result_summary"] = str(getattr(event, "result_summary", "") or "")
        base["ok"] = bool(getattr(event, "ok", True))
        return base
    if kind == "stage_started":
        base["stage"] = str(getattr(event, "stage_name", "") or "")
        base["message"] = f"Stage {base['stage']}" if base["stage"] else "Stage started"
        return base
    if kind == "stage_completed":
        base["stage"] = str(getattr(event, "stage_name", "") or "")
        base["message"] = f"Stage {base['stage']} done" if base["stage"] else "Stage done"
        return base
    if kind == "error":
        base["message"] = str(getattr(event, "message", "") or "error")
        return base
    # Unknown kinds: dump common scalar attrs only (keep events.json small).
    for attr in ("text", "message", "reason", "tool_name", "stage_name"):
        if hasattr(event, attr):
            val = getattr(event, attr)
            if val is not None and val != "":
                base[attr] = val if isinstance(val, (str, int, float, bool)) else str(val)
    return base


def _append(
    workspace_root: str,
    task_id: str,
    event_type: str,
    payload: dict[str, Any],
) -> None:
    from molexp.services.agent_task_store import append_agent_task_events

    append_agent_task_events(
        workspace_root,
        task_id,
        [
            {
                "type": event_type,
                "ts": _now_iso(),
                "payload": payload,
            }
        ],
    )
