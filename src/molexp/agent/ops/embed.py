"""Chat embed tools — structured artifacts for molplot / molvis in the UI.

Tools return a JSON envelope so the pydantic-ai router can peel off
``artifacts`` into :class:`ToolResultChunk` (full payload, not truncated
``result_summary``). The conversation UI renders ``kind=plot|structure|table``.
"""

from __future__ import annotations

import json
from typing import Any

__all__ = [
    "EMBED_MARKER",
    "encode_embed_result",
    "parse_tool_result_payload",
]

#: Optional prefix for greppable logs; envelope is pure JSON when present alone.
EMBED_MARKER = "MOLEXP_EMBED_V1"


def encode_embed_result(
    *,
    summary: str,
    artifacts: list[dict[str, Any]],
    ok: bool = True,
) -> str:
    """Serialize a tool return that carries inline conversation artifacts."""
    body = {
        "ok": ok,
        "summary": summary,
        "artifacts": artifacts,
        "marker": EMBED_MARKER,
    }
    return json.dumps(body, ensure_ascii=False, default=str)


def parse_tool_result_payload(content: str) -> tuple[str, bool, list[dict[str, Any]]]:
    """Parse a tool return string into ``(summary, ok, artifacts)``.

    Non-embed returns pass through as ``(content, True, [])`` (caller may
    still set ok from RetryPrompt).
    """
    text = (content or "").strip()
    if not text:
        return "", True, []
    # Allow accidental prefix noise before JSON.
    start = text.find("{")
    if start < 0:
        return text, True, []
    try:
        data = json.loads(text[start:])
    except json.JSONDecodeError:
        return text, True, []
    if not isinstance(data, dict):
        return text, True, []
    if data.get("marker") != EMBED_MARKER and "artifacts" not in data:
        return text, True, []
    arts_raw = data.get("artifacts") or []
    artifacts: list[dict[str, Any]] = []
    if isinstance(arts_raw, list):
        for item in arts_raw:
            if isinstance(item, dict) and item.get("kind"):
                artifacts.append(dict(item))
    summary = str(data.get("summary") or data.get("message") or "ok")
    ok = bool(data.get("ok", True))
    return summary, ok, artifacts
