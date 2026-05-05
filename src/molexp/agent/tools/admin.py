"""Read-only introspection helpers for the native-tool catalog.

The settings UI calls into this to render the native-tool list
without instantiating an :class:`AgentService` (so it works even
when no provider is configured).
"""

from __future__ import annotations

import inspect
from typing import Any

from molexp.agent.tools.registry import iter_native_tools


def describe_native_tools() -> list[dict[str, Any]]:
    """Return a row per ``@native_tool`` in :mod:`molexp.agent.tools.native`.

    Each row mirrors the tool's ``ToolSpec`` plus the callable's parameter
    signature (excluding ``args`` / ``ctx``, which the dispatcher injects).
    """

    rows: list[dict[str, Any]] = []
    for spec, fn in iter_native_tools():
        sig = inspect.signature(fn)
        params: list[dict[str, Any]] = [
            {
                "name": pname,
                "annotation": _format_annotation(p.annotation),
                "required": p.default is inspect.Parameter.empty,
            }
            for pname, p in sig.parameters.items()
            if pname not in ("args", "ctx", "context")
        ]
        rows.append(
            {
                "name": spec.name,
                "description": spec.description,
                "parameters": params,
                "requires_approval": spec.requires_approval,
                "category": spec.category,
                "mutates": spec.mutates,
                "source": spec.source,
            }
        )
    return rows


def _format_annotation(annotation: Any) -> str:
    if annotation is inspect.Parameter.empty:
        return "Any"
    return getattr(annotation, "__name__", repr(annotation))


__all__ = ["describe_native_tools"]
