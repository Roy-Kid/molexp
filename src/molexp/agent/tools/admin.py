"""Read-only introspection helpers for the native-tool catalog.

The settings UI calls into this to render the native-tool list
without instantiating an :class:`AgentService` (so it works even
when no provider is configured).
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import Any

from molexp.agent.tools import native as native_pkg
from molexp.agent.tools.registry import get_native_spec, is_native_tool


def describe_native_tools() -> list[dict[str, Any]]:
    """Walk ``molexp.agent.tools.native`` and return a row per tagged tool.

    Each row mirrors the ``@native_tool`` ``ToolSpec`` plus the
    callable's parameter signature (excluding ``args`` / ``ctx``,
    which the dispatcher injects).
    """

    rows: list[dict[str, Any]] = []
    for module_info in pkgutil.iter_modules(
        native_pkg.__path__, prefix=f"{native_pkg.__name__}."
    ):
        module = importlib.import_module(module_info.name)
        for attr_name in dir(module):
            obj = getattr(module, attr_name)
            if not is_native_tool(obj):
                continue
            spec = get_native_spec(obj)
            sig = inspect.signature(obj)
            params: list[dict[str, Any]] = []
            for pname, p in sig.parameters.items():
                if pname in ("args", "ctx", "context"):
                    continue
                params.append(
                    {
                        "name": pname,
                        "annotation": _format_annotation(p.annotation),
                        "required": p.default is inspect.Parameter.empty,
                    }
                )
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
