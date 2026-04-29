"""Read-only introspection helpers for the agent's native tool catalog.

MCP server inspection has moved to :mod:`mcp_store` (multi-scope CRUD)
and :mod:`mcp_probe` (live connection test). This module is now the
single home for native-tool catalog rendering used by the settings UI.
"""

from __future__ import annotations

import inspect
from typing import Any

from .policy import ApprovalPolicy


def describe_native_tools(approval_policy: ApprovalPolicy | None = None) -> list[dict[str, Any]]:
    """Inspect the native tool functions registered with the catalog.

    Returns name/description/parameters/approval-required for each.
    MCP-discovered tools are intentionally excluded — they are
    reported by the MCP server itself at agent.run() time and are
    not visible without an active connection.
    """
    from ._pydantic_ai.catalog import DEFAULT_APPROVAL_TOOLS
    from ._pydantic_ai.workspace_tools import get_all_builtin_tools

    policy = approval_policy or ApprovalPolicy(
        require_approval_for=list(DEFAULT_APPROVAL_TOOLS),
    )

    rows: list[dict[str, Any]] = []
    for fn in get_all_builtin_tools():
        sig = inspect.signature(fn)
        params: list[dict[str, Any]] = []
        for pname, p in sig.parameters.items():
            if pname in ("ctx", "context"):
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
                "name": fn.__name__,
                "description": (inspect.getdoc(fn) or "").strip(),
                "parameters": params,
                "requires_approval": policy.needs_approval(fn.__name__),
                "source": "native",
            }
        )
    return rows


def _format_annotation(annotation: Any) -> str:
    if annotation is inspect.Parameter.empty:
        return "Any"
    return getattr(annotation, "__name__", repr(annotation))
