"""Open MCP toolsets for :class:`InteractiveLoop` from :class:`McpStore`.

Best-effort: invalid / shadowed / unresolved-secret entries are skipped;
a single ``build_mcp_server`` failure is logged and does not abort the
turn. Callers pass the returned opaque toolset objects into
``Router.stream_agentic(toolsets=...)`` — pydantic-ai's ``Agent.iter``
enters/exits MCP transports for the duration of the run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mollog import get_logger

__all__ = ["open_mcp_toolsets"]

_LOG = get_logger(__name__)


def open_mcp_toolsets(workspace_root: Path) -> tuple[Any, ...]:
    """Build pydantic-ai MCP toolsets for valid, unshadowed store entries.

    Args:
        workspace_root: Workspace directory whose ``mcp.json`` (and the
            user-home store) are merged by :class:`McpStore`.

    Returns:
        Zero or more opaque toolset objects suitable for
        ``stream_agentic(toolsets=...)``. Never raises for per-entry
        failures — logs a warning and continues.
    """
    from molexp.agent._pydanticai.mcp import build_mcp_server
    from molexp.agent.mcp.store import McpStore

    root = Path(workspace_root)
    try:
        store = McpStore(root)
        entries = store.list()
    except OSError as exc:
        _LOG.warning(f"[interactive.mcp] could not open McpStore at {root}: {exc!r}")
        return ()

    toolsets: list[Any] = []
    for entry in entries:
        if not entry.valid or entry.shadowed or entry.unresolved_secrets:
            continue
        try:
            resolved = store.resolve(entry)
            # Point molexp (and similar) providers at this session workspace
            # so tools without an explicit path still target the right root.
            env = dict(resolved.env) if resolved.env else {}
            if resolved.transport == "stdio":
                env.setdefault("MOLEXP_WORKSPACE", str(root.resolve()))
            toolset = build_mcp_server(
                transport=resolved.transport,
                name=entry.name,
                command=resolved.command or "",
                args=resolved.args,
                env=env or None,
                url=resolved.url or "",
                headers=resolved.headers or None,
            )
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            _LOG.warning(
                f"[interactive.mcp] skip server {entry.name!r} (scope={entry.scope.value}): {exc!r}"
            )
            continue
        toolsets.append(toolset)
    return tuple(toolsets)
