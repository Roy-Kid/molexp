"""Open MCP toolsets for InteractiveLoop from McpStore + list live tool specs.

Best-effort: invalid / shadowed / unresolved-secret entries are skipped;
a single build or list failure is logged and does not abort the turn.

Tool **names** are never hard-coded — they come from the live MCP
``list_tools`` catalog after the toolset is opened.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, cast

from mollog import get_logger

from molexp.agent.ops.protocols import ToolSpec

__all__ = ["list_mcp_tool_specs", "open_mcp_toolsets"]

_LOG = get_logger(__name__)


def open_mcp_toolsets(workspace_root: Path) -> tuple[object, ...]:
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

    toolsets: list[object] = []
    for entry in entries:
        if not entry.valid or entry.shadowed or entry.unresolved_secrets:
            continue
        try:
            resolved = store.resolve(entry)
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


class _ListsTools(Protocol):
    """What the unwrap probe actually establishes: an awaitable ``list_tools``."""

    async def list_tools(self) -> Sequence[object]: ...


def _unwrap_mcp_listable(toolset: object) -> tuple[_ListsTools | None, str]:
    """Return ``(object_with_list_tools, name_prefix)`` for a toolset."""
    prefix = str(getattr(toolset, "prefix", "") or "")
    current: object | None = toolset
    # Walk WrapperToolset.wrapped until we find list_tools (MCPToolset).
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if callable(getattr(current, "list_tools", None)):
            return cast("_ListsTools", current), prefix
        current = getattr(current, "wrapped", None)
    return None, prefix


async def list_mcp_tool_specs(toolsets: tuple[object, ...]) -> tuple[ToolSpec, ...]:
    """Enumerate tools from openable MCP toolsets (runtime catalog only).

    Each toolset is entered briefly via its own ``list_tools`` (which
    typically does ``async with self``). Prefixed toolsets get
    ``{prefix}_{name}`` so names match what the agent sees at call time.
    """
    specs: list[ToolSpec] = []
    for toolset in toolsets:
        listable, prefix = _unwrap_mcp_listable(toolset)
        if listable is None:
            _LOG.debug(f"[interactive.mcp] no list_tools on toolset {type(toolset).__name__}")
            continue
        source = prefix or str(getattr(toolset, "id", "") or type(toolset).__name__)
        try:
            tools = await listable.list_tools()
        except Exception as exc:
            _LOG.warning(f"[interactive.mcp] list_tools failed for {source!r}: {exc!r}")
            continue
        for tool in tools:
            raw_name = getattr(tool, "name", None)
            if not isinstance(raw_name, str) or not raw_name:
                continue
            name = f"{prefix}_{raw_name}" if prefix else raw_name
            desc = getattr(tool, "description", None)
            specs.append(
                ToolSpec(
                    name=name,
                    description=desc if isinstance(desc, str) else "",
                    source=source,
                )
            )
    return tuple(specs)
