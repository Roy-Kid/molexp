"""Turn-scoped MCP catalog for one ReAct.

Config lives in :mod:`molexp.agent.mcp.store`. This module opens
pydantic-ai toolsets for one turn via a lazy ``build_mcp_server``
import, holds them, and disposes them in LIFO on :meth:`McpCatalog.aclose`.

Turn lifetime is **not** the router keep-alive cache
(:class:`~molexp.agent._pydanticai.router.PydanticAIRouter`): these
handles use ``keep_alive=False`` and must not be fed into
``_mcp_toolsets``.

Best-effort: invalid / shadowed / unresolved-secret entries are skipped;
a single build or list failure is logged and does not abort the turn.

Tool **names** are never hard-coded — they come from the live MCP
``list_tools`` catalog after the toolset is opened.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol, cast

from mollog import get_logger

from molexp.agent.ops.protocols import ToolSpec

__all__ = [
    "McpCatalog",
    "filter_toolsets",
    "list_mcp_tool_specs",
]

_LOG = get_logger(__name__)


def _build_named_toolsets(workspace_root: Path, *, keep_alive: bool) -> list[tuple[str, object]]:
    """Resolve store entries to ``(name, toolset)`` pairs. Never raises."""
    from molexp.agent._pydanticai.mcp import build_mcp_server
    from molexp.agent.mcp.store import McpStore

    root = Path(workspace_root)
    try:
        store = McpStore(root)
        entries = store.list()
    except OSError as exc:
        _LOG.warning(f"[mcp.catalog] could not open McpStore at {root}: {exc!r}")
        return []

    built: list[tuple[str, object]] = []
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
                keep_alive=keep_alive,
            )
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            _LOG.warning(
                f"[mcp.catalog] skip server {entry.name!r} (scope={entry.scope.value}): {exc!r}"
            )
            continue
        built.append((entry.name, toolset))
    return built


class McpCatalog:
    """Opened MCP servers for one ReAct turn.

    Opening is acquisition: :meth:`aclose` is the inverse. The caller
    owns this handle in ``try`` / ``finally``.
    """

    def __init__(self, workspace_root: Path) -> None:
        self._root = Path(workspace_root)
        self._handles: list[tuple[str, object, bool]] = []
        self._opened = False

    @property
    def toolsets(self) -> tuple[object, ...]:
        """Opaque toolset objects for ``stream_agentic(toolsets=...)``."""
        return tuple(toolset for _, toolset, _ in self._handles)

    async def open(self) -> None:
        """Best-effort open from :class:`~molexp.agent.mcp.store.McpStore`.

        Idempotent. Enters each toolset that exposes ``__aenter__`` so
        ``list_specs`` and every outer-loop pass share one session; failed
        enters are dropped, not kept.
        """
        if self._opened:
            return
        self._opened = True
        for name, toolset in _build_named_toolsets(self._root, keep_alive=False):
            entered = False
            aenter = getattr(toolset, "__aenter__", None)
            if callable(aenter):
                try:
                    await aenter()
                    entered = True
                except Exception as exc:
                    _LOG.warning(f"[mcp.catalog] enter {name!r} failed: {exc!r}")
                    continue
            self._handles.append((name, toolset, entered))

    async def list_specs(self) -> tuple[ToolSpec, ...]:
        """Enumerate tools from the open handles (runtime catalog only)."""
        return await list_mcp_tool_specs(self.toolsets)

    async def aclose(self) -> None:
        """LIFO dispose of entered handles. Safe to call more than once."""
        while self._handles:
            name, toolset, entered = self._handles.pop()
            if not entered:
                continue
            aexit = getattr(toolset, "__aexit__", None)
            if not callable(aexit):
                continue
            try:
                await aexit(None, None, None)
            except Exception as exc:
                _LOG.warning(f"[mcp.catalog] dispose {name!r} failed: {exc!r}")
        self._opened = False


def filter_toolsets(
    toolsets: tuple[object, ...],
    *,
    allow: Callable[[str], bool],
) -> tuple[object, ...]:
    """Wrap each toolset so disallowed names never appear or dispatch.

    Uses pydantic-ai ``AbstractToolset.filtered`` when present (lazy —
    no SDK import here). The predicate sees the **wire** name
    (already prefixed on a ``PrefixedToolset``). Toolsets without
    ``filtered`` are passed through unchanged.
    """
    wrapped: list[object] = []
    for toolset in toolsets:
        filtered = getattr(toolset, "filtered", None)
        if not callable(filtered):
            wrapped.append(toolset)
            continue

        def _keep(
            _ctx: object,
            tool_def: object,
            *,
            _allow: Callable[[str], bool] = allow,
        ) -> bool:
            name = getattr(tool_def, "name", "")
            return bool(name) and _allow(str(name))

        wrapped.append(filtered(_keep))
    return tuple(wrapped)


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
    When the caller already entered the catalog for the turn,
    ``list_tools`` reuses that session.
    """
    specs: list[ToolSpec] = []
    for toolset in toolsets:
        listable, prefix = _unwrap_mcp_listable(toolset)
        if listable is None:
            _LOG.debug(f"[mcp.catalog] no list_tools on toolset {type(toolset).__name__}")
            continue
        source = prefix or str(getattr(toolset, "id", "") or type(toolset).__name__)
        try:
            tools = await listable.list_tools()
        except Exception as exc:
            _LOG.warning(f"[mcp.catalog] list_tools failed for {source!r}: {exc!r}")
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
