"""``McpToolSource`` — :class:`molexp.agent.ToolSource` for MCP servers.

Bridges the harness tool-source contract (``list_tools`` + ``call``)
to the workspace-scoped MCP store at :mod:`molexp.agent.mcp.store`.
Each MCP tool is exposed under the ``mcp:<server>.<tool>`` name
convention.

The source is intentionally stateless: it re-reads the store each
``list_tools`` call so admin-route edits land without restarts. Tool
calls connect to the underlying MCP server fresh per invocation;
keeping a single long-lived connection per server is left to a later
phase (R5 — survives without it).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Protocol

from molexp._typing import JSONMapping, JSONValue, TaskOutput
from molexp.agent.mcp.probe import (
    PROBE_TIMEOUT_SECONDS,
    _build_pydantic_ai_server,
)
from molexp.agent.mcp.store import (
    McpScope,
    McpServerEntry,
    McpStore,
    UnresolvedSecretError,
)
from molexp.agent.tools.spec import ToolContext, ToolResult, ToolSpec
from molexp.agent.types import AgentFailure, FailureKind


class _SubsystemStoreLike(Protocol):
    """Subsystem-store handle exposing a ``.dir()`` accessor."""

    def dir(self) -> Path: ...


class _WorkspaceLike(Protocol):
    """Duck-typed shape of ``molexp.workspace.workspace.Workspace`` reached by the MCP source.

    The MCP source needs only ``subsystem_store(name)`` (preferred) plus a
    ``root`` fallback for legacy callers. Defined as a Protocol so the
    agent layer doesn't take a hard dependency on the workspace class.
    """

    root: Path

    def subsystem_store(self, kind: str) -> _SubsystemStoreLike: ...


SOURCE_NAME = "mcp"


def _qualified_name(server_name: str, tool_name: str) -> str:
    return f"mcp:{server_name}.{tool_name}"


def _split_name(name: str) -> tuple[str, str]:
    if not name.startswith("mcp:"):
        raise ValueError(f"not an MCP tool name: {name!r}")
    rest = name[len("mcp:") :]
    server, _, tool = rest.partition(".")
    if not server or not tool:
        raise ValueError(f"malformed MCP tool name: {name!r}")
    return server, tool


def _err(kind: FailureKind, message: str, **detail: JSONValue) -> ToolResult:
    return ToolResult(
        ok=False,
        error=AgentFailure(kind=kind, message=message, detail=dict(detail)),
    )


class McpToolSource:
    """Workspace-scoped MCP tool source.

    Holds a workspace ``root`` so admin routes that mount the source
    can stay stateless; ``list_tools`` re-reads the store on every
    call to pick up live edits. The default constructor uses the
    process-wide source (registered at import time) and resolves
    ``root`` from the ``ToolContext.workspace`` argument at call time.
    """

    source_name = SOURCE_NAME

    def __init__(self, root: Path | None = None) -> None:
        self._root = root

    async def list_tools(self, workspace: _WorkspaceLike) -> list[ToolSpec]:
        store = self._store_for(workspace)
        if store is None:
            return []
        out: list[ToolSpec] = []
        for entry in store.list():
            if not entry.valid or entry.shadowed:
                continue
            for spec in await self._list_one(store, entry):
                out.append(spec)
        return out

    async def call(self, name: str, args: JSONMapping, ctx: ToolContext) -> ToolResult:
        try:
            server_name, tool_name = _split_name(name)
        except ValueError as exc:
            return _err(FailureKind.TOOL_NOT_FOUND, str(exc))
        if ctx.workspace is None:
            return _err(
                FailureKind.TOOL_NOT_FOUND,
                "MCP source has no workspace bound",
                tool=name,
            )
        store = self._store_for(ctx.workspace)
        if store is None:
            return _err(
                FailureKind.TOOL_NOT_FOUND,
                "MCP source has no workspace bound",
                tool=name,
            )
        entry = _find_entry(store, server_name)
        if entry is None or not entry.valid or entry.shadowed:
            return _err(
                FailureKind.TOOL_NOT_FOUND,
                f"MCP server '{server_name}' not registered or invalid",
                tool=name,
            )
        try:
            resolved = store.resolve(entry)
        except UnresolvedSecretError as exc:
            return _err(
                FailureKind.TOOL_ERROR,
                f"Missing secrets: {', '.join(exc.keys)}",
                tool=name,
            )
        try:
            server = _build_pydantic_ai_server(resolved, entry.name, entry.scope, store)
        except ImportError:
            return _err(
                FailureKind.TOOL_ERROR,
                "pydantic-ai MCP support unavailable",
                tool=name,
            )

        async def _invoke() -> ToolResult:
            async with server:
                value = await server.call_tool(tool_name, args)
                return ToolResult(ok=True, value=_jsonable(value))

        try:
            return await asyncio.wait_for(_invoke(), timeout=PROBE_TIMEOUT_SECONDS)
        except TimeoutError:
            return _err(
                FailureKind.TOOL_ERROR,
                f"Timeout after {PROBE_TIMEOUT_SECONDS:.0f}s",
                tool=name,
            )
        except Exception as exc:
            return _err(
                FailureKind.TOOL_ERROR,
                f"{type(exc).__name__}: {exc}",
                tool=name,
            )

    def _store_for(self, workspace: _WorkspaceLike) -> McpStore | None:
        # Prefer the workspace API (gives us the canonical
        # <root>/.subsystems/agent.mcp/ path); fall back to a bare root
        # for legacy callers that supplied only a Path.
        if hasattr(workspace, "subsystem_store"):
            return McpStore(workspace.subsystem_store("agent.mcp").dir())
        root = self._root if self._root is not None else getattr(workspace, "root", None)
        if root is None:
            return None
        from pathlib import Path

        from molexp.workspace.subsystem import SUBSYSTEMS_DIRNAME

        return McpStore(Path(root) / SUBSYSTEMS_DIRNAME / "agent.mcp")

    async def _list_one(self, store: McpStore, entry: McpServerEntry) -> list[ToolSpec]:
        try:
            resolved = store.resolve(entry)
        except (UnresolvedSecretError, Exception):
            return []
        try:
            server = _build_pydantic_ai_server(resolved, entry.name, entry.scope, store)
        except (ImportError, Exception):
            return []
        try:
            async with server:
                tools = await asyncio.wait_for(server.list_tools(), timeout=PROBE_TIMEOUT_SECONDS)
        except (TimeoutError, Exception):
            return []

        out: list[ToolSpec] = []
        scope = (
            "user"
            if entry.scope is McpScope.USER
            else "workspace"
            if entry.scope is McpScope.WORKSPACE
            else "builtin"
        )
        for tool in tools:
            out.append(
                ToolSpec(
                    name=_qualified_name(entry.name, tool.name),
                    description=getattr(tool, "description", "") or "",
                    input_schema=getattr(tool, "input_schema", None) or {"type": "object"},
                    source=f"mcp:{scope}",
                    category="mcp",
                    mutates=False,
                    requires_approval=False,
                )
            )
        return out


def _find_entry(store: McpStore, name: str) -> McpServerEntry | None:
    """Resolve a server by name across scopes (workspace shadows user)."""

    for entry in store.list():
        if entry.name == name and not entry.shadowed:
            return entry
    return None


def _jsonable(value: TaskOutput) -> JSONValue:
    """Best-effort coercion so an MCP value can ride a JSON wire format."""

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return repr(value)


__all__ = ["SOURCE_NAME", "McpToolSource"]
