"""Tool sources — pluggable suppliers of out-of-package tools.

A :class:`ToolSource` reports what tools it offers for a given
workspace and dispatches calls to them. Native tools live on the
``AgentService``-owned :class:`ToolRegistry`; sources contribute
*extra* tools (MCP, future plugin packs) without leaking provider or
SDK details into core.
"""

from __future__ import annotations

from threading import Lock
from typing import Any, AsyncIterator, Protocol, runtime_checkable

from molexp.agent.tools.spec import ToolContext, ToolResult, ToolSpec


@runtime_checkable
class ToolSource(Protocol):
    """Tool plugin contract."""

    source_name: str

    async def list_tools(self, workspace: Any) -> list[ToolSpec]: ...

    async def call(
        self, name: str, args: dict[str, Any], ctx: ToolContext
    ) -> ToolResult: ...


class UnknownToolSourceError(KeyError):
    """Raised when no source is registered under the given name."""


class _ToolSourceRegistry:
    def __init__(self) -> None:
        self._sources: dict[str, ToolSource] = {}
        self._lock = Lock()

    def register(self, source: ToolSource) -> None:
        with self._lock:
            self._sources[source.source_name] = source

    def unregister(self, source_name: str) -> None:
        with self._lock:
            self._sources.pop(source_name, None)

    def names(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._sources))

    def get(self, source_name: str) -> ToolSource:
        with self._lock:
            source = self._sources.get(source_name)
        if source is None:
            raise UnknownToolSourceError(source_name)
        return source

    def all(self) -> tuple[ToolSource, ...]:
        with self._lock:
            return tuple(self._sources.values())


_default_registry = _ToolSourceRegistry()


def register_tool_source(source: ToolSource) -> None:
    _default_registry.register(source)


def unregister_tool_source(source_name: str) -> None:
    _default_registry.unregister(source_name)


def list_tool_sources() -> tuple[str, ...]:
    return _default_registry.names()


def get_tool_source(source_name: str) -> ToolSource:
    return _default_registry.get(source_name)


def all_tool_sources() -> tuple[ToolSource, ...]:
    return _default_registry.all()


async def discover_source_tools(
    workspace: Any,
) -> AsyncIterator[tuple[str, ToolSpec]]:
    """Yield ``(source_name, spec)`` for every tool from every source.

    Streaming form so the caller can collate or filter without buffering
    all sources in memory at once.
    """

    for source in all_tool_sources():
        for spec in await source.list_tools(workspace):
            yield source.source_name, spec


__all__ = [
    "ToolSource",
    "UnknownToolSourceError",
    "all_tool_sources",
    "discover_source_tools",
    "get_tool_source",
    "list_tool_sources",
    "register_tool_source",
    "unregister_tool_source",
]
