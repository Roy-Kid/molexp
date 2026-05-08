"""Plugin registry for optional molexp capabilities.

Core molexp (workspace + local workflow) has zero optional dependencies.
Heavy backends are loaded on demand through this registry so that
``import molexp`` never fails due to a missing package.

Plugin naming convention: ``{category}_{implementation}``

- ``submit_molq`` — Job submission via molq schedulers
- ``gh`` — GitHub API client (GraphQL + REST) backed by httpx

The agent capability is now provided directly by ``molexp.agent``
(see :class:`molexp.agent.AgentRunner`); the old plugin-registry
indirection for agent / coding-agent providers was removed alongside
the legacy ``AgentService`` / ``CodingAgentClient`` surfaces.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum, StrEnum

from molexp.plugins.cli import (
    CLI_PLUGIN_API_VERSION,
    CliPlugin,
    discover_cli_plugins,
)
from molexp.plugins.ui import discover_ui_plugin_dirs


class Capability(StrEnum):
    """Extension points that plugins can provide."""

    GH = "gh"


class CapabilityNotAvailable(RuntimeError):
    """Raised when a requested capability is not installed / registered."""

    def __init__(self, cap: Capability, hint: str = "") -> None:
        msg = f"Capability '{cap.value}' is not available."
        if hint:
            msg += f" {hint}"
        super().__init__(msg)
        self.capability = cap


_INSTALL_HINTS: dict[Capability, str] = {
    Capability.GH: "Install with: pip install httpx",
}


def _load_gh() -> type:
    import httpx

    from molexp.plugins.gh import GitHubClient

    return GitHubClient


_LOADERS: dict[Capability, Callable[[], type]] = {
    Capability.GH: _load_gh,
}


class PluginRegistry:
    """Discovers and caches optional capabilities at runtime."""

    def __init__(self) -> None:
        self._cache: dict[Capability, type] = {}
        self._unavailable: set[Capability] = set()

    def is_available(self, cap: Capability) -> bool:
        if cap in self._cache:
            return True
        if cap in self._unavailable:
            return False
        try:
            self._cache[cap] = _LOADERS[cap]()
            return True
        except (ImportError, KeyError):
            self._unavailable.add(cap)
            return False

    def get(self, cap: Capability) -> type:
        if cap in self._cache:
            return self._cache[cap]
        if self.is_available(cap):
            return self._cache[cap]
        raise CapabilityNotAvailable(cap, _INSTALL_HINTS.get(cap, ""))

    def available_capabilities(self) -> list[Capability]:
        return [c for c in Capability if self.is_available(c)]

    def reset(self) -> None:
        self._cache.clear()
        self._unavailable.clear()


registry = PluginRegistry()


__all__ = [
    "CLI_PLUGIN_API_VERSION",
    "Capability",
    "CapabilityNotAvailable",
    "CliPlugin",
    "PluginRegistry",
    "discover_cli_plugins",
    "discover_ui_plugin_dirs",
    "registry",
]
