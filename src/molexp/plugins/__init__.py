"""Plugin registry for optional molexp capabilities.

Core molexp (workspace + local workflow) has zero optional dependencies.
Heavy backends — AI agent, etc. — are loaded on demand through this
registry so that ``import molexp`` never fails due to a missing package.

Plugin naming convention: ``{category}_{implementation}``

- ``submit_molq``  — Job submission via molq schedulers
- ``agent_pydanticai`` — AI agent via PydanticAI

Usage::

    from molexp.plugins import registry, Capability

    if registry.is_available(Capability.AGENT):
        service = registry.get(Capability.AGENT)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from molexp.plugins.submit_molq.metadata import supported_schedulers


class Capability(str, Enum):
    """Extension points that plugins can provide."""

    AGENT = "agent"


class CapabilityNotAvailable(RuntimeError):
    """Raised when a requested capability is not installed / registered."""

    def __init__(self, cap: Capability, hint: str = "") -> None:
        msg = f"Capability '{cap.value}' is not available."
        if hint:
            msg += f" {hint}"
        super().__init__(msg)
        self.capability = cap


@dataclass(frozen=True)
class UiPluginDescriptor:
    """Descriptor for frontend-facing plugin manifests."""

    id: str
    title: str
    description: str = ""
    ui_module: str | None = None
    capabilities: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Lazy loaders ────────────────────────────────────────────────────────────
# Each function attempts the import and returns the entry-point object,
# or raises ImportError if the backing package is missing.

_INSTALL_HINTS: dict[Capability, str] = {
    Capability.AGENT: "Install with: pip install molexp[agent]",
}


def _load_agent() -> Any:
    from molexp.plugins.agent_pydanticai import get_agent_plugin  # type: ignore[import-not-found]
    return get_agent_plugin()


_LOADERS: dict[Capability, Any] = {
    Capability.AGENT: _load_agent,
}


# ── Registry singleton ─────────────────────────────────────────────────────


class PluginRegistry:
    """Discovers and caches optional capabilities at runtime."""

    def __init__(self) -> None:
        self._cache: dict[Capability, Any] = {}
        self._unavailable: set[Capability] = set()

    def is_available(self, cap: Capability) -> bool:
        """Check whether *cap* can be loaded (dependency installed)."""
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

    def get(self, cap: Capability) -> Any:
        """Return the loaded plugin for *cap*, or raise."""
        if cap in self._cache:
            return self._cache[cap]
        if self.is_available(cap):
            return self._cache[cap]
        raise CapabilityNotAvailable(
            cap, _INSTALL_HINTS.get(cap, "")
        )

    def available_capabilities(self) -> list[Capability]:
        """Return all capabilities whose backing packages are installed."""
        return [c for c in Capability if self.is_available(c)]

    def reset(self) -> None:
        """Clear caches (useful in tests)."""
        self._cache.clear()
        self._unavailable.clear()


registry = PluginRegistry()


def discover_ui_plugins() -> list[UiPluginDescriptor]:
    """Return frontend plugin manifests for all available UI integrations."""
    plugins = [
        UiPluginDescriptor(
            id="core",
            title="Core Workspace UI",
            description="Built-in Molexp workspace renderers and previews.",
            ui_module="core",
            capabilities=("workspace", "renderers", "file_previews"),
        )
    ]

    schedulers = supported_schedulers()
    if schedulers:
        plugins.append(
            UiPluginDescriptor(
                id="molq",
                title="Molq",
                description="Scheduler-aware run viewers and monitor surfaces for molq-backed runs.",
                ui_module="molq",
                capabilities=("submit", "monitor", "scheduler_inspector"),
                metadata={"schedulers": list(schedulers)},
            )
        )

    return plugins


__all__ = [
    "Capability",
    "CapabilityNotAvailable",
    "PluginRegistry",
    "UiPluginDescriptor",
    "discover_ui_plugins",
    "registry",
]
