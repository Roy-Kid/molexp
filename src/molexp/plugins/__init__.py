"""Plugin registry for optional molexp capabilities.

Core molexp (workspace + local workflow) has zero optional dependencies.
Heavy backends — remote HPC, AI agent, etc. — are loaded on demand through
this registry so that ``import molexp`` never fails due to a missing package.

Usage::

    from molexp.plugins import registry, Capability

    if registry.is_available(Capability.AGENT):
        service = registry.get(Capability.AGENT)
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class Capability(str, Enum):
    """Extension points that plugins can provide."""

    REMOTE_EXECUTION = "remote_execution"
    EXECUTION_BACKEND = "execution_backend"
    AGENT = "agent"
    TRANSFER = "transfer"


class CapabilityNotAvailable(RuntimeError):
    """Raised when a requested capability is not installed / registered."""

    def __init__(self, cap: Capability, hint: str = "") -> None:
        msg = f"Capability '{cap.value}' is not available."
        if hint:
            msg += f" {hint}"
        super().__init__(msg)
        self.capability = cap


# ── Lazy loaders ────────────────────────────────────────────────────────────
# Each function attempts the import and returns the entry-point object,
# or raises ImportError if the backing package is missing.

_INSTALL_HINTS: dict[Capability, str] = {
    Capability.REMOTE_EXECUTION: "Install with: pip install molexp[remote]",
    Capability.EXECUTION_BACKEND: "Install with: pip install molexp[remote]",
    Capability.AGENT: "Install with: pip install molexp[agent]",
    Capability.TRANSFER: "Install with: pip install molexp[remote]",
}


def _load_remote() -> Any:
    from molexp.plugins.remote import get_remote_plugin  # type: ignore[import-not-found]
    return get_remote_plugin()


def _load_agent() -> Any:
    from molexp.plugins.agent import get_agent_plugin  # type: ignore[import-not-found]
    return get_agent_plugin()


def _load_execution_backend() -> Any:
    from molq import Submitor  # noqa: F401 — verify molq is installed

    from molexp.plugins.remote.backend import ExecutionBackend  # noqa: F401
    from molexp.plugins.remote.molq_backend import MolqBackend

    return MolqBackend


_LOADERS: dict[Capability, Any] = {
    Capability.REMOTE_EXECUTION: _load_remote,
    Capability.EXECUTION_BACKEND: _load_execution_backend,
    Capability.AGENT: _load_agent,
    Capability.TRANSFER: _load_remote,
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
