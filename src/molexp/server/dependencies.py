"""Dependency injection for MolExp API.

This module provides FastAPI dependencies for:
- Workspace instances
- WorkspaceFolderStore
- Configuration settings

Using dependency injection instead of global variables improves:
- Testability (easy to mock dependencies)
- Separation of concerns
- Configuration management
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from molcfg import Config, ConfigLoader, DictSource, Source
from pydantic import BaseModel

from molexp.workspace import Workspace

if TYPE_CHECKING:
    from molexp.server.agent_runtime import AgentSessionRegistry

_SERVER_DEFAULTS: dict[str, Any] = {
    "workspace_path": "",
    "debug": False,
}


class Settings(BaseModel):
    """Application settings loaded via molcfg."""

    workspace_path: str = ""
    debug: bool = False

    @classmethod
    def from_config(cls, config: Config | None = None) -> Settings:
        """Create settings from a molcfg Config."""
        if config is None:
            config = _load_server_config()
        return cls(
            workspace_path=str(config.get("workspace_path", "")),
            debug=bool(config.get("debug", False)),
        )

    def get_workspace_path(self) -> Path:
        """Get workspace path, defaulting to current directory."""
        if self.workspace_path:
            return Path(self.workspace_path)
        return Path.cwd()


def _load_server_config() -> Config:
    """Load server configuration from defaults + optional molexp.toml."""
    sources: list[Source] = [DictSource(_SERVER_DEFAULTS)]
    config_file = Path.cwd() / "molexp.toml"
    if config_file.exists():
        from molcfg import TomlFileSource

        sources.append(TomlFileSource(str(config_file)))
    return ConfigLoader(sources).load()


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings.from_config()


# Cache key is ``(kind, identifier)`` — for ``kind == "local"`` the identifier
# is the resolved absolute path; for ``kind == "remote"`` it is the
# ``WorkspaceTarget`` name (registered via the workspace-targets registry).
_workspace_cache: dict[tuple[str, str], Workspace] = {}


def get_workspace():  # noqa: ANN201
    """FastAPI dependency to get a cached Workspace instance.

    The cache key is ``(kind, identifier)`` — local-vs-remote workspaces
    coexist without collision.  Repeated requests against the same active
    workspace reuse the same instance.
    """
    kind, identifier = _active_workspace_key()
    cache_key = (kind, identifier)
    if cache_key not in _workspace_cache:
        if kind == "remote":
            from molexp.server.workspace_targets import (
                target_to_filesystem_for_workspace_target,
            )

            registry = get_workspace_target_registry()
            try:
                target = registry.get(identifier)
            except KeyError as exc:
                raise KeyError(
                    f"active workspace target {identifier!r} no longer registered"
                ) from exc
            fs = target_to_filesystem_for_workspace_target(target)
            _workspace_cache[cache_key] = Workspace(target.root_path, fs=fs)
        else:
            _workspace_cache[cache_key] = Workspace(Path(identifier))
    return _workspace_cache[cache_key]


def reset_workspace_cache() -> None:
    """Clear the workspace cache (for testing or workspace switching)."""
    _workspace_cache.clear()


# ============================================================================
# Workspace-target registry (server-process scope)
# ============================================================================


_workspace_target_registry: object | None = None  # lazy singleton; typed via accessor


def get_workspace_target_registry():  # noqa: ANN201 — return type stated in docstring
    """Return the process-singleton :class:`WorkspaceTargetRegistry`.

    The registry is server-process scope (lives at
    ``~/.molexp/workspace_targets.json`` in production) so it must
    exist *before* any workspace is open.  Tests should override this
    dependency via ``app.dependency_overrides`` rather than touching
    the singleton.
    """
    global _workspace_target_registry
    from molexp.server.workspace_targets import WorkspaceTargetRegistry

    if _workspace_target_registry is None:
        _workspace_target_registry = WorkspaceTargetRegistry()
    return _workspace_target_registry


def reset_workspace_target_registry() -> None:
    """Drop the process singleton (test fixture support)."""
    global _workspace_target_registry
    _workspace_target_registry = None


def get_remote_fs_factory():  # noqa: ANN201
    """Return the FS-factory callable used by the workspace-targets probe.

    Separated as a FastAPI dependency so tests can substitute a fake
    factory via ``app.dependency_overrides[get_remote_fs_factory]``.
    """
    from molexp.server.workspace_targets import (
        target_to_filesystem_for_workspace_target,
    )

    return target_to_filesystem_for_workspace_target


# ============================================================================
# Workspace Folder Store (in-memory singleton)
# ============================================================================


class WorkspaceFolderStore:
    """In-memory storage for workspace folders.

    Thread-safe storage for tracking added workspace folders.
    Each folder has an ID, path, name, and timestamp.
    """

    def __init__(self) -> None:
        self._folders: dict[str, dict[str, Any]] = {}

    def add(self, folder_id: str, path: str, name: str, added_at: str) -> None:
        """Add a folder to the store."""
        self._folders[folder_id] = {
            "id": folder_id,
            "path": path,
            "name": name,
            "added_at": added_at,
        }

    def remove(self, folder_id: str) -> bool:
        """Remove a folder from the store.

        Returns:
            True if folder was removed, False if not found
        """
        if folder_id in self._folders:
            del self._folders[folder_id]
            return True
        return False

    def get(self, folder_id: str) -> dict[str, Any] | None:
        """Get a folder by ID."""
        return self._folders.get(folder_id)

    def list_all(self) -> list[dict[str, Any]]:
        """List all folders."""
        return list(self._folders.values())

    def find_by_path(self, path: Path) -> dict[str, Any] | None:
        """Find a folder by its path."""
        resolved = path.resolve()
        for folder in self._folders.values():
            if Path(folder["path"]).resolve() == resolved:
                return folder
        return None


# Global singleton instance
_workspace_folder_store: WorkspaceFolderStore | None = None
_workspace_path_override: Path | None = None


def get_workspace_folder_store() -> WorkspaceFolderStore:
    """FastAPI dependency to get WorkspaceFolderStore instance.

    Usage:
        @app.post("/api/workspace/folders")
        def add_folder(store: WorkspaceFolderStore = Depends(get_workspace_folder_store)):
            ...
    """
    global _workspace_folder_store
    if _workspace_folder_store is None:
        _workspace_folder_store = WorkspaceFolderStore()
    return _workspace_folder_store


def reset_workspace_folder_store() -> None:
    """Reset the workspace folder store (for testing)."""
    global _workspace_folder_store
    _workspace_folder_store = None


# ============================================================================
# Agent session runtime (server-process singleton)
# ============================================================================
#
# A process-singleton mirroring ``get_workspace_folder_store`` rather than
# ``app.state``: the relit session routes in ``routes/agent.py`` are plain
# functions called directly by ``agent_tasks.py`` (not request-scoped FastAPI
# endpoints), so they need a callable accessor, not a ``request``. The same
# accessor doubles as a FastAPI dependency; tests reset via
# ``reset_agent_runtime()`` (which cancels any in-flight turns).

_agent_runtime_registry: AgentSessionRegistry | None = None


def get_agent_runtime() -> AgentSessionRegistry:
    """Return the process-singleton :class:`AgentSessionRegistry`.

    Usable both as a FastAPI dependency (``Depends(get_agent_runtime)``) and as
    a plain accessor from the directly-called session routes. Lazily created on
    first use; the app lifespan cancels its in-flight turns on shutdown via
    :func:`reset_agent_runtime`.
    """
    global _agent_runtime_registry
    if _agent_runtime_registry is None:
        from molexp.server.agent_runtime import AgentSessionRegistry

        _agent_runtime_registry = AgentSessionRegistry()
    return _agent_runtime_registry


async def reset_agent_runtime() -> None:
    """Cancel every in-flight turn and drop the registry singleton.

    Awaited by the app lifespan on shutdown and by test fixtures for isolation.
    """
    global _agent_runtime_registry
    if _agent_runtime_registry is not None:
        await _agent_runtime_registry.aclose()
        _agent_runtime_registry = None


_workspace_descriptor_override: str | None = None


def set_workspace_path_override(path: Path | None) -> None:
    """Activate a local workspace path.

    Clears the descriptor override (mutual exclusion), drains workspace-bound
    subscribers, and resets the workspace cache.
    """
    global _workspace_path_override, _workspace_descriptor_override
    _drain_workspace_subscribers()
    _workspace_path_override = path
    _workspace_descriptor_override = None
    reset_workspace_cache()


def set_active_workspace_descriptor(name: str | None) -> None:
    """Activate a remote ``WorkspaceTarget`` by name.

    Clears the path override (mutual exclusion), drains workspace-bound
    subscribers, and resets the workspace cache.
    """
    global _workspace_path_override, _workspace_descriptor_override
    _drain_workspace_subscribers()
    _workspace_descriptor_override = name
    _workspace_path_override = None
    reset_workspace_cache()


def get_workspace_path() -> Path:
    """Resolve the active *local* workspace path.

    Raises :class:`RuntimeError` if the active workspace is remote.
    """
    if _workspace_descriptor_override is not None:
        raise RuntimeError(
            "Active workspace is remote — no local path; "
            "use _active_workspace_key() / get_workspace() instead."
        )
    if _workspace_path_override is not None:
        return _workspace_path_override
    settings = get_settings()
    return settings.get_workspace_path()


def _active_workspace_key() -> tuple[str, str]:
    """Compute the (kind, identifier) key of the active workspace."""
    if _workspace_descriptor_override is not None:
        return ("remote", _workspace_descriptor_override)
    if _workspace_path_override is not None:
        return ("local", str(_workspace_path_override.resolve()))
    settings = get_settings()
    return ("local", str(settings.get_workspace_path().resolve()))


# ============================================================================
# Workspace-switch subscriber drain
# ============================================================================
#
# Long-lived workspace-bound resources (SSE streams, file watchers) register a
# closer that gets awaited *before* the cache is reset on a switch.  Any
# closer may return an awaitable; the drain helper runs sync-callables
# directly and awaits async ones via ``asyncio.run`` (creating a fresh loop)
# or ``loop.run_until_complete`` if one is already running.


_active_workspace_subscribers: list = []


def register_workspace_subscriber(closer) -> None:  # noqa: ANN001
    """Register a closer to be invoked on the next workspace switch.

    Closers are one-shot: drained and discarded on switch.  Callers that
    want to re-subscribe must register again after the switch completes.
    """
    _active_workspace_subscribers.append(closer)


def _drain_workspace_subscribers() -> None:
    """Invoke and clear every registered subscriber.

    Sync callables are called directly; async callables are awaited via a
    fresh event loop (or the running loop's ``run_until_complete``).
    """
    import asyncio
    import inspect

    closers = list(_active_workspace_subscribers)
    _active_workspace_subscribers.clear()
    for closer in closers:
        try:
            result = closer()
        except Exception:
            # Closer is best-effort; one failing closer must not block the switch.
            continue
        if inspect.isawaitable(result):
            try:
                asyncio.run(_await_one(result))
            except RuntimeError:
                # Already inside an event loop — run synchronously.
                loop = asyncio.get_event_loop()
                loop.run_until_complete(_await_one(result))


async def _await_one(awaitable) -> None:  # noqa: ANN001
    import contextlib

    # Best-effort drain — swallow per-closer failures.
    with contextlib.suppress(Exception):
        await awaitable
