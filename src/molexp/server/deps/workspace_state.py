"""Active-workspace state: cache, overrides, and switch-subscriber drain.

This module is the **single owner** of the mutable active-workspace globals
(``_workspace_cache``, ``_workspace_path_override``,
``_workspace_descriptor_override``, ``_active_workspace_subscribers``).
All writes happen through the functions defined here so singleton semantics
hold across both import paths (this module and the
``molexp.server.dependencies`` facade).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from molexp.server.deps.config import get_settings

if TYPE_CHECKING:
    from molexp.workspace import Workspace

_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})

# Cache key is ``(kind, identifier)`` — for ``kind == "local"`` the identifier
# is the resolved absolute path; for ``kind == "remote"`` it is the
# ``WorkspaceTarget`` name (registered via the workspace-targets registry).
_workspace_cache: dict[tuple[str, str], Workspace] = {}

_workspace_path_override: Path | None = None
_workspace_descriptor_override: str | None = None


def reset_workspace_cache() -> None:
    """Clear the workspace cache (for testing or workspace switching)."""
    _workspace_cache.clear()


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
