"""Agent session runtime (server-process singleton).

A process-singleton mirroring ``get_workspace_folder_store`` rather than
``app.state``: the relit session routes in ``routes/agent.py`` are plain
functions called directly by ``agent_tasks.py`` (not request-scoped FastAPI
endpoints), so they need a callable accessor, not a ``request``. The same
accessor doubles as a FastAPI dependency; tests reset via
``reset_agent_runtime()`` (which cancels any in-flight turns).

This module is the **single owner** of the mutable ``_agent_runtime_registry``
global; tests that substitute a registry should monkeypatch *this* module,
not the ``molexp.server.dependencies`` facade.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from molexp.server.agent_runtime import AgentSessionRegistry

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
