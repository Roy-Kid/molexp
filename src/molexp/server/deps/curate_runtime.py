"""Curate-task runtime (server-process singleton).

Mirrors :mod:`molexp.server.deps.plan_runtime`: the sole owner of the mutable
``_curate_runtime_registry`` global, exposed as a callable accessor (usable as a
FastAPI dependency) and reset on lifespan shutdown / by test fixtures (which
cancels any in-flight curate tasks).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from molexp.server.curate_runtime import CurateTaskRegistry

_curate_runtime_registry: CurateTaskRegistry | None = None


def get_curate_runtime() -> CurateTaskRegistry:
    """Return the process-singleton :class:`CurateTaskRegistry` (lazily created)."""
    global _curate_runtime_registry
    if _curate_runtime_registry is None:
        from molexp.server.curate_runtime import CurateTaskRegistry

        _curate_runtime_registry = CurateTaskRegistry()
    return _curate_runtime_registry


async def reset_curate_runtime() -> None:
    """Cancel every in-flight curate task and drop the registry singleton."""
    global _curate_runtime_registry
    if _curate_runtime_registry is not None:
        await _curate_runtime_registry.aclose()
        _curate_runtime_registry = None
