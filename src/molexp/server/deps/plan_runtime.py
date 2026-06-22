"""Plan-task runtime (server-process singleton).

Mirrors :mod:`molexp.server.deps.agent_runtime`: the sole owner of the mutable
``_plan_runtime_registry`` global, exposed as a callable accessor (usable as a
FastAPI dependency) and reset on lifespan shutdown / by test fixtures (which
cancels any in-flight plan tasks).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from molexp.server.plan_runtime import PlanTaskRegistry

_plan_runtime_registry: PlanTaskRegistry | None = None


def get_plan_runtime() -> PlanTaskRegistry:
    """Return the process-singleton :class:`PlanTaskRegistry` (lazily created)."""
    global _plan_runtime_registry
    if _plan_runtime_registry is None:
        from molexp.server.plan_runtime import PlanTaskRegistry

        _plan_runtime_registry = PlanTaskRegistry()
    return _plan_runtime_registry


async def reset_plan_runtime() -> None:
    """Cancel every in-flight plan task and drop the registry singleton."""
    global _plan_runtime_registry
    if _plan_runtime_registry is not None:
        await _plan_runtime_registry.aclose()
        _plan_runtime_registry = None
