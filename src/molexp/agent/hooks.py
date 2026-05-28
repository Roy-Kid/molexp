"""The typed hook registry.

:class:`HookRegistry` maps a :class:`HookPoint` to an ordered list of
async handlers. The harness fires hooks at five well-defined points; a
mode or a policy registers handlers to observe or steer the run.

:class:`HookContext` is the typed payload every handler receives. It is
a frozen-pydantic model carrying the hook point plus optional context
fields (stage name, gate, free-form ``payload``). Handlers optionally
return a typed value; :meth:`HookRegistry.dispatch` collects the
non-``None`` returns in registration order so callers can inspect them
(e.g. a ``before_approval`` handler returning a
:class:`~molexp.agent.review.ReviewDecision`).

No ``pydantic_ai`` / ``pydantic_graph`` imports.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "HookContext",
    "HookHandler",
    "HookPoint",
    "HookRegistry",
]


class HookPoint(StrEnum):
    """The five lifecycle points the harness fires hooks at."""

    before_stage = "before_stage"
    after_stage = "after_stage"
    before_approval = "before_approval"
    before_compact = "before_compact"
    before_model_call = "before_model_call"


class HookContext(BaseModel):
    """Typed payload passed to every hook handler.

    Attributes:
        point: Which :class:`HookPoint` is firing.
        stage_name: The stage name, for ``before_stage`` / ``after_stage``.
        gate: The approval-gate name, for ``before_approval``.
        payload: Free-form structured context the harness attaches —
            e.g. the review view's summary, the compaction plan's
            token count.
    """

    model_config = ConfigDict(frozen=True)

    point: HookPoint
    stage_name: str = ""
    gate: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)


# A handler takes the typed context and optionally returns a typed
# value (a ReviewDecision, a veto dict, …) or ``None``.
HookHandler = Callable[[HookContext], Awaitable[Any]]


class HookRegistry:
    """Ordered, per-point registry of async hook handlers.

    Plain runtime class — it holds live callables. :meth:`register`
    appends; :meth:`dispatch` fires every handler for a point in
    registration order and returns the tuple of non-``None`` results.
    """

    def __init__(self) -> None:
        self._handlers: dict[HookPoint, list[HookHandler]] = {point: [] for point in HookPoint}

    def register(self, point: HookPoint, handler: HookHandler) -> None:
        """Append ``handler`` to ``point``'s ordered handler list."""
        self._handlers[point].append(handler)

    def handlers(self, point: HookPoint) -> tuple[HookHandler, ...]:
        """Return the registered handlers for ``point``, in order."""
        return tuple(self._handlers[point])

    async def dispatch(self, point: HookPoint, context: HookContext) -> tuple[Any, ...]:
        """Fire every handler for ``point`` in order; collect non-``None`` results.

        Handlers run sequentially (registration order is load-bearing —
        a later handler may depend on an earlier side effect). A handler
        returning ``None`` contributes nothing to the result tuple.
        """
        results: list[Any] = []
        for handler in self._handlers[point]:
            outcome = await handler(context)
            if outcome is not None:
                results.append(outcome)
        return tuple(results)
