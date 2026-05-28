"""Declarative ``RepairPolicy`` for the executable pipeline.

A :class:`RepairPolicy` says "when an event of kind X is yielded by a
stage, jump back to stage Y at most N times; on exhaustion route to
terminal Z." Pure data — :func:`~molexp.agent.pipeline.execute_pipeline`
reads it; modes register repairs on their
:class:`~molexp.agent.mode.ModePipeline`.

This is **frozen pydantic** — pure declarative data, no callables,
qualifies as the lower half of the agent-layer "pydantic vs plain
class" rule. The per-mode repair *executor* helpers
(``modes/author/repair.py``, ``modes/run/repair.py``) stay where they
are — :class:`RepairPolicy` is the routing policy, not the executor.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

__all__ = ["RepairPolicy"]


class RepairPolicy(BaseModel):
    """Declarative routing for one repair condition in a pipeline.

    Attributes:
        trigger_event_kind: The :attr:`~molexp.agent.events.AgentEvent.kind`
            literal to listen for (e.g. ``"preflight_failed"``,
            ``"repair_proposed"``). When a stage yields an event whose
            ``kind`` equals this, the policy fires.
        rewind_to: Name of the stage to jump back to when the policy
            fires and the iteration budget is not yet exhausted.
        max_iterations: Maximum number of times the policy may rewind.
            Once this count is reached, the next trigger routes to
            :attr:`on_exhausted` instead.
        on_exhausted: Name of the terminal state (or stage) to route to
            when ``max_iterations`` is reached.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    trigger_event_kind: str
    rewind_to: str
    max_iterations: int
    on_exhausted: str
