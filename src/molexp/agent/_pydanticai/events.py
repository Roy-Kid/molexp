"""Provider-event records for the invoke-start / invoke-end hooks.

A frozen :class:`ProviderEvent` is fired through the caller-supplied
:class:`EventCallback` slots before and after every attempt the
provider makes. This is the integration seam telemetry sinks (logging,
OpenTelemetry, Prometheus) plug into; concrete sinks live in a
separate spec.

This module imports neither ``pydantic_ai`` nor ``asyncio``. The
``Outcome`` enum has exactly three values — ``ok`` (success), ``retry``
(closing event of a non-final failed attempt), ``error`` (closing
event of a non-retryable or final failure).
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from molexp.agent.modes.plan.protocols import ModelTier

__all__ = [
    "EventCallback",
    "Outcome",
    "ProviderEvent",
]


class Outcome(StrEnum):
    """Closing-event outcome label."""

    ok = "ok"
    retry = "retry"
    error = "error"


class ProviderEvent(BaseModel):
    """One observation emitted at the start or end of a provider attempt.

    Start events always carry ``outcome=ok`` — the field's job is to
    label the **closing** event. ``duration_seconds`` is ``0.0`` on
    start events and the elapsed wall-time on closing events.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    tier: ModelTier
    node_id: str
    schema_name: str
    attempt: int
    duration_seconds: float
    outcome: Outcome


EventCallback = Callable[[ProviderEvent], None]
"""Type alias for a hook that consumes :class:`ProviderEvent` records."""


def _noop_callback(event: ProviderEvent) -> None:
    """Default no-op hook; substituted when the user supplies ``None``."""
    del event
