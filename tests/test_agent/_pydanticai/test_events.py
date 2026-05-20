"""Tests for :mod:`molexp.agent._pydanticai.events`.

Acceptance criterion ac-004: ``ProviderEvent`` is frozen; ``Outcome``
covers ``ok`` / ``retry`` / ``error``; the :data:`EventCallback`
type alias is structurally callable.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from molexp.agent._pydanticai.events import (
    EventCallback,
    Outcome,
    ProviderEvent,
)
from molexp.agent.router import ModelTier


def _sample_event(**overrides: object) -> ProviderEvent:
    base = {
        "tier": ModelTier.DEFAULT,
        "node_id": "ingest",
        "schema_name": "ReportDigest",
        "attempt": 1,
        "duration_seconds": 0.0,
        "outcome": Outcome.ok,
    }
    base.update(overrides)
    return ProviderEvent(**base)  # type: ignore[arg-type]


# ── Outcome enum ───────────────────────────────────────────────────────────


def test_outcome_members_are_exactly_ok_retry_error() -> None:
    assert {member.value for member in Outcome} == {"ok", "retry", "error"}


# ── ProviderEvent ──────────────────────────────────────────────────────────


def test_provider_event_construction_with_all_fields() -> None:
    event = _sample_event(attempt=2, duration_seconds=0.42, outcome=Outcome.retry)
    assert event.attempt == 2
    assert event.duration_seconds == 0.42
    assert event.outcome is Outcome.retry


def test_provider_event_is_frozen() -> None:
    event = _sample_event()
    with pytest.raises(ValidationError):
        event.attempt = 99  # type: ignore[misc]


def test_provider_event_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        ProviderEvent(
            tier=ModelTier.DEFAULT,
            node_id="x",
            schema_name="S",
            attempt=1,
            duration_seconds=0.0,
            outcome=Outcome.ok,
            stray=1,  # type: ignore[call-arg]
        )


def test_provider_event_rejects_unknown_outcome() -> None:
    with pytest.raises(ValidationError):
        ProviderEvent(
            tier=ModelTier.DEFAULT,
            node_id="x",
            schema_name="S",
            attempt=1,
            duration_seconds=0.0,
            outcome="cancelled",  # type: ignore[arg-type]
        )


# ── EventCallback type alias ───────────────────────────────────────────────


def test_event_callback_alias_is_callable_compatible() -> None:
    """Any callable matching the (event) -> None shape satisfies the alias."""
    received: list[ProviderEvent] = []

    def hook(event: ProviderEvent) -> None:
        received.append(event)

    cb: EventCallback = hook
    cb(_sample_event())
    assert len(received) == 1
    assert received[0].outcome is Outcome.ok
