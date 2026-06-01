"""``EventLog`` Protocol — append-only audit timeline contract."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from molexp.harness.schemas import EventType, HarnessEvent

__all__ = ["EventLog"]


@runtime_checkable
class EventLog(Protocol):
    """Structural type for any event-log backend."""

    def append(
        self,
        run_id: str,
        type: EventType,
        actor: str,
        payload: dict[str, Any] | None = None,
        artifact_ids: list[str] | None = None,
    ) -> HarnessEvent: ...

    def list_events(self, run_id: str) -> list[HarnessEvent]: ...

    def get_timeline(self, run_id: str) -> list[HarnessEvent]: ...
