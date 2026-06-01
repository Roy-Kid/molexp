"""Replay + resume helpers (Phase 10).

- :func:`replay_metadata` — re-read events for a run; the simplest form of
  "replay" per harness-goal §13.4 metadata-replay tier.
- :func:`find_last_successful_stage` — walk the event log and return the
  name of the most-recently-completed stage that wasn't subsequently
  marked failed. Resume-from-failure entry point.
"""

from __future__ import annotations

from molexp.harness.schemas import HarnessEvent
from molexp.harness.store.event_log import EventLog

__all__ = ["find_last_successful_stage", "replay_metadata"]


def replay_metadata(event_log: EventLog, run_id: str) -> list[HarnessEvent]:
    """Return every event for ``run_id`` in seq order. No re-execution."""
    return event_log.list_events(run_id)


def find_last_successful_stage(event_log: EventLog, run_id: str) -> str | None:
    """Return the most-recently-completed stage name whose success record
    was not subsequently invalidated by a matching ``stage_failed`` event.

    Sequence ``[A_completed, B_completed, C_started, C_failed]`` returns
    ``"B"`` — the failure of C does not invalidate B's completion. A
    ``stage_failed`` only invalidates ``last_completed`` when the failed
    stage's name matches it (i.e. the same stage was rerun and then
    failed). Returns ``None`` for empty logs, only-started runs, or runs
    whose only completed stage was subsequently failed.
    """
    last_completed: str | None = None
    for event in event_log.list_events(run_id):
        if event.type == "stage_completed":
            last_completed = event.payload.get("stage")
        elif event.type == "stage_failed":
            failed_stage = event.payload.get("stage")
            if failed_stage is not None and failed_stage == last_completed:
                last_completed = None
    return last_completed
