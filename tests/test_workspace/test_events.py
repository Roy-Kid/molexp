"""WorkspaceEventLog — the workspace event spine (workspace-event-02-eventlog, P0.3).

RED-first: ``molexp.workspace.events`` does not exist yet.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from molexp.workspace.events import WORKSPACE_EVENTS_DB, WorkspaceEvent, WorkspaceEventLog


def test_event_round_trip(tmp_path: Path) -> None:
    log = WorkspaceEventLog(tmp_path)
    event = log.append("run.failed", "run-lifecycle", payload={"error": "boom"}, refs=["run-r1"])
    assert isinstance(event, WorkspaceEvent)
    with pytest.raises(ValidationError):
        event.type = "asset.added"  # type: ignore[misc]

    got = log.list_events()
    assert len(got) == 1
    assert got[0].type == "run.failed"
    assert got[0].actor == "run-lifecycle"
    assert got[0].payload == {"error": "boom"}
    assert got[0].refs == ["run-r1"]


def test_monotonic_workspace_seq(tmp_path: Path) -> None:
    log = WorkspaceEventLog(tmp_path)
    log.append("run.created", "test", refs=["r1"])
    log.append("asset.added", "test", refs=["a1"])
    log.append("knowledge.created", "test", refs=["k1"])
    # one monotonic timeline across the whole workspace, not per-object
    assert [e.seq for e in log.list_events()] == [1, 2, 3]


def test_events_persist(tmp_path: Path) -> None:
    WorkspaceEventLog(tmp_path).append("run.completed", "test", refs=["r1"])
    fresh = WorkspaceEventLog(tmp_path)
    assert [e.type for e in fresh.list_events()] == ["run.completed"]
    assert (Path(tmp_path) / WORKSPACE_EVENTS_DB).exists()
