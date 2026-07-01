"""WorkspaceEventLog — the workspace event spine (workspace-event-02-eventlog, P0.3).

The slice-03 emit tests exercise the opt-in-by-existence + non-fatal
``emit_workspace_event`` helper and the wired ``run.*`` milestone emits.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from molexp.workspace import Workspace
from molexp.workspace.events import (
    WORKSPACE_EVENTS_DB,
    WorkspaceEvent,
    WorkspaceEventLog,
    emit_workspace_event,
)


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


# ── slice-03: opt-in-by-existence + non-fatal emit wiring ────────────────────


def test_emit_is_optin(tmp_path: Path) -> None:
    """No events DB → emit is a no-op that returns None and adds no file."""
    result = emit_workspace_event(tmp_path, "run.created", "test", refs=["r1"])
    assert result is None
    assert not (Path(tmp_path) / WORKSPACE_EVENTS_DB).exists()


def test_run_lifecycle_no_db_adds_no_file(tmp_path: Path) -> None:
    """A full run lifecycle in a workspace that never opted in adds no events DB."""
    ws = Workspace(root=tmp_path, name="Lab")
    exp = ws.add_project("p").add_experiment("e", workflow_source="t.py")
    run = exp.add_run(id="r1")
    with run.start():
        pass
    assert not (Path(ws.resolve()) / WORKSPACE_EVENTS_DB).exists()


def test_run_lifecycle_emits(tmp_path: Path) -> None:
    """An enabled workspace records run.created → run.started → run.completed."""
    ws = Workspace(root=tmp_path, name="Lab")
    exp = ws.add_project("p").add_experiment("e", workflow_source="t.py")
    log = WorkspaceEventLog(ws.resolve())  # opt in (creates the DB)

    run = exp.add_run(id="r1")
    with run.start():
        pass

    events = log.list_events()
    assert [e.type for e in events] == ["run.created", "run.started", "run.completed"]
    # each milestone carries the run id in refs (drill-down pointer)
    for event in events:
        assert "r1" in event.refs
    # seq-ordered timeline
    assert [e.seq for e in events] == [1, 2, 3]


def test_run_failed_emits(tmp_path: Path) -> None:
    """A raising run records run.failed as its terminal milestone."""
    ws = Workspace(root=tmp_path, name="Lab")
    exp = ws.add_project("p").add_experiment("e", workflow_source="t.py")
    log = WorkspaceEventLog(ws.resolve())

    run = exp.add_run(id="rf")
    with pytest.raises(RuntimeError), run.start():
        raise RuntimeError("boom")

    assert [e.type for e in log.list_events()] == ["run.created", "run.started", "run.failed"]


def test_emit_non_fatal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A raising WorkspaceEventLog never propagates out of emit_workspace_event."""
    import molexp.workspace.events as ev

    ev.WorkspaceEventLog(tmp_path)  # opt in (real DB exists)

    class _Boom:
        def __init__(self, *args: object, **kwargs: object) -> None: ...

        def append(self, *args: object, **kwargs: object) -> WorkspaceEvent:
            raise RuntimeError("boom")

    monkeypatch.setattr(ev, "WorkspaceEventLog", _Boom)
    assert ev.emit_workspace_event(tmp_path, "run.created", "test") is None
