"""WorkspaceEventLog — the workspace event spine (workspace-event-02-eventlog, P0.3).

The emit tests exercise the default-on + non-fatal ``emit_workspace_event``
helper and the wired ``run.*`` milestone emits: run-lifecycle milestones land
on the timeline without any opt-in step (one low-frequency row per run status
change), so ``molexp runs info`` / the server's events endpoint show them on
the default path. Reading stays side-effect free — it never creates the DB.
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


# ── default-on + non-fatal emit wiring ───────────────────────────────────────


def test_emit_is_default_on(tmp_path: Path) -> None:
    """The first emit creates the timeline DB — no opt-in step exists."""
    result = emit_workspace_event(tmp_path, "run.created", "test", refs=["r1"])
    assert result is not None
    assert (Path(tmp_path) / WORKSPACE_EVENTS_DB).exists()
    assert [e.type for e in WorkspaceEventLog(tmp_path).list_events()] == ["run.created"]


def test_run_lifecycle_records_timeline_by_default(tmp_path: Path) -> None:
    """A plain run lifecycle lands its milestones without any opt-in step —
    this is what makes the ``runs info`` Recent-events section real UX."""
    ws = Workspace(root=tmp_path, name="Lab")
    exp = ws.add_project("p").add_experiment("e", workflow_source="t.py")
    run = exp.add_run(id="r1")
    with run.start():
        pass

    from molexp.workspace.events import read_workspace_events

    events = read_workspace_events(ws.resolve(), ref="r1")
    assert [e.type for e in events] == ["run.completed", "run.started", "run.created"]


def test_run_lifecycle_emits(tmp_path: Path) -> None:
    """A workspace records run.created → run.started → run.completed."""
    ws = Workspace(root=tmp_path, name="Lab")
    exp = ws.add_project("p").add_experiment("e", workflow_source="t.py")
    log = WorkspaceEventLog(ws.resolve())

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

    ev.WorkspaceEventLog(tmp_path)  # pre-create the real DB

    class _Boom:
        def __init__(self, *args: object, **kwargs: object) -> None: ...

        def append(self, *args: object, **kwargs: object) -> WorkspaceEvent:
            raise RuntimeError("boom")

    monkeypatch.setattr(ev, "WorkspaceEventLog", _Boom)
    assert ev.emit_workspace_event(tmp_path, "run.created", "test") is None


# ── Read path (run-recovery: the event spine gains consumers) ────────────────


def test_list_events_filters_and_ordering(tmp_path: Path) -> None:
    log = WorkspaceEventLog(tmp_path)
    log.append("run.created", "test", refs=["r1"])
    log.append("run.started", "test", refs=["r1"])
    log.append("run.created", "test", refs=["r2"])
    log.append("run.failed", "test", refs=["r1"])

    by_ref = log.list_events(ref="r1")
    assert [e.type for e in by_ref] == ["run.created", "run.started", "run.failed"]

    by_type = log.list_events(type="run.created")
    assert [e.refs[0] for e in by_type] == ["r1", "r2"]

    newest = log.list_events(ref="r1", newest_first=True, limit=2)
    assert [e.type for e in newest] == ["run.failed", "run.started"]


def test_read_workspace_events_without_db_is_empty(tmp_path: Path) -> None:
    from molexp.workspace.events import read_workspace_events

    assert read_workspace_events(tmp_path) == []
    # Reading is side-effect free — it must never create the DB.
    assert not (tmp_path / WORKSPACE_EVENTS_DB).exists()


def test_read_workspace_events_newest_first(tmp_path: Path) -> None:
    from molexp.workspace.events import read_workspace_events

    log = WorkspaceEventLog(tmp_path)
    log.append("run.created", "test", refs=["r1"])
    log.append("run.completed", "test", refs=["r1"])

    events = read_workspace_events(tmp_path, ref="r1", limit=1)
    assert [e.type for e in events] == ["run.completed"]
