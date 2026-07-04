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

from molexp.workspace import Bundle, Workspace
from molexp.workspace.concepts import NOTE_KIND
from molexp.workspace.events import (
    WORKSPACE_EVENTS_DB,
    WorkspaceEvent,
    WorkspaceEventLog,
    emit_workspace_event,
    read_workspace_events,
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


# ── default-on + non-fatal emit wiring ───────────────────────────────────────


def test_emit_is_default_on(tmp_path: Path) -> None:
    """The first emit creates the timeline DB — no opt-in step exists."""
    result = emit_workspace_event(tmp_path, "run.created", "test", refs=["r1"])
    assert result is not None
    assert (Path(tmp_path) / WORKSPACE_EVENTS_DB).exists()
    assert [e.type for e in WorkspaceEventLog(tmp_path).list_events()] == ["run.created"]


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


# ── vision-loop-12: asset.added emit sites (artifact save + data registration) ─


def _lab_run(tmp_path: Path, run_id: str = "r1"):
    """A workspace + one pending run — the standard emit-site fixture shape."""
    ws = Workspace(root=tmp_path, name="Lab")
    exp = ws.add_project("p").add_experiment("e", workflow_source="t.py")
    return ws, exp.add_run(id=run_id)


class _BoomEventLog:
    """``WorkspaceEventLog`` stand-in whose every append raises (non-fatal net)."""

    def __init__(self, *args: object, **kwargs: object) -> None: ...

    def append(self, *args: object, **kwargs: object) -> WorkspaceEvent:
        raise RuntimeError("boom")


def test_artifact_save_emits_exactly_one_asset_added(tmp_path: Path) -> None:
    """One ``ctx.artifact.save`` inside ``run.start()`` → exactly ONE
    ``asset.added`` on the workspace spine: refs carry the asset id AND the
    run id (drill-down pointers), payload names kind/name/content_hash."""
    ws, run = _lab_run(tmp_path)
    with run.start() as ctx:
        asset = ctx.artifact.save("result.json", {"x": 1})

    events = read_workspace_events(ws.resolve(), type="asset.added")
    assert len(events) == 1
    event = events[0]
    assert event.actor == "asset-accessor"
    assert asset.asset_id in event.refs
    assert "r1" in event.refs
    assert event.payload["kind"] == "artifact"
    assert event.payload["name"] == "result.json"
    assert event.payload["content_hash"] == asset.content_hash


def test_register_in_place_emits_asset_added(tmp_path: Path) -> None:
    """The data-import verb (``data_assets.register_in_place``) is the second
    ``asset.added`` emit site — same actor + payload vocabulary."""
    ws = Workspace(root=tmp_path, name="Lab")
    src = tmp_path / "data.csv"
    src.write_text("a,b\n1,2\n")

    asset = ws.data_assets.register_in_place("data", src)

    events = read_workspace_events(ws.resolve(), type="asset.added")
    assert len(events) == 1
    event = events[0]
    assert event.actor == "asset-accessor"
    assert asset.asset_id in event.refs
    assert event.payload["kind"] == "data"
    assert event.payload["name"] == "data"
    assert event.payload["content_hash"] == asset.content_hash


def test_log_and_checkpoint_writes_emit_no_events(tmp_path: Path) -> None:
    """Frequency budget: N log-line appends + checkpoint writes emit ZERO
    events — the spine holds only the three run-lifecycle milestones."""
    ws, run = _lab_run(tmp_path)
    with run.start() as ctx:
        for i in range(25):
            ctx.log("train").append(f"line {i}")
        ctx.checkpoint("mid", data={"step": 1})
        ctx.checkpoint("end", data={"step": 2})

    events = read_workspace_events(ws.resolve())
    assert [e.type for e in events] == ["run.completed", "run.started", "run.created"]


def test_broken_event_log_never_breaks_artifact_save(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-fatal contract: a raising event log leaves the artifact save (and
    the run lifecycle around it) fully intact."""
    import molexp.workspace.events as ev

    _ws, run = _lab_run(tmp_path)
    monkeypatch.setattr(ev, "WorkspaceEventLog", _BoomEventLog)

    with run.start() as ctx:
        asset = ctx.artifact.save("result.json", {"x": 1})

    assert (Path(str(run.run_dir)) / "artifacts" / "result.json").exists()
    assert asset.content_hash is not None
    assert asset.asset_id in {a.asset_id for a in run.assets.list()}


def test_accessor_without_event_root_emits_nothing(tmp_path: Path) -> None:
    """Constructor compat: an ``ArtifactAccessor`` built WITHOUT ``event_root``
    (the default) stays byte-identical — no emit, no spine DB anywhere."""
    from molexp.workspace.assets import (
        ArtifactAccessor,
        AssetManifest,
        AssetScope,
        Producer,
    )

    scope_dir = tmp_path / "scope"
    scope_dir.mkdir()
    scope = AssetScope(kind="run", ids=("p", "e", "r1"))
    accessor = ArtifactAccessor(
        scope_dir, scope, AssetManifest(scope_dir), lambda: Producer(run_id="r1")
    )

    accessor.save("out.txt", "hello")

    assert list(tmp_path.rglob(WORKSPACE_EVENTS_DB)) == []


# ── vision-loop-12: knowledge.created emit site (Bundle create verb) ──────────


def test_bundle_create_note_emits_knowledge_created(tmp_path: Path) -> None:
    """``Bundle.create_note`` (the create verb behind the knowledge routes +
    CLI) lands one ``knowledge.created``: ref = bundle-relative path, payload
    ``{type, title}``, actor ``bundle``."""
    Bundle(tmp_path).create_note("findings")

    events = read_workspace_events(tmp_path, type="knowledge.created")
    assert len(events) == 1
    event = events[0]
    assert event.actor == "bundle"
    assert event.refs == ["findings"]
    assert event.payload["type"] == NOTE_KIND
    assert event.payload["title"] == "findings"


def test_bundle_create_note_ref_is_the_bundle_relative_path(tmp_path: Path) -> None:
    """A nested note's ref is its full bundle-relative identity path (slugged),
    not the bare name — path-as-identity carries into the spine."""
    bundle = Bundle(tmp_path)
    parent = bundle.create_note("protocols")
    bundle.create_note("gel prep", parent=parent)

    events = read_workspace_events(tmp_path, type="knowledge.created")
    assert [e.refs for e in events] == [["protocols/gel-prep"], ["protocols"]]
