"""``WorkspaceEventLog`` — the workspace event spine (``workspace.events``).

The append/list/read primitives, the default-on + non-fatal
``emit_workspace_event`` helper, and the wired milestone emit sites: run
lifecycle (``run.*``), asset registration (``asset.added``), and Bundle note
creation (``knowledge.created``). One low-frequency row per milestone —
log-line appends and checkpoints deliberately stay silent — and reading is
side-effect free (it never creates the DB).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from molexp.workspace import Bundle, Workspace
from molexp.workspace.concepts import NOTE_KIND
from molexp.workspace.events import (
    WORKSPACE_EVENTS_DB,
    WorkspaceEvent,
    WorkspaceEventLog,
    emit_workspace_event,
    read_workspace_events,
)


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


class TestWorkspaceEventLog:
    """The append-only, workspace-scope timeline primitive."""

    def test_append_returns_event_and_list_round_trips_fields(self, tmp_path: Path) -> None:
        log = WorkspaceEventLog(tmp_path)
        event = log.append(
            "run.failed", "run-lifecycle", payload={"error": "boom"}, refs=["run-r1"]
        )
        assert isinstance(event, WorkspaceEvent)

        got = log.list_events()
        assert len(got) == 1
        assert got[0].type == "run.failed"
        assert got[0].actor == "run-lifecycle"
        assert got[0].payload == {"error": "boom"}
        assert got[0].refs == ["run-r1"]

    def test_seq_is_monotonic_across_the_whole_workspace(self, tmp_path: Path) -> None:
        log = WorkspaceEventLog(tmp_path)
        log.append("run.created", "test", refs=["r1"])
        log.append("asset.added", "test", refs=["a1"])
        log.append("knowledge.created", "test", refs=["k1"])
        # one monotonic timeline across the whole workspace, not per-object
        assert [e.seq for e in log.list_events()] == [1, 2, 3]

    def test_list_events_filters_by_ref_type_and_orders_newest_first(self, tmp_path: Path) -> None:
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


class TestEmitWorkspaceEvent:
    """``emit_workspace_event`` — default-on + non-fatal best-effort append."""

    def test_first_emit_creates_the_timeline_db(self, tmp_path: Path) -> None:
        result = emit_workspace_event(tmp_path, "run.created", "test", refs=["r1"])
        assert result is not None
        assert (Path(tmp_path) / WORKSPACE_EVENTS_DB).exists()
        assert [e.type for e in WorkspaceEventLog(tmp_path).list_events()] == ["run.created"]

    def test_event_log_failure_is_swallowed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import molexp.workspace.events as ev

        monkeypatch.setattr(ev, "WorkspaceEventLog", _BoomEventLog)
        assert ev.emit_workspace_event(tmp_path, "run.created", "test") is None


class TestReadWorkspaceEvents:
    """``read_workspace_events`` — the side-effect-free consumer read."""

    def test_missing_db_reads_empty_and_creates_nothing(self, tmp_path: Path) -> None:
        assert read_workspace_events(tmp_path) == []
        # Reading is side-effect free — it must never create the DB.
        assert not (tmp_path / WORKSPACE_EVENTS_DB).exists()


class TestRunLifecycleEmits:
    """The run lifecycle wires one milestone per status change onto the spine."""

    def test_success_records_created_started_completed(self, tmp_path: Path) -> None:
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
        assert [e.seq for e in events] == [1, 2, 3]

    def test_raising_run_records_failed_as_terminal_milestone(self, tmp_path: Path) -> None:
        ws = Workspace(root=tmp_path, name="Lab")
        exp = ws.add_project("p").add_experiment("e", workflow_source="t.py")
        log = WorkspaceEventLog(ws.resolve())

        run = exp.add_run(id="rf")
        with pytest.raises(RuntimeError), run.start():
            raise RuntimeError("boom")

        assert [e.type for e in log.list_events()] == [
            "run.created",
            "run.started",
            "run.failed",
        ]


class TestAssetAddedEmits:
    """The two ``asset.added`` emit sites + the frequency budget + non-fatal net."""

    def test_artifact_save_emits_exactly_one_asset_added(self, tmp_path: Path) -> None:
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

    def test_register_in_place_emits_asset_added(self, tmp_path: Path) -> None:
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

    def test_log_lines_and_checkpoints_emit_no_events(self, tmp_path: Path) -> None:
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
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-fatal contract at the emit site: a raising event log leaves the
        artifact save (and the run lifecycle around it) fully intact."""
        import molexp.workspace.events as ev

        _ws, run = _lab_run(tmp_path)
        monkeypatch.setattr(ev, "WorkspaceEventLog", _BoomEventLog)

        with run.start() as ctx:
            asset = ctx.artifact.save("result.json", {"x": 1})

        assert (Path(str(run.run_dir)) / "artifacts" / "result.json").exists()
        assert asset.content_hash is not None
        assert asset.asset_id in {a.asset_id for a in run.assets.list()}


class TestKnowledgeCreatedEmit:
    """``Bundle.create_note`` lands one ``knowledge.created`` on the spine."""

    def test_create_note_emits_knowledge_created_with_payload(self, tmp_path: Path) -> None:
        Bundle(tmp_path).create_note("findings")

        events = read_workspace_events(tmp_path, type="knowledge.created")
        assert len(events) == 1
        event = events[0]
        assert event.actor == "bundle"
        assert event.refs == ["findings"]
        assert event.payload["type"] == NOTE_KIND
        assert event.payload["title"] == "findings"
