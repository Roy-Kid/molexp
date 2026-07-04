"""Workspace-level activity-stream route (vision-loop-12).

``GET /api/events`` is the global "what just happened" read over the workspace
event spine — the same shared :func:`molexp.workspace.events.read_workspace_events`
code path the per-run route and ``molexp runs info`` use. RED contract:

* cross-run timeline, newest first, rows in the frozen event shape
  (``id`` / ``seq`` / ``type`` / ``actor`` / ``created_at`` / ``payload`` / ``refs``);
* ``type`` / ``ref`` / ``limit`` query filters (limit default 50, bounded 1..500);
* a workspace with no spine answers ``[]`` WITHOUT creating the DB — reading
  is always side-effect free (events.py:247-248 contract);
* the per-run ``GET .../runs/{run_id}/events`` keeps its path and wire shape
  unchanged (the shared response model is an alias, never a wire change).
"""

from __future__ import annotations

from pathlib import Path

from molexp.workspace.events import WORKSPACE_EVENTS_DB, WorkspaceEventLog

_EVENT_KEYS = {"id", "seq", "type", "actor", "created_at", "payload", "refs"}


class TestWorkspaceEventsRoute:
    def test_returns_cross_run_timeline_newest_first(self, client, workspace) -> None:
        log = WorkspaceEventLog(workspace.root)
        log.append("run.created", "run-lifecycle", refs=["r1"])
        log.append("asset.added", "asset-accessor", refs=["a1", "r1"])
        log.append("knowledge.created", "bundle", refs=["notes/findings"])

        resp = client.get("/api/events")
        assert resp.status_code == 200
        events = resp.json()
        assert [e["type"] for e in events] == [
            "knowledge.created",
            "asset.added",
            "run.created",
        ]

    def test_rows_reuse_the_frozen_event_shape(self, client, workspace) -> None:
        WorkspaceEventLog(workspace.root).append(
            "asset.added",
            "asset-accessor",
            payload={"kind": "artifact", "name": "out.json"},
            refs=["a1", "r1"],
        )

        resp = client.get("/api/events")
        assert resp.status_code == 200
        row = resp.json()[0]
        assert set(row) == _EVENT_KEYS
        assert row["payload"] == {"kind": "artifact", "name": "out.json"}
        assert row["refs"] == ["a1", "r1"]

    def test_type_filter(self, client, workspace) -> None:
        log = WorkspaceEventLog(workspace.root)
        log.append("run.created", "run-lifecycle", refs=["r1"])
        log.append("asset.added", "asset-accessor", refs=["a1"])
        log.append("run.completed", "run-lifecycle", refs=["r1"])

        resp = client.get("/api/events", params={"type": "asset.added"})
        assert resp.status_code == 200
        events = resp.json()
        assert [e["type"] for e in events] == ["asset.added"]
        assert events[0]["refs"] == ["a1"]

    def test_ref_filter(self, client, workspace) -> None:
        log = WorkspaceEventLog(workspace.root)
        log.append("run.created", "run-lifecycle", refs=["r1"])
        log.append("run.created", "run-lifecycle", refs=["r2"])
        log.append("asset.added", "asset-accessor", refs=["a1", "r1"])

        resp = client.get("/api/events", params={"ref": "r1"})
        assert resp.status_code == 200
        events = resp.json()
        assert [e["type"] for e in events] == ["asset.added", "run.created"]
        assert all("r1" in e["refs"] for e in events)

    def test_limit_truncates_to_the_newest(self, client, workspace) -> None:
        log = WorkspaceEventLog(workspace.root)
        log.append("run.created", "test", refs=["r1"])
        log.append("run.started", "test", refs=["r1"])
        log.append("run.completed", "test", refs=["r1"])

        resp = client.get("/api/events", params={"limit": 2})
        assert resp.status_code == 200
        assert [e["type"] for e in resp.json()] == ["run.completed", "run.started"]

    def test_limit_defaults_to_50(self, client, workspace) -> None:
        log = WorkspaceEventLog(workspace.root)
        for i in range(60):
            log.append("run.created", "test", refs=[f"r{i}"])

        resp = client.get("/api/events")
        assert resp.status_code == 200
        events = resp.json()
        assert len(events) == 50
        assert events[0]["seq"] == 60  # newest first, default window

    def test_limit_bounds_rejected(self, client, workspace) -> None:
        WorkspaceEventLog(workspace.root).append("run.created", "test", refs=["r1"])

        assert client.get("/api/events", params={"limit": 0}).status_code == 422
        assert client.get("/api/events", params={"limit": 501}).status_code == 422

    def test_no_spine_returns_empty_without_creating_db(self, client, workspace) -> None:
        """Reading never creates the DB: an event-less workspace answers []
        and workspace.events.sqlite stays absent after the GET."""
        db_path = Path(str(workspace.root)) / WORKSPACE_EVENTS_DB
        assert not db_path.exists()

        resp = client.get("/api/events")
        assert resp.status_code == 200
        assert resp.json() == []
        assert not db_path.exists()


class TestPerRunEventsRouteWireCompat:
    def test_per_run_route_path_and_shape_unchanged(self, client, project, experiment, run) -> None:
        """The per-run events route keeps its path and frozen row shape —
        sharing a response model with /api/events must not change the wire."""
        resp = client.get(
            f"/api/projects/{project.id}/experiments/{experiment.id}/runs/{run.id}/events"
        )
        assert resp.status_code == 200
        events = resp.json()
        assert [e["type"] for e in events] == ["run.created"]
        assert all(set(e) == _EVENT_KEYS for e in events)
        assert run.id in events[0]["refs"]
