"""Tests for ``GET /api/workspaces`` — the served-workspace set."""

import pytest
from fastapi.testclient import TestClient

from molexp.server.app import create_app
from molexp.server.dependencies import (
    ServedWorkspace,
    get_served_workspaces,
    set_served_workspaces,
)


@pytest.fixture
def client():
    app = create_app(serve_static=False)
    yield TestClient(app)
    set_served_workspaces([])  # reset process-global state


def test_empty_when_unset(client):
    set_served_workspaces([])
    r = client.get("/api/workspaces")
    assert r.status_code == 200
    assert r.json() == []


def test_lists_local_and_remote_in_order(client):
    set_served_workspaces(
        [
            ServedWorkspace(key="local-a", label="/tmp/a", is_remote=False, path="/tmp/a"),
            ServedWorkspace(key="hpc-runs", label="me@hpc:/runs", is_remote=True, target_name="hpc-runs"),
        ]
    )
    r = client.get("/api/workspaces")
    assert r.status_code == 200
    body = r.json()
    assert [w["key"] for w in body] == ["local-a", "hpc-runs"]
    assert body[0] == {
        "key": "local-a",
        "label": "/tmp/a",
        "isRemote": False,
        "path": "/tmp/a",
        "active": False,
        "unreachable": False,
    }
    assert body[1]["isRemote"] is True
    assert body[1]["path"] is None  # remote has no local path
    assert body[1]["unreachable"] is True  # unregistered target → unreachable


def test_set_get_round_trip():
    set_served_workspaces([ServedWorkspace(key="k", label="l", is_remote=False, path="/x")])
    got = get_served_workspaces()
    assert [w.key for w in got] == ["k"]
    # returned list is a copy — mutating it does not affect the registry
    got.clear()
    assert len(get_served_workspaces()) == 1
    set_served_workspaces([])
