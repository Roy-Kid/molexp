"""Tests for /api/targets — workspace ComputeTarget CRUD."""

from __future__ import annotations

import pytest


@pytest.mark.unit
def test_list_targets_empty(client):
    resp = client.get("/api/targets")
    assert resp.status_code == 200
    body = resp.json()
    assert body == {"targets": [], "total": 0}


@pytest.mark.unit
def test_create_target_local(client, workspace):
    resp = client.post(
        "/api/targets",
        json={
            "name": "laptop",
            "scratchRoot": "/tmp/molexp",
            "scheduler": "shell",
        },
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["name"] == "laptop"
    assert body["isRemote"] is False
    assert body["host"] is None
    assert body["scratchRoot"] == "/tmp/molexp"

    assert any(t.name == "laptop" for t in workspace.metadata.targets)


@pytest.mark.unit
def test_create_target_remote_slurm(client):
    resp = client.post(
        "/api/targets",
        json={
            "name": "hpc1",
            "scratchRoot": "/scratch/me/molexp",
            "scheduler": "slurm",
            "host": "me@hpc.example.org",
        },
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["isRemote"] is True
    assert body["scheduler"] == "slurm"


@pytest.mark.unit
def test_create_target_duplicate_name_returns_409(client):
    payload = {"name": "dup", "scratchRoot": "/tmp/x", "scheduler": "shell"}
    assert client.post("/api/targets", json=payload).status_code == 201
    resp = client.post("/api/targets", json=payload)
    assert resp.status_code == 409
    assert "already exists" in resp.json()["detail"]


@pytest.mark.unit
def test_create_target_rejects_ssh_opts_without_host(client):
    resp = client.post(
        "/api/targets",
        json={
            "name": "bad",
            "scratchRoot": "/tmp/x",
            "scheduler": "shell",
            "port": 22,
        },
    )
    assert resp.status_code == 422


@pytest.mark.unit
def test_create_target_rejects_invalid_scheduler(client):
    resp = client.post(
        "/api/targets",
        json={
            "name": "bad",
            "scratchRoot": "/tmp/x",
            "scheduler": "k8s",
        },
    )
    assert resp.status_code == 422


@pytest.mark.unit
def test_list_then_delete(client):
    client.post(
        "/api/targets",
        json={"name": "tmp", "scratchRoot": "/tmp/x", "scheduler": "shell"},
    )
    listed = client.get("/api/targets").json()
    assert listed["total"] == 1

    resp = client.delete("/api/targets/tmp")
    assert resp.status_code == 204

    listed = client.get("/api/targets").json()
    assert listed["total"] == 0


@pytest.mark.unit
def test_delete_unknown_target_returns_404(client):
    resp = client.delete("/api/targets/no-such")
    assert resp.status_code == 404


@pytest.mark.unit
def test_test_target_local_succeeds(client, tmp_path):
    scratch = tmp_path / "scratch"
    client.post(
        "/api/targets",
        json={
            "name": "local-test",
            "scratchRoot": str(scratch),
            "scheduler": "shell",
        },
    )
    resp = client.post("/api/targets/local-test/test")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["error"] is None
    labels = [c["label"] for c in body["checks"]]
    assert "command execution" in labels
    assert "file round-trip" in labels


@pytest.mark.unit
def test_test_unknown_target_returns_404(client):
    resp = client.post("/api/targets/no-such/test")
    assert resp.status_code == 404
