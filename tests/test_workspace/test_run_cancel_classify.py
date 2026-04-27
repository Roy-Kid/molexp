"""Unit tests for molexp._run_cancel classification."""

from __future__ import annotations

import platform

import pytest

from molexp._run_cancel import classify
from molexp.workspace import Workspace


@pytest.fixture
def running_run(tmp_path):
    ws = Workspace(root=tmp_path, name="lab")
    ws.materialize()
    e = ws.project("p").experiment("e", workflow_source="s.py", params={})
    r = e.run(parameters={"seed": 1})
    r._update_metadata(status="running")
    return r


def test_molq_backend_uses_job_id(running_run):
    running_run._update_metadata(
        executor_info={
            "backend": "molq",
            "scheduler": "slurm",
            "cluster_name": "hpc",
            "job_id": "abc-uuid-123",
            "scheduler_job_id": "88001",
        }
    )
    plan = classify(running_run)
    assert plan.kind == "molq"
    assert plan.detail == "hpc"
    assert plan.job_id == "abc-uuid-123"


def test_local_same_host_uses_pid(running_run):
    running_run._update_metadata(labels={"pid": "12345", "host": platform.node()})
    plan = classify(running_run)
    assert plan.kind == "local"
    assert plan.detail == "12345"


def test_local_different_host_is_uncancellable(running_run):
    running_run._update_metadata(labels={"pid": "12345", "host": "some-other-host.example"})
    plan = classify(running_run)
    assert plan.kind == "none"
    assert "different host" in plan.detail


def test_terminal_status_is_uncancellable(running_run):
    running_run._update_metadata(status="succeeded")
    plan = classify(running_run)
    assert plan.kind == "none"
    assert plan.detail == "already terminal"


def test_no_info_is_uncancellable(running_run):
    plan = classify(running_run)
    assert plan.kind == "none"
