"""Shared fixtures for workspace tests."""

from __future__ import annotations

import pytest

from molexp.workspace import Experiment, Project, Run, Workspace
from molexp.workspace.checkpoint import CheckpointState
from molexp.workspace.run import RunStatus


@pytest.fixture
def workspace(tmp_path):
    ws = Workspace(root=tmp_path, name="Test Lab")
    ws.materialize()
    return ws


@pytest.fixture
def project(workspace):
    return workspace.create_project(name="test-project")


@pytest.fixture
def experiment(project):
    return project.create_experiment(
        name="test-experiment",
        workflow_source="train.py",
        parameter_space={"lr": [1e-4, 1e-3]},
        git_commit="abc123",
    )


@pytest.fixture
def run(experiment):
    return experiment.create_run(parameters={"lr": 1e-4})


@pytest.fixture
def checkpoint_state():
    from datetime import datetime

    return CheckpointState(
        ckpt_id="ckpt_test123",
        run_id="run-1",
        experiment_id="exp-1",
        project_id="proj-1",
        timestamp=datetime.now(),
        context={"results": {"acc": 0.9}},
    )


class MockRun:
    def __init__(self, status: str = "pending"):
        self.status = status


@pytest.fixture
def mock_run():
    return MockRun
