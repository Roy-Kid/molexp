"""Shared fixtures for server tests."""

import pytest
from fastapi.testclient import TestClient

from molexp.server.app import create_app
from molexp.server.dependencies import get_workspace
from molexp.workspace import Workspace


@pytest.fixture
def workspace(tmp_path):
    ws = Workspace(root=tmp_path, name="Test")
    ws.materialize()
    return ws


@pytest.fixture
def project(workspace):
    return workspace.create_project("test-project")


@pytest.fixture
def experiment(project):
    return project.create_experiment(
        "test-exp",
        workflow_source="train.py",
        parameter_space={"lr": [1e-4]},
    )


@pytest.fixture
def run(experiment):
    return experiment.create_run(parameters={"lr": 1e-4})


@pytest.fixture
def client(workspace):
    app = create_app()
    app.dependency_overrides[get_workspace] = lambda: workspace
    return TestClient(app)
