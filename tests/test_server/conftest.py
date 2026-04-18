"""Shared fixtures for server tests."""

import pytest
from fastapi.testclient import TestClient

from molexp.server.app import create_app
from molexp.server.dependencies import get_workspace
from molexp.workspace import Workspace


@pytest.fixture
def workspace(tmp_path):
    return Workspace(root=tmp_path, name="Test")


@pytest.fixture
def project(workspace):
    return workspace.project("test-project")


@pytest.fixture
def experiment(project):
    return project.experiment(
        "test-exp",
        workflow_source="train.py",
        params={"lr": 1e-4},
    )


@pytest.fixture
def run(experiment):
    return experiment.run(parameters={"lr": 1e-4})


@pytest.fixture
def client(workspace):
    app = create_app()
    app.dependency_overrides[get_workspace] = lambda: workspace
    return TestClient(app)
