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


def _noop_workflow(ctx):
    """Module-level callable so ``set_workflow`` can resolve an entrypoint."""


@pytest.fixture
def experiment(project):
    """Bare experiment: workflow_source label only, no entrypoint bound.

    Tests that need a dispatch-ready experiment should use
    :func:`experiment_with_entrypoint` instead.
    """
    return project.experiment(
        "test-exp",
        workflow_source="train.py",
        params={"lr": 1e-4},
    )


@pytest.fixture
def experiment_with_entrypoint(project):
    """Experiment with a module-level callable bound — dispatch-ready."""
    exp = project.experiment(
        "test-exp",
        workflow_source="train.py",
        params={"lr": 1e-4},
    )
    exp.set_workflow(_noop_workflow)
    return exp


@pytest.fixture
def run(experiment):
    return experiment.run(parameters={"lr": 1e-4})


@pytest.fixture
def client(workspace):
    app = create_app()
    app.dependency_overrides[get_workspace] = lambda: workspace
    return TestClient(app)
