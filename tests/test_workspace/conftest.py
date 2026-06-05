"""Shared fixtures for workspace tests."""

from __future__ import annotations

import pytest

from molexp.workspace import Workspace


@pytest.fixture
def workspace(tmp_path):
    return Workspace(root=tmp_path, name="Test Lab")


@pytest.fixture
def project(workspace):
    return workspace.add_project("test-project")


@pytest.fixture
def experiment(project):
    return project.add_experiment(
        "test-experiment",
        workflow_source="train.py",
        params={"lr": 1e-4},
        git_commit="abc123",
    )


@pytest.fixture
def run(experiment):
    return experiment.add_run(parameters={"lr": 1e-4})
