"""Shared fixtures for agent_pydanticai tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from molexp.workspace import Workspace


@pytest.fixture
def workspace(tmp_path):
    """Empty workspace anchored at tmp_path."""
    return Workspace(root=tmp_path, name="Agent Test Lab")


@pytest.fixture
def project(workspace):
    return workspace.project("proj-a")


@pytest.fixture
def experiment(project):
    return project.experiment(
        "exp-a",
        workflow_source=None,
        params={"temperature": 300},
    )


@pytest.fixture
def existing_run(experiment):
    return experiment.run(parameters={"temperature": 300, "seed": 1})


@pytest.fixture
def fake_ctx(workspace) -> Any:
    """A minimal stand-in for ``RunContext[MolexpDeps]`` used in tool tests.

    Only the ``deps.workspace`` and ``deps.session`` attributes are
    accessed by the workspace tools, so a SimpleNamespace is enough.
    """
    deps = SimpleNamespace(workspace=workspace, session=None, current_run=None)
    return SimpleNamespace(deps=deps)
