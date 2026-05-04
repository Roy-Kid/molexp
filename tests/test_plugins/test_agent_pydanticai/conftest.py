"""Shared fixtures for agent_pydanticai tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from molexp.workspace import Workspace


@pytest.fixture(autouse=True)
def _isolated_user_home(tmp_path_factory, monkeypatch):
    """Redirect ``Path.home()`` so tests cannot read or pollute ``~/.molexp``.

    The agent's :class:`SkillStore` uses ``~/.molexp/skills.json`` as the
    user-home tier; without this fixture the suite would observe the
    developer's real config and writes would persist between runs.
    """
    fake_home = tmp_path_factory.mktemp("home")
    real_home = Path.home

    def _fake_home(cls=Path):
        return fake_home

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    yield fake_home
    monkeypatch.setattr(Path, "home", real_home)


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
