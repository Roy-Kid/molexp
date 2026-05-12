"""Shared fixtures for server tests."""

import pytest
from fastapi.testclient import TestClient

from molexp.server.app import create_app
from molexp.server.dependencies import get_workspace
from molexp.workflow import (
    Task,
    TaskContext,
    Workflow,
    WorkflowBuilder,
)
from molexp.workspace import Workspace


@pytest.fixture(autouse=True)
def _isolate_workflow_bindings():
    """Each test gets a fresh process-local workflow-binding registry."""
    Workflow._reset_registry()
    yield
    Workflow._reset_registry()


@pytest.fixture(autouse=True)
def _isolate_molcrafts_home(tmp_path, monkeypatch):
    """Redirect molcfg's project-config base to a tmp dir for each test.

    Routes that dispatch through the molq plugin auto-bootstrap a
    ``JobStore`` via :func:`molq.store.default_jobs_db_path`, which
    delegates to :func:`molcfg.project_config_dir`. Setting
    ``MOLCRAFTS_HOME`` redirects the *bootstrap location* (and any
    other molcfg-managed paths) under ``tmp_path`` so tests don't
    touch the developer's real ``~/.molcrafts`` tree.
    """
    fake_home = tmp_path / "_molcrafts_home"
    fake_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MOLCRAFTS_HOME", str(fake_home))


@pytest.fixture
def workspace(tmp_path):
    return Workspace(root=tmp_path, name="Test")


@pytest.fixture
def project(workspace):
    return workspace.Project("test-project")


class _NoopTask(Task):
    """Module-level Task subclass — gives the entrypoint resolver a
    user-module first task so ``resolve_spec_entrypoint`` can find
    the spec's module-level binding."""

    async def execute(self, ctx: TaskContext) -> None:
        return None


# Module-level Workflow — explicitly named at module scope so
# ``resolve_spec_entrypoint`` returns ``<this-file>:_NOOP_SPEC``.
_NOOP_SPEC = WorkflowBuilder(name="noop").add(_NoopTask(), name="step").build()


@pytest.fixture
def experiment(project):
    """Bare experiment: workflow_source label only, no spec bound.

    Tests that need a dispatch-ready experiment should use
    :func:`experiment_with_entrypoint` instead.
    """
    return project.Experiment(
        "test-exp",
        workflow_source="train.py",
        params={"lr": 1e-4},
    )


@pytest.fixture
def experiment_with_entrypoint(project):
    """Experiment with a workflow spec bound through the registry.

    The spec is module-level (``_NOOP_SPEC``) so the server route's
    ``resolve_spec_entrypoint`` returns a valid ``<file>:<varname>``
    handle.
    """
    exp = project.Experiment(
        "test-exp",
        workflow_source="train.py",
        params={"lr": 1e-4},
    )
    _NOOP_SPEC.bind_to(exp)
    return exp


@pytest.fixture
def run(experiment):
    return experiment.Run(parameters={"lr": 1e-4})


@pytest.fixture
def client(workspace):
    app = create_app()
    app.dependency_overrides[get_workspace] = lambda: workspace
    return TestClient(app)
