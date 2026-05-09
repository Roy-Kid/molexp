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
