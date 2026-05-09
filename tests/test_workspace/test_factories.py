"""SRP factory split + workspace error hierarchy.

Covers acceptance criteria ac-001, ac-002, ac-003 of the
``oop-api-rectification`` spec:

- ``workspace/errors.py`` exists with six concrete exception classes
- ``Workspace.Project / .create_project / .project`` enforce
  idempotent / strict-create / strict-get semantics; mirrored on
  ``Project.Experiment / .create_experiment / .experiment`` and
  ``Experiment.Run / .create_run / .run``
- ``Workspace.get_project`` / ``Project.get_experiment`` /
  ``Experiment.get_run`` are removed — the strict lowercase getters
  replace them.
"""

from __future__ import annotations

import pytest

from molexp.workspace import (
    ExperimentExistsError,
    ExperimentNotFoundError,
    ProjectExistsError,
    ProjectNotFoundError,
    RunExistsError,
    RunNotFoundError,
    Workspace,
)

# ── Exception module shape ─────────────────────────────────────────────────


def test_not_found_errors_are_lookup_errors():
    assert issubclass(ProjectNotFoundError, LookupError)
    assert issubclass(ExperimentNotFoundError, LookupError)
    assert issubclass(RunNotFoundError, LookupError)


def test_exists_errors_are_value_errors():
    assert issubclass(ProjectExistsError, ValueError)
    assert issubclass(ExperimentExistsError, ValueError)
    assert issubclass(RunExistsError, ValueError)


def test_errors_module_carries_entity_id_in_message():
    assert "demo-proj" in str(ProjectNotFoundError("demo-proj"))
    assert "demo-exp" in str(ExperimentExistsError("demo-exp"))


# ── Workspace SRP factory split ────────────────────────────────────────────


def test_workspace_Project_is_idempotent(tmp_path):
    ws = Workspace(root=tmp_path, name="ws")
    p1 = ws.Project(name="demo")
    p2 = ws.Project(name="demo")
    assert p1 is p2
    assert p1.name == "demo"


def test_workspace_create_project_raises_on_collision(tmp_path):
    ws = Workspace(root=tmp_path, name="ws")
    ws.create_project(name="demo")
    with pytest.raises(ProjectExistsError) as exc:
        ws.create_project(name="demo")
    assert "demo" in str(exc.value)


def test_workspace_project_is_strict_getter(tmp_path):
    ws = Workspace(root=tmp_path, name="ws")
    ws.Project(name="demo")
    got = ws.project("demo")
    assert got.name == "demo"


def test_workspace_project_raises_on_missing(tmp_path):
    ws = Workspace(root=tmp_path, name="ws")
    with pytest.raises(ProjectNotFoundError) as exc:
        ws.project("never-created")
    assert "never-created" in str(exc.value)


def test_workspace_get_project_is_removed(tmp_path):
    ws = Workspace(root=tmp_path, name="ws")
    assert not hasattr(ws, "get_project")


# ── Project SRP factory split ──────────────────────────────────────────────


def test_project_Experiment_is_idempotent(tmp_path):
    ws = Workspace(root=tmp_path, name="ws")
    proj = ws.Project(name="demo")
    e1 = proj.Experiment(name="counter")
    e2 = proj.Experiment(name="counter")
    assert e1 is e2


def test_project_create_experiment_raises_on_collision(tmp_path):
    ws = Workspace(root=tmp_path, name="ws")
    proj = ws.Project(name="demo")
    proj.create_experiment(name="counter")
    with pytest.raises(ExperimentExistsError):
        proj.create_experiment(name="counter")


def test_project_experiment_is_strict_getter(tmp_path):
    ws = Workspace(root=tmp_path, name="ws")
    proj = ws.Project(name="demo")
    proj.Experiment(name="counter")
    got = proj.experiment("counter")
    assert got.name == "counter"


def test_project_experiment_raises_on_missing(tmp_path):
    ws = Workspace(root=tmp_path, name="ws")
    proj = ws.Project(name="demo")
    with pytest.raises(ExperimentNotFoundError):
        proj.experiment("never-created")


def test_project_get_experiment_is_removed(tmp_path):
    ws = Workspace(root=tmp_path, name="ws")
    proj = ws.Project(name="demo")
    assert not hasattr(proj, "get_experiment")


# ── Experiment SRP factory split ───────────────────────────────────────────


def test_experiment_Run_is_idempotent(tmp_path):
    ws = Workspace(root=tmp_path, name="ws")
    proj = ws.Project(name="demo")
    exp = proj.Experiment(name="counter")
    r1 = exp.Run(parameters={"lr": 1e-3}, id="r1")
    r2 = exp.Run(parameters={"lr": 1e-3}, id="r1")
    assert r1.id == r2.id == "r1"


def test_experiment_create_run_raises_on_collision(tmp_path):
    ws = Workspace(root=tmp_path, name="ws")
    proj = ws.Project(name="demo")
    exp = proj.Experiment(name="counter")
    exp.create_run(parameters={"lr": 1e-3}, id="r1")
    with pytest.raises(RunExistsError):
        exp.create_run(parameters={"lr": 1e-3}, id="r1")


def test_experiment_run_is_strict_getter(tmp_path):
    ws = Workspace(root=tmp_path, name="ws")
    proj = ws.Project(name="demo")
    exp = proj.Experiment(name="counter")
    exp.Run(parameters={"lr": 1e-3}, id="r1")
    got = exp.run("r1")
    assert got.id == "r1"


def test_experiment_run_raises_on_missing(tmp_path):
    ws = Workspace(root=tmp_path, name="ws")
    proj = ws.Project(name="demo")
    exp = proj.Experiment(name="counter")
    with pytest.raises(RunNotFoundError):
        exp.run("never-created")


def test_experiment_get_run_is_removed(tmp_path):
    ws = Workspace(root=tmp_path, name="ws")
    proj = ws.Project(name="demo")
    exp = proj.Experiment(name="counter")
    assert not hasattr(exp, "get_run")
