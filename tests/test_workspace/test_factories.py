"""Typed semantic-sugar CRUD + workspace error hierarchy.

Covers the post-rectification ``unify-folder-abstraction-03`` spec
acceptance — workspace entities all subclass ``Folder`` and expose a
snake_case verb-noun CRUD (``add_* / get_* / has_* / list_*s / remove_*``)
that is one-line sugar over the generic ``Folder.add_folder / get_folder
/ has_folder / list_folders / remove_folder``. All ``add_*`` verbs are
**idempotent** on slugified name (return the cached / on-disk instance
on collision); strict ``get_*`` raises typed ``*NotFoundError``.

The legacy PascalCase / ``create_*`` / lowercase-strict-get / ``delete_*``
triplets are retired by the same spec.
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


# ── Workspace CRUD ─────────────────────────────────────────────────────────


def test_workspace_add_project_is_idempotent(tmp_path):
    ws = Workspace(root=tmp_path, name="ws")
    p1 = ws.add_project(name="demo")
    p2 = ws.add_project(name="demo")
    assert p1 is p2
    assert p1.name == "demo"


def test_workspace_get_project_is_strict_getter(tmp_path):
    ws = Workspace(root=tmp_path, name="ws")
    ws.add_project(name="demo")
    got = ws.get_project("demo")
    assert got.name == "demo"


def test_workspace_get_project_raises_on_missing(tmp_path):
    ws = Workspace(root=tmp_path, name="ws")
    with pytest.raises(ProjectNotFoundError) as exc:
        ws.get_project("never-created")
    assert "never-created" in str(exc.value)


def test_workspace_exposes_typed_semantic_sugar(tmp_path):
    ws = Workspace(root=tmp_path, name="ws")
    assert hasattr(ws, "add_project")
    assert hasattr(ws, "get_project")
    assert hasattr(ws, "has_project")
    assert hasattr(ws, "list_projects")
    assert hasattr(ws, "remove_project")


# ── Project CRUD ───────────────────────────────────────────────────────────


def test_project_add_experiment_is_idempotent(tmp_path):
    ws = Workspace(root=tmp_path, name="ws")
    proj = ws.add_project(name="demo")
    e1 = proj.add_experiment(name="counter")
    e2 = proj.add_experiment(name="counter")
    assert e1 is e2


def test_project_get_experiment_is_strict_getter(tmp_path):
    ws = Workspace(root=tmp_path, name="ws")
    proj = ws.add_project(name="demo")
    proj.add_experiment(name="counter")
    got = proj.get_experiment("counter")
    assert got.name == "counter"


def test_project_get_experiment_raises_on_missing(tmp_path):
    ws = Workspace(root=tmp_path, name="ws")
    proj = ws.add_project(name="demo")
    with pytest.raises(ExperimentNotFoundError):
        proj.get_experiment("never-created")


def test_project_exposes_typed_semantic_sugar(tmp_path):
    ws = Workspace(root=tmp_path, name="ws")
    proj = ws.add_project(name="demo")
    assert hasattr(proj, "add_experiment")
    assert hasattr(proj, "get_experiment")
    assert hasattr(proj, "has_experiment")
    assert hasattr(proj, "remove_experiment")


# ── Experiment CRUD ────────────────────────────────────────────────────────


def test_experiment_add_run_is_idempotent(tmp_path):
    ws = Workspace(root=tmp_path, name="ws")
    proj = ws.add_project(name="demo")
    exp = proj.add_experiment(name="counter")
    r1 = exp.add_run(params={"lr": 1e-3}, id="r1")
    r2 = exp.add_run(params={"lr": 1e-3}, id="r1")
    assert r1.id == r2.id == "r1"


def test_experiment_get_run_is_strict_getter(tmp_path):
    ws = Workspace(root=tmp_path, name="ws")
    proj = ws.add_project(name="demo")
    exp = proj.add_experiment(name="counter")
    exp.add_run(params={"lr": 1e-3}, id="r1")
    got = exp.get_run("r1")
    assert got.id == "r1"


def test_experiment_get_run_raises_on_missing(tmp_path):
    ws = Workspace(root=tmp_path, name="ws")
    proj = ws.add_project(name="demo")
    exp = proj.add_experiment(name="counter")
    with pytest.raises(RunNotFoundError):
        exp.get_run("never-created")


def test_experiment_exposes_typed_semantic_sugar(tmp_path):
    ws = Workspace(root=tmp_path, name="ws")
    proj = ws.add_project(name="demo")
    exp = proj.add_experiment(name="counter")
    assert hasattr(exp, "add_run")
    assert hasattr(exp, "get_run")
    assert hasattr(exp, "has_run")
    assert hasattr(exp, "remove_run")
