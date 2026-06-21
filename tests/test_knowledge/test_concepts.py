"""Tests for the OKF storage hierarchy on ``molexp.knowledge``.

``Workspace → Project → Experiment → Run`` are typed Concept subclasses of
``Folder`` with typed semantic sugar (``add_project`` / ``add_experiment`` /
``add_run`` …) mirroring the ``workspace.Folder`` family — but on the OKF
substrate, idempotent on slug, with each level's ``meta.yaml`` carrying the
matching ``type``.
"""

from __future__ import annotations

from pathlib import Path

from molexp.knowledge import Experiment, Folder, Project, Run, Workspace


def test_default_type_classvars() -> None:
    assert Workspace.DEFAULT_TYPE == "workspace"
    assert Project.DEFAULT_TYPE == "project"
    assert Experiment.DEFAULT_TYPE == "experiment"
    assert Run.DEFAULT_TYPE == "run"
    assert issubclass(Project, Folder)


def test_hierarchy_chain_types_meta_and_nesting(tmp_path: Path) -> None:
    ws = Workspace(name="lab", root=tmp_path)
    project = ws.add_project("QM9")
    experiment = project.add_experiment("baseline")
    run = experiment.add_run("r1")

    # right subclass at each level
    assert isinstance(project, Project)
    assert isinstance(experiment, Experiment)
    assert isinstance(run, Run)

    # each level's meta.yaml carries the matching type (workspace included)
    assert ws.read_meta().type == "workspace"
    assert project.read_meta().type == "project"
    assert experiment.read_meta().type == "experiment"
    assert run.read_meta().type == "run"

    # nesting is correct (slugified path identity)
    assert Path(run.resolve()) == tmp_path / "lab" / "qm9" / "baseline" / "r1"


def test_typed_getters_honor_name_and_slug(tmp_path: Path) -> None:
    ws = Workspace(name="lab", root=tmp_path)
    ws.add_project("QM9")

    assert ws.has_project("QM9")
    assert ws.has_project("qm9")
    assert isinstance(ws.get_project("QM9"), Project)
    assert [p.name for p in ws.list_projects()] == ["qm9"]


def test_add_is_idempotent_on_slug(tmp_path: Path) -> None:
    ws = Workspace(name="lab", root=tmp_path)
    first = ws.add_project("My Project")
    second = ws.add_project("my-project")  # same slug
    assert Path(first.resolve()) == Path(second.resolve())
    assert len(ws.list_projects()) == 1


def test_remove_typed(tmp_path: Path) -> None:
    project = Project(name="p", root=tmp_path)
    project.add_experiment("doomed")
    assert project.has_experiment("doomed")
    project.remove_experiment("doomed")
    assert not project.has_experiment("doomed")


def test_list_typed_filters_to_subtype(tmp_path: Path) -> None:
    experiment = Experiment(name="e", root=tmp_path)
    experiment.add_run("r1")
    experiment.add_run("r2")
    # a non-run child mounted via the generic verb is excluded from list_runs
    experiment.add_folder("notes", concept_type="folder")

    runs = experiment.list_runs()
    assert all(isinstance(r, Run) for r in runs)
    assert {r.name for r in runs} == {"r1", "r2"}
