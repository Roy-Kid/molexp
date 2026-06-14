"""add_* CRUD convergence (workflow-workspace-hardening P1-4 / ac-009).

``Experiment.add_run`` / ``Project.add_experiment`` / ``Workspace.add_project``
were three near-duplicate idempotent dances, two of which also maintained a
redundant typed cache (``_experiments_cache`` / ``_projects_cache``) mirroring
the generic ``_children_cache``. They now route through the single
``Folder._construct_child`` + ``add_folder`` path, and the redundant caches are
gone. Idempotency (same instance per slug) must be unchanged.
"""

from __future__ import annotations

from molexp.workspace import Workspace


def test_redundant_typed_caches_removed(tmp_path):
    """The duplicate per-type caches no longer exist — ``_children_cache`` is
    the single in-memory child index."""
    ws = Workspace(root=tmp_path / "lab", name="lab")
    proj = ws.add_project("demo")

    assert not hasattr(ws, "_projects_cache")
    assert not hasattr(proj, "_experiments_cache")


def test_add_star_idempotent_same_instance(tmp_path):
    """Adding the same slug twice returns the identical cached instance at
    every layer."""
    ws = Workspace(root=tmp_path / "lab", name="lab")

    assert ws.add_project("demo") is ws.add_project("demo")

    proj = ws.add_project("demo")
    assert proj.add_experiment("baseline") is proj.add_experiment("baseline")

    exp = proj.add_experiment("baseline")
    r = exp.add_run(params={"seed": 1}, id="run-x")
    assert exp.add_run(id="run-x") is r


def test_add_star_idempotent_across_fresh_instances(tmp_path):
    """A second process/instance re-adding the same slug loads the on-disk
    child rather than creating a duplicate (the ``add_folder`` on-disk path)."""
    root = tmp_path / "lab"
    ws1 = Workspace(root=root, name="lab")
    proj1 = ws1.add_project("demo")
    exp1 = proj1.add_experiment("baseline")
    exp1.add_run(params={"seed": 1}, id="run-x")

    # Fresh instance — no in-memory caches warmed.
    ws2 = Workspace(root)
    proj2 = ws2.add_project("demo")
    assert proj2.id == proj1.id
    exp2 = proj2.add_experiment("baseline")
    assert exp2.id == exp1.id
    run2 = exp2.add_run(id="run-x")
    assert run2.id == "run-x"
    # No duplicate directories created.
    assert len(proj2.list_experiments()) == 1
    assert len(exp2.list_runs()) == 1
