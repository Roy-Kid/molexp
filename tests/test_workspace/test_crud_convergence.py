"""``add_*`` CRUD convergence across the ``Folder`` family.

``Experiment.add_run`` / ``Project.add_experiment`` / ``Workspace.add_project``
route through the single ``Folder._construct_child`` + ``add_folder`` path.
``add_*`` is idempotent on the slugified id — a hard workspace invariant
(CLAUDE.md): the same slug must resolve to the same node whether warmed from
the in-memory child cache or reloaded on disk by a fresh instance.
"""

from __future__ import annotations

from molexp.workspace import Workspace


class TestAddStarConvergence:
    def test_same_slug_returns_identical_instance_in_memory(self, tmp_path) -> None:
        """Re-adding a slug on a warm instance returns the cached node at every
        layer (``is`` identity)."""
        ws = Workspace(root=tmp_path / "lab", name="lab")

        assert ws.add_project("demo") is ws.add_project("demo")

        proj = ws.add_project("demo")
        assert proj.add_experiment("baseline") is proj.add_experiment("baseline")

        exp = proj.add_experiment("baseline")
        run = exp.add_run(params={"seed": 1}, id="run-x")
        assert exp.add_run(id="run-x") is run

    def test_same_slug_converges_on_disk_across_fresh_instances(self, tmp_path) -> None:
        """A second instance re-adding a slug loads the on-disk child rather than
        creating a duplicate directory (the ``add_folder`` on-disk path)."""
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
        assert exp2.add_run(id="run-x").id == "run-x"
        assert len(proj2.list_experiments()) == 1
        assert len(exp2.list_runs()) == 1
