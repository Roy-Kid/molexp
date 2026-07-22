"""Regression tests for ``Folder.from_disk`` ``_fs`` preservation.

Mid-2026 bug: subclass ``from_disk`` overrides silently dropped ``_fs`` from
the reconstructed attrs dict, so a disk-reloaded entity could not compute its
children's paths (``AttributeError: 'Project' object has no attribute '_fs'``).
Every Folder subclass that overrides ``from_disk`` must route through
``Folder.base_from_disk_attrs`` — one guard per subclass override site
(Project / Experiment / Run) so any one regressing is caught independently.
"""

from __future__ import annotations

import molexp as me


class TestFromDiskPreservesFs:
    def test_project_reload_then_add_experiment(self, tmp_path):
        """Project reloaded via ``from_disk`` keeps ``_fs`` (the original bug:
        ``add_experiment`` on a reloaded Project used to ``AttributeError``)."""
        ws = me.Workspace(tmp_path)
        ws.add_project("p")  # materializes projects/p/ + project.json

        ws2 = me.Workspace(tmp_path)  # fresh cache → add_project goes through from_disk
        proj2 = ws2.add_project("p")
        assert proj2._fs is ws2._fs, "Project._fs must inherit from parent on reload"
        exp = proj2.add_experiment("e", params={})
        assert exp._fs is ws2._fs

    def test_experiment_reload_then_add_run(self, tmp_path):
        """Experiment reloaded via ``from_disk`` keeps ``_fs`` for child runs."""
        ws = me.Workspace(tmp_path)
        ws.add_project("p").add_experiment("e", params={})

        ws2 = me.Workspace(tmp_path)  # all three levels reload from disk
        exp2 = ws2.add_project("p").add_experiment("e", params={})
        assert exp2._fs is ws2._fs
        run = exp2.add_run({})
        assert run._fs is ws2._fs

    def test_run_reload_via_get_run_preserves_fs(self, tmp_path):
        """A Run reloaded through ``get_run`` (Run.from_disk) keeps ``_fs``."""
        ws = me.Workspace(tmp_path)
        run = ws.add_project("p").add_experiment("e", params={}).add_run({"k": "v"})
        run_id = run.id

        ws2 = me.Workspace(tmp_path)
        exp2 = ws2.add_project("p").add_experiment("e", params={})
        run2 = exp2.get_run(run_id)
        assert run2._fs is ws2._fs
