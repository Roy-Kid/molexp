"""Regression tests for Folder ``from_disk`` invariants.

These tests pin down the bugs we found mid-2026: subclass ``from_disk``
overrides used to silently drop ``_fs`` from the reconstructed attrs
dict, which made any disk-reloaded entity unable to compute its children's
paths (``AttributeError: 'Project' object has no attribute '_fs'``).

Every Folder subclass that overrides ``from_disk`` must route through
:meth:`Folder.base_from_disk_attrs` so this can never silently regress.
"""

from __future__ import annotations

import molexp as me


def test_project_reload_then_add_experiment(tmp_path):
    """Project re-loaded from disk must still know its _fs.

    Reproduces the original bug: Workspace -> add_project goes through
    Project.from_disk (because the project dir already exists), and the
    reconstructed Project then needs ``self._fs`` inside add_experiment
    to compute the child's path.
    """
    ws = me.Workspace(tmp_path)
    ws.add_project("p")  # materializes projects/p/ + project.json
    # New Workspace instance -> children_cache is empty -> goes through from_disk.
    ws2 = me.Workspace(tmp_path)
    proj2 = ws2.add_project("p")
    assert proj2._fs is ws2._fs, "Project._fs must inherit from parent on reload"
    # The real symptom of the original bug: this used to AttributeError.
    exp = proj2.add_experiment("e", params={})
    assert exp._fs is ws2._fs


def test_experiment_reload_then_add_run(tmp_path):
    ws = me.Workspace(tmp_path)
    proj = ws.add_project("p")
    proj.add_experiment("e", params={})
    # Fresh handles -> all three levels reload from disk.
    ws2 = me.Workspace(tmp_path)
    proj2 = ws2.add_project("p")
    exp2 = proj2.add_experiment("e", params={})
    assert exp2._fs is ws2._fs
    run = exp2.add_run({})
    assert run._fs is ws2._fs


def test_run_reload_round_trip(tmp_path):
    ws = me.Workspace(tmp_path)
    proj = ws.add_project("p")
    exp = proj.add_experiment("e", params={})
    run = exp.add_run({"k": "v"})
    run_id = run.id
    ws2 = me.Workspace(tmp_path)
    proj2 = ws2.add_project("p")
    exp2 = proj2.add_experiment("e", params={})
    run2 = exp2.get_run(run_id)
    assert run2._fs is ws2._fs
    assert run2.parameters == {"k": "v"}
