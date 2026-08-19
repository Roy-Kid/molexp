"""Regression tests for ``Folder.from_disk`` disk inheritance.

The disk lives on :attr:`Workspace.fs`. Children must resolve the same
backend via :meth:`Folder._disk` after reload — they must not store their
own FileSystem (that was the mid-2026 ``from_disk`` drop bug, now
impossible because there is nothing to drop).
"""

from __future__ import annotations

import molexp as me


class TestFromDiskPreservesDisk:
    def test_project_reload_then_add_experiment(self, tmp_path):
        ws = me.Workspace(tmp_path)
        ws.add_project("p")

        ws2 = me.Workspace(tmp_path)
        proj2 = ws2.add_project("p")
        assert proj2._disk() is ws2.fs
        assert not hasattr(proj2, "fs")
        exp = proj2.add_experiment("e", params={})
        assert exp._disk() is ws2.fs

    def test_experiment_reload_then_add_run(self, tmp_path):
        ws = me.Workspace(tmp_path)
        ws.add_project("p").add_experiment("e", params={})

        ws2 = me.Workspace(tmp_path)
        exp2 = ws2.add_project("p").add_experiment("e", params={})
        assert exp2._disk() is ws2.fs
        run = exp2.add_run({})
        assert run._disk() is ws2.fs

    def test_run_reload_via_get_run_preserves_disk(self, tmp_path):
        ws = me.Workspace(tmp_path)
        run = ws.add_project("p").add_experiment("e", params={}).add_run({"k": "v"})
        run_id = run.id

        ws2 = me.Workspace(tmp_path)
        exp2 = ws2.add_project("p").add_experiment("e", params={})
        run2 = exp2.get_run(run_id)
        assert run2._disk() is ws2.fs
