"""OKF capabilities on ``workspace.Folder`` (wsokf-01/02/03).

Every Folder gains a narrative ``index.md`` whose markdown links are the
knowledge graph (``out_edges`` / ``links``) and a sole concept identity file
``meta.json`` (``type`` → registry; path basename = id). Domain entities keep
class-named JSON (``project.json`` / ``run.json`` / …). A ``Run`` additionally
carries the hot-state ``ops/run.json`` sidecar.
"""

from __future__ import annotations

import os
from pathlib import Path

from molexp.workspace import Workspace
from molexp.workspace.models import RunStatus
from molexp.workspace.run_ops import RunOpsState


class TestFolderOKF:
    def test_index_round_trips_and_is_additive(self, tmp_path: Path) -> None:
        ws = Workspace(root=tmp_path / "lab")
        ws.materialize()
        proj = ws.add_project("alpha")
        assert proj.read_index() == ""  # absent → empty
        proj.write_index("# Alpha\n\nnarrative\n")
        assert proj.read_index() == "# Alpha\n\nnarrative\n"
        # additive: the project's own metadata is untouched (still listed)
        assert [p.name for p in ws.list_projects()] == ["alpha"]

    def test_out_edges_resolves_in_tree_and_classifies_external(self, tmp_path: Path) -> None:
        ws = Workspace(root=tmp_path / "lab")
        ws.materialize()
        alpha = ws.add_project("alpha")
        beta = ws.add_project("beta")

        alpha.write_index(
            "# Alpha\n\n- [to-beta](../beta)\n- [ext](https://example.com)\n- [nowhere](./nope)\n"
        )

        edges = {os.path.normpath(e) for e in alpha.out_edges()}
        assert os.path.normpath(str(beta.resolve())) in edges

        scan = alpha.links()
        assert any("example.com" in e for e in scan.external)
        assert any("nope" in o for o in scan.other)

    def test_meta_json_marks_concept_type_path_is_id(self, tmp_path: Path) -> None:
        ws = Workspace(root=tmp_path / "lab")
        ws.materialize()
        proj = ws.add_project("alpha")
        assert ws.read_meta()["type"] == "workspace.root"
        pmeta = proj.read_meta()
        assert pmeta["type"] == "workspace.project"
        # Path basename is identity — id is not required on the marker.
        assert "id" not in pmeta or pmeta.get("id") in (None, "alpha")
        assert not (Path(proj.resolve()) / "metadata.json").exists()


class TestRunOpsSidecar:
    def test_write_read_round_trip_defaults_pending_and_isolates_file(self, tmp_path: Path) -> None:
        ws = Workspace(root=tmp_path / "lab")
        ws.materialize()
        run = ws.add_project("p").add_experiment("e").add_run()
        assert run.read_ops().status == RunStatus.PENDING  # default when absent
        run.write_ops(RunOpsState(status=RunStatus.RUNNING, owner_pid=7))
        assert run.read_ops().status == RunStatus.RUNNING
        assert run.read_ops().owner_pid == 7
        # hot state lands in ops/run.json, isolated from run.json
        assert (Path(run.resolve()) / "ops" / "run.json").exists()

    def test_read_ops_falls_back_to_legacy_underscore_dir(self, tmp_path: Path) -> None:
        ws = Workspace(root=tmp_path / "lab")
        ws.materialize()
        run = ws.add_project("p").add_experiment("e").add_run()
        legacy = Path(run.resolve()) / "_ops"
        legacy.mkdir()
        (legacy / "run.json").write_text(
            '{"status": "running", "owner_pid": 3}',
            encoding="utf-8",
        )
        assert run.read_ops().status == RunStatus.RUNNING
        assert run.read_ops().owner_pid == 3
        run.update_ops(lambda s: s.model_copy(update={"owner_pid": 9}))
        assert (Path(run.resolve()) / "ops" / "run.json").exists()
        assert run.read_ops().owner_pid == 9

    def test_update_ops_read_modify_writes(self, tmp_path: Path) -> None:
        ws = Workspace(root=tmp_path / "lab")
        ws.materialize()
        run = ws.add_project("p").add_experiment("e").add_run()
        run.write_ops(RunOpsState(status=RunStatus.PENDING))
        out = run.update_ops(lambda s: s.model_copy(update={"status": RunStatus.RUNNING}))
        assert out.status == RunStatus.RUNNING
        assert run.read_ops().status == RunStatus.RUNNING


class TestFolderFiles:
    def test_files_is_store_on_workspace_disk(self, tmp_path: Path) -> None:
        ws = Workspace(root=tmp_path / "lab")
        ws.materialize()
        proj = ws.add_project("alpha")
        dest = proj.files.put("note.txt", "hello")
        assert dest.read_text() == "hello"
        assert dest.parent == Path(proj.resolve())
        assert proj._disk() is ws.fs
        assert not hasattr(proj, "fs")
