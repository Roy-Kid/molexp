"""Tests for Workspace / Project / Experiment / Run hierarchy."""

import json
from pathlib import Path

import pytest

from molexp.workspace import Workspace


class TestLegacyLibraryRemoved:
    """wsokf-11: the legacy per-scope Library stack is gone."""

    def test_no_library_property_on_folders(self, workspace):
        project = workspace.add_project("p")
        experiment = project.add_experiment("e")
        run = experiment.add_run({"x": 1})
        for scope in (workspace, project, experiment, run):
            assert not hasattr(scope, "library")

    def test_legacy_symbols_not_importable_from_workspace(self):
        import molexp.workspace as ws

        for name in ("Library", "LibraryIndex", "NoteEntry", "NoteAsset", "ReferenceStore"):
            assert not hasattr(ws, name)
            assert name not in ws.__all__

    def test_library_subpackage_gone(self):
        with pytest.raises(ModuleNotFoundError):
            import molexp.workspace.library  # noqa: F401

    def test_note_kind_no_longer_parses(self):
        from molexp.workspace.assets import parse_asset

        with pytest.raises(Exception):  # noqa: B017 - pydantic ValidationError
            parse_asset({"kind": "note", "asset_id": "x", "name": "n", "path": "p.md"})


class TestWorkspace:
    def test_creation_no_side_effects(self, tmp_path):
        Workspace(root=tmp_path / "new", name="Lab")
        assert not (tmp_path / "new" / "workspace.json").exists()

    def test_materialize_creates_files(self, tmp_path):
        ws = Workspace(root=tmp_path, name="Lab")
        ws.materialize()
        assert (tmp_path / "workspace.json").exists()

    def test_child_factory_auto_materializes(self, tmp_path):
        ws = Workspace(root=tmp_path, name="Lab")
        ws.add_project("first")
        assert (tmp_path / "workspace.json").exists()

    def test_load_preserves_metadata(self, workspace):
        workspace.materialize()
        loaded = Workspace.load(workspace.root)
        assert loaded.id == workspace.id
        assert loaded.name == workspace.name

    def test_constructor_creates_if_missing(self, tmp_path):
        ws = Workspace(tmp_path / "auto")
        ws.materialize()
        assert Path(ws.root).exists()
        assert Path(ws.root / "workspace.json").exists()

    def test_metadata_has_no_child_lists(self, workspace):
        workspace.materialize()
        data = json.loads(Path(workspace.root / "workspace.json").read_text())
        assert "projects" not in data


class TestProject:
    def test_creation(self, workspace):
        proj = workspace.add_project("QM9")
        assert proj.id == "qm9"
        assert proj.name == "QM9"

    def test_get_by_id(self, workspace, project):
        found = workspace.get_project(project.id)
        assert found.id == project.id

    def test_get_by_name_slugified(self, workspace):
        workspace.add_project("My Project")
        found = workspace.get_project("My Project")
        assert found.id == "my-project"

    def test_list_projects(self, workspace):
        workspace.add_project("a")
        workspace.add_project("b")
        assert len(workspace.list_projects()) == 2

    def test_sync_folders_imports_orphan_dirs_into_index(self, tmp_path):
        """``sync_folders`` reconciles the per-class index with disk reality.

        External tooling (rsync, manual rm, legacy migration) may leave
        directories that ``add_*`` never indexed. ``sync_folders`` is the
        explicit reconciliation hook; without it, the index stays
        authoritative for ``list_*``.
        """
        from molexp.workspace import Project

        # Orphan project dir left by external tooling (not via add_project).
        orphan = tmp_path / "projects" / "orphan"
        orphan.mkdir(parents=True)
        (orphan / "project.json").write_text(
            '{"id":"orphan","name":"orphan","description":"","owner":"",'
            '"tags":[],"config":{},"created_at":"2026-04-21T12:00:00"}'
        )
        ws = Workspace(tmp_path)
        ws.add_project("registered")
        # Index is authoritative: ``list_projects`` sees only what was
        # added through the API. Orphan is invisible until sync.
        assert {p.id for p in ws.list_projects()} == {"registered"}
        # Reconcile.
        ws.sync_folders(cls=Project)
        assert {p.id for p in ws.list_projects()} == {"orphan", "registered"}

    def test_delete(self, workspace, project):
        from molexp.workspace import ProjectNotFoundError

        workspace.remove_project(project.id)
        # remove_folder evicts the single _children_cache; no manual clear needed.
        with pytest.raises(ProjectNotFoundError):
            workspace.get_project(project.id)

    def test_idempotent_returns_same_instance(self, workspace):
        p1 = workspace.add_project("dup")
        p2 = workspace.add_project("dup")
        assert p1 is p2

    def test_metadata_has_no_child_lists(self, project):
        data = json.loads((Path(project.project_dir) / "project.json").read_text())
        assert "experiments" not in data
        assert "assets" not in data


class TestExperiment:
    def test_creation_with_workflow(self, project):
        exp = project.add_experiment(
            "baseline",
            workflow_source="train.py",
            params={"lr": 1e-4},
            git_commit="abc",
        )
        assert exp.metadata.workflow_source == "train.py"
        assert exp.metadata.parameter_space == {"lr": 1e-4}
        assert exp.metadata.git_commit == "abc"

    def test_ir_workflow_source_externalized_to_workflow_json(self, project):
        """A compiled-IR ``workflow_source`` lands as a standalone ``workflow.json``.

        The IR is the contract the molexp VSCode preview reads directly, so it
        must be a clean, pretty-printed file (no ``schema_version`` envelope) and
        must have a single on-disk home — stripped from the embedded
        ``experiment.json`` field, rehydrated from the file on reload.
        """
        ir = {"workflow_id": "wf", "name": "demo", "task_configs": [], "links": []}
        exp = project.add_experiment("ir-exp", workflow_source=json.dumps(ir))

        doc = Path(exp.experiment_dir) / "workflow.json"
        assert doc.is_file()
        # Clean IR — directly previewable, no version-envelope pollution.
        assert json.loads(doc.read_text()) == ir
        # Single home: the embedded field is cleared in experiment.json.
        raw = json.loads((Path(exp.experiment_dir) / "experiment.json").read_text())
        assert raw["workflow_source"] is None
        # File is canonical: reload rehydrates the in-memory field from it.
        reloaded = project.get_experiment("ir-exp")
        assert json.loads(reloaded.metadata.workflow_source) == ir

    def test_non_ir_workflow_source_stays_embedded(self, project):
        """A non-JSON ``workflow_source`` (a Python path) is never externalized."""
        exp = project.add_experiment("py-exp", workflow_source="train.py")
        assert not (Path(exp.experiment_dir) / "workflow.json").exists()
        raw = json.loads((Path(exp.experiment_dir) / "experiment.json").read_text())
        assert raw["workflow_source"] == "train.py"

    def test_parent_references(self, experiment):
        assert experiment.project is not None
        assert experiment.workspace is experiment.project.workspace

    def test_list_experiments(self, project):
        project.add_experiment("a")
        project.add_experiment("b")
        assert len(project.list_experiments()) == 2

    def test_sync_folders_imports_orphan_experiment_dirs(self, workspace):
        """``project.sync_folders(cls=Experiment)`` reconciles drift."""
        from molexp.workspace import Experiment

        project = workspace.add_project("p")
        # Orphan experiment dir left by external tooling.
        orphan = Path(project.project_dir) / "experiments" / "orphan"
        orphan.mkdir(parents=True)
        (orphan / "experiment.json").write_text(
            '{"id":"orphan","name":"orphan","parameter_space":{},'
            '"n_replicas":1,"seeds":[],"workflow_source":null,'
            '"workflow_type":null,"git_commit":null,"description":"",'
            '"tags":[],"created_at":"2026-04-21T12:00:00"}'
        )
        project.add_experiment("kept")
        # Index is authoritative — orphan invisible until sync.
        assert {e.id for e in project.list_experiments()} == {"kept"}
        project.sync_folders(cls=Experiment)
        assert {e.id for e in project.list_experiments()} == {"orphan", "kept"}

    def test_get_experiment(self, project, experiment):
        found = project.get_experiment(experiment.id)
        assert found.name == experiment.name

    def test_idempotent(self, project):
        e1 = project.add_experiment("same")
        e2 = project.add_experiment("same")
        assert e1 is e2


class TestRun:
    def test_creation(self, experiment):
        run = experiment.add_run(params={"x": 1})
        assert run.parameters == {"x": 1}
        assert run.status == "pending"

    def test_workflow_snapshot_passed_through_as_dict(self, experiment):
        # workspace no longer auto-captures workflow-snapshot data —
        # the caller (workflow / agent layer) supplies an opaque dict
        # at run-creation time.
        run = experiment.add_run(workflow_snapshot={"source": "train.py", "git_commit": "abc123"})
        snap = run.metadata.workflow_snapshot
        assert snap is not None
        assert snap["source"] == "train.py"
        assert snap["git_commit"] == "abc123"

    def test_list_runs(self, experiment):
        experiment.add_run(params={"x": 1})
        experiment.add_run(params={"x": 2})
        assert len(experiment.list_runs()) == 2

    def test_reload_from_disk(self, experiment):
        run = experiment.add_run(
            params={"x": 42},
            workflow_snapshot={"source": "train.py"},
        )
        reloaded = experiment.get_run(run.id)
        assert reloaded.parameters == {"x": 42}
        assert reloaded.metadata.workflow_snapshot is not None
        assert reloaded.metadata.workflow_snapshot["source"] == "train.py"
