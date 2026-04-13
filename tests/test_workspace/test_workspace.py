"""Tests for Workspace / Project / Experiment / Run hierarchy."""

import json

from molexp.workspace import Workspace


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
        ws.project("first")
        assert (tmp_path / "workspace.json").exists()

    def test_load_preserves_metadata(self, workspace):
        workspace.materialize()
        loaded = Workspace.load(workspace.root)
        assert loaded.id == workspace.id
        assert loaded.name == workspace.name

    def test_from_path_creates_if_missing(self, tmp_path):
        ws = Workspace.from_path(tmp_path / "auto")
        ws.materialize()
        assert ws.root.exists()
        assert (ws.root / "workspace.json").exists()

    def test_metadata_has_no_child_lists(self, workspace):
        workspace.materialize()
        data = json.loads((workspace.root / "workspace.json").read_text())
        assert "projects" not in data


class TestProject:
    def test_creation(self, workspace):
        proj = workspace.project("QM9")
        assert proj.id == "qm9"
        assert proj.name == "QM9"

    def test_get_by_id(self, workspace, project):
        found = workspace.get_project(project.id)
        assert found is not None
        assert found.id == project.id

    def test_get_by_name_slugified(self, workspace):
        workspace.project("My Project")
        found = workspace.get_project("My Project")
        assert found is not None
        assert found.id == "my-project"

    def test_list_projects(self, workspace):
        workspace.project("a")
        workspace.project("b")
        assert len(workspace.list_projects()) == 2

    def test_delete(self, workspace, project):
        workspace.delete_project(project.id)
        # clear cache after delete (otherwise in-memory stays)
        workspace._projects_cache.pop(project.id, None)
        assert workspace.get_project(project.id) is None

    def test_idempotent_returns_same_instance(self, workspace):
        p1 = workspace.project("dup")
        p2 = workspace.project("dup")
        assert p1 is p2

    def test_metadata_has_no_child_lists(self, project):
        data = json.loads((project.project_dir / "project.json").read_text())
        assert "experiments" not in data
        assert "assets" not in data


class TestExperiment:
    def test_creation_with_workflow(self, project):
        exp = project.experiment(
            "baseline",
            workflow_source="train.py",
            params={"lr": 1e-4},
            git_commit="abc",
        )
        assert exp.metadata.workflow_source == "train.py"
        assert exp.metadata.parameter_space == {"lr": 1e-4}
        assert exp.metadata.git_commit == "abc"

    def test_parent_references(self, experiment):
        assert experiment.project is not None
        assert experiment.workspace is experiment.project.workspace

    def test_list_experiments(self, project):
        project.experiment("a")
        project.experiment("b")
        assert len(project.list_experiments()) == 2

    def test_get_experiment(self, project, experiment):
        found = project.get_experiment(experiment.id)
        assert found is not None
        assert found.name == experiment.name

    def test_idempotent(self, project):
        e1 = project.experiment("same")
        e2 = project.experiment("same")
        assert e1 is e2


class TestRun:
    def test_creation(self, experiment):
        run = experiment.run(parameters={"x": 1})
        assert run.parameters == {"x": 1}
        assert run.status == "pending"

    def test_workflow_snapshot_auto_captured(self, experiment):
        run = experiment.run()
        snap = run.metadata.workflow_snapshot
        assert snap is not None
        assert snap.source == "train.py"
        assert snap.git_commit == "abc123"

    def test_list_runs(self, experiment):
        experiment.run(parameters={"x": 1})
        experiment.run(parameters={"x": 2})
        assert len(experiment.list_runs()) == 2

    def test_reload_from_disk(self, experiment):
        run = experiment.run(parameters={"x": 42})
        reloaded = experiment.get_run(run.id)
        assert reloaded.parameters == {"x": 42}
        assert reloaded.metadata.workflow_snapshot.source == "train.py"
