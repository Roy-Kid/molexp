"""Tests for Experiment workflow binding (previously the user-facing spec)."""

import json

import pytest

from molexp.workflow.spec import WorkflowBuilder, WorkflowSpec
from molexp.workspace import Workspace
from molexp.workspace.experiment import _promote_to_workflow


@pytest.fixture
def workspace(tmp_path):
    return Workspace(root=tmp_path / "lab", name="Test")


@pytest.fixture
def project(workspace):
    return workspace.project("test-project")


@pytest.fixture
def experiment(project):
    return project.experiment(
        "test-experiment",
        params={"lr": 1e-3},
        n_replicas=2,
    )


class TestExperimentConstruction:
    def test_stores_fields(self, project):
        exp = project.experiment(
            "sweep",
            params={"x": 1},
            n_replicas=3,
        )
        assert exp.name == "sweep"
        assert exp.project is project
        assert exp.n_replicas == 3

    def test_default_values(self, project):
        exp = project.experiment("minimal")
        assert exp.params == {}
        assert exp.n_replicas == 1
        assert exp.description == ""
        assert exp.tags == []
        assert exp.seeds is None

    def test_workflow_initially_none(self, experiment):
        assert experiment.workflow is None


class TestSetWorkflow:
    def test_callable_promotes_to_workflow_spec(self, experiment):
        def train(ctx):
            pass

        experiment.set_workflow(train)
        assert isinstance(experiment.workflow, WorkflowSpec)

    def test_workflow_spec_stored_directly(self, experiment):
        spec = WorkflowBuilder(name="test").build()
        experiment.set_workflow(spec)
        assert experiment.workflow is spec

    def test_workflow_always_workflow_spec_type(self, experiment):
        def train(ctx):
            pass

        experiment.set_workflow(train)
        assert type(experiment.workflow) is WorkflowSpec

    def test_non_callable_raises_type_error(self, experiment):
        with pytest.raises(TypeError, match="Expected WorkflowSpec or callable"):
            experiment.set_workflow(42)

    def test_double_set_raises_value_error(self, experiment):
        experiment.set_workflow(lambda ctx: None)
        with pytest.raises(ValueError, match="already has a workflow"):
            experiment.set_workflow(lambda ctx: None)


class TestGetSeeds:
    def test_default_seeds(self, project):
        exp = project.experiment("t1", n_replicas=3)
        seeds = exp.get_seeds()
        assert len(seeds) == 3
        assert seeds[0] == 42

    def test_explicit_seeds(self, project):
        exp = project.experiment("t2", n_replicas=2, seeds=[100, 200, 300])
        seeds = exp.get_seeds()
        assert seeds == [100, 200]

    def test_seeds_extended_when_short(self, project):
        exp = project.experiment("t3", n_replicas=8)
        seeds = exp.get_seeds()
        assert len(seeds) == 8
        assert len(set(seeds)) == 8


class TestPromoteToWorkflow:
    def test_creates_workflow_spec(self):
        def my_func(ctx):
            pass

        spec = _promote_to_workflow(my_func, "test")
        assert isinstance(spec, WorkflowSpec)
        assert spec.name == "test"

    @pytest.mark.asyncio
    async def test_promoted_run_context_function_receives_profile_config(self, tmp_path):
        from molexp.config import ProfileConfig

        def train(ctx):
            assert ctx.config.name == "dry_run"
            assert ctx.config["skip_heavy"] is True
            ctx.set_result("profile", ctx.config.name)

        spec = _promote_to_workflow(train, "test")

        ws = Workspace(root=tmp_path / "lab", name="Test Lab")
        project = ws.project("demo")
        experiment = project.experiment("runtime")
        run = experiment.run(parameters={})

        profile_cfg = ProfileConfig({"skip_heavy": True}, name="dry-run")
        result = await spec.execute(run=run, profile_config=profile_cfg)

        assert result.status == "completed"
        run_json = json.loads((run.run_dir / "run.json").read_text())
        assert run_json["context"]["results"]["profile"] == "dry_run"
