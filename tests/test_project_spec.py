"""Tests for user-facing Project spec."""

import pytest
from pathlib import Path

from molexp.project import Project
from molexp.experiment import Experiment
from molexp.workspace.param import GridSpace


class TestProjectConstruction:
    def test_name_stored(self):
        p = Project("my-project")
        assert p.name == "my-project"

    def test_default_workspace_root(self):
        p = Project("test")
        assert p.workspace_root == Path("./workspace")

    def test_custom_config_dict(self):
        p = Project("test", config={"workspace_root": "/data/lab"})
        assert p.workspace_root == Path("/data/lab")

    def test_no_experiments_initially(self):
        p = Project("test")
        assert p.experiments == []


class TestProjectExperimentFactory:
    def test_returns_experiment(self):
        p = Project("test")
        exp = p.experiment("sweep", params=GridSpace({"lr": [1e-4]}))
        assert isinstance(exp, Experiment)
        assert exp.name == "sweep"

    def test_stores_experiment(self):
        p = Project("test")
        exp = p.experiment("sweep")
        assert exp in p.experiments

    def test_multiple_experiments(self):
        p = Project("test")
        e1 = p.experiment("a")
        e2 = p.experiment("b")
        assert len(p.experiments) == 2
        assert p.experiments[0] is e1
        assert p.experiments[1] is e2

    def test_experiments_returns_copy(self):
        p = Project("test")
        p.experiment("a")
        exps = p.experiments
        exps.clear()
        assert len(p.experiments) == 1

    def test_passes_all_fields(self):
        p = Project("test")
        exp = p.experiment(
            "sweep",
            params=GridSpace({"x": [1, 2]}),
            n_replicas=5,
            description="test desc",
            tags=["a", "b"],
            seeds=[10, 20, 30, 40, 50],
        )
        assert exp.n_replicas == 5
        assert exp.description == "test desc"
        assert exp.tags == ["a", "b"]
        assert exp.project is p
