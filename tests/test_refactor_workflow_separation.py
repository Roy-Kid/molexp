"""Tests for the workflow separation refactor.

Covers:
- TaskConfig.phase + engine phase filtering
- MapTask execute
- RunContext.from_run_dir + lazy resolution
- RunContext.get_asset / find_asset
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from molexp.workflow.engine import WorkflowEngine
from molexp.workflow.link import Link
from molexp.workflow.task import Task, TaskConfig
from molexp.workflow.registry import register_task
from molexp.workflow.workflow import Workflow
from molexp.workflow.control.map import MapTask, MapConfig
from molexp.workspace import Workspace, RunContext


# ============================================================================
# Helpers
# ============================================================================


def _make_hierarchy(tmp_path):
    workspace = Workspace(root=tmp_path / "workspace", name="Test")
    workspace.materialize()
    project = workspace.create_project(name="Project")
    experiment = project.create_experiment(name="Experiment")
    return workspace, project, experiment


def _make_run(tmp_path, parameters=None):
    _, _, experiment = _make_hierarchy(tmp_path)
    return experiment.create_run(parameters=parameters or {})


# ============================================================================
# Phase filtering tests (tasks.md 1.x)
# ============================================================================


class PhaseConfig(BaseModel):
    label: str = ""


class PhaseTaskA(Task[PhaseConfig, dict]):
    config_type = PhaseConfig
    inputs = {}
    outputs = {"value": int}

    def execute(self, ctx=None, **inputs):
        return {"value": 1}


class PhaseTaskB(Task[PhaseConfig, dict]):
    config_type = PhaseConfig
    inputs = {"value": int}
    outputs = {"result": int}

    def execute(self, ctx=None, **inputs):
        return {"result": inputs["value"] * 10}


register_task(PhaseTaskA)
register_task(PhaseTaskB)


def test_taskconfig_phase_field():
    tc = TaskConfig(task_id="t1", task_type="foo", config={}, phase="setup")
    assert tc.phase == "setup"

    tc_none = TaskConfig(task_id="t2", task_type="foo", config={})
    assert tc_none.phase is None


def test_phase_filtering_single_phase(tmp_path):
    """Only tasks matching the requested phase (+ phase=None) should run."""
    task_a = PhaseTaskA(label="a")
    task_a.phase = "setup"
    task_b = PhaseTaskB(label="b")
    task_b.phase = "compute"

    workflow = Workflow.from_tasks(
        tasks=[task_a, task_b],
        links=[
            Link(
                source=task_a.task_id,
                target=task_b.task_id,
                mapping={"value": "value"},
            )
        ],
    )

    run = _make_run(tmp_path)
    with run.start() as ctx:
        results = WorkflowEngine(workflow).execute(ctx, phases=["setup"])

    # Only task_a should have executed
    assert task_a.task_id in results
    assert task_b.task_id not in results


def test_no_phase_runs_all(tmp_path):
    """When phases=None, all tasks run."""
    task_a = PhaseTaskA(label="a")
    task_a.phase = "setup"
    task_b = PhaseTaskB(label="b")
    task_b.phase = "compute"

    workflow = Workflow.from_tasks(
        tasks=[task_a, task_b],
        links=[
            Link(
                source=task_a.task_id,
                target=task_b.task_id,
                mapping={"value": "value"},
            )
        ],
    )

    run = _make_run(tmp_path)
    with run.start() as ctx:
        results = WorkflowEngine(workflow).execute(ctx, phases=None)

    assert task_a.task_id in results
    assert task_b.task_id in results
    assert results[task_b.task_id]["result"] == 10


def test_cross_phase_persisted_results(tmp_path):
    """Phase 2 can read results persisted by phase 1 from run.json."""
    task_a = PhaseTaskA(label="a")
    task_a.phase = "setup"
    task_b = PhaseTaskB(label="b")
    task_b.phase = "compute"

    workflow = Workflow.from_tasks(
        tasks=[task_a, task_b],
        links=[
            Link(
                source=task_a.task_id,
                target=task_b.task_id,
                mapping={"value": "value"},
            )
        ],
    )

    run = _make_run(tmp_path)

    # Phase 1: run setup
    with run.start() as ctx:
        WorkflowEngine(workflow).execute(ctx, phases=["setup"])

    # Phase 2: run compute (should pick up task_a results from run.json)
    with run.start() as ctx:
        results = WorkflowEngine(workflow).execute(ctx, phases=["compute"])

    assert task_b.task_id in results
    assert results[task_b.task_id]["result"] == 10


def test_class_level_phase_declaration(tmp_path):
    """Tasks can declare phase at class level."""

    class SetupTask(Task[PhaseConfig, dict]):
        config_type = PhaseConfig
        inputs = {}
        outputs = {"value": int}
        phase = "setup"  # Class-level declaration

        def execute(self, ctx=None, **inputs):
            return {"value": 42}

    class ComputeTask(Task[PhaseConfig, dict]):
        config_type = PhaseConfig
        inputs = {"value": int}
        outputs = {"result": int}
        phase = "compute"  # Class-level declaration

        def execute(self, ctx=None, **inputs):
            return {"result": inputs["value"] * 2}

    register_task(SetupTask)
    register_task(ComputeTask)

    task_a = SetupTask(label="setup")
    task_b = ComputeTask(label="compute")

    workflow = Workflow.from_tasks(
        tasks=[task_a, task_b],
        links=[
            Link(
                source=task_a.task_id,
                target=task_b.task_id,
                mapping={"value": "value"},
            )
        ],
    )

    run = _make_run(tmp_path)
    with run.start() as ctx:
        results = WorkflowEngine(workflow).execute(ctx, phases=["setup"])

    # Only setup task should execute
    assert task_a.task_id in results
    assert task_b.task_id not in results
    assert results[task_a.task_id]["value"] == 42


def test_instance_phase_overrides_class_phase():
    """Instance-level phase assignment overrides class-level default."""

    class DefaultPhaseTask(Task[PhaseConfig, dict]):
        config_type = PhaseConfig
        inputs = {}
        outputs = {"value": int}
        phase = "default"  # Class-level default

        def execute(self, ctx=None, **inputs):
            return {"value": 100}

    register_task(DefaultPhaseTask)

    task = DefaultPhaseTask(label="test")
    assert task.phase == "default"  # Initially has class-level phase

    task.phase = "custom"  # Instance override
    assert task.phase == "custom"  # Now has instance-level phase

    workflow = Workflow.from_tasks(tasks=[task], links=[])

    # Verify TaskConfig has the overridden phase
    assert workflow.task_configs[0].phase == "custom"


def test_phase_inheritance():
    """Subclass inherits parent's phase if not overridden."""

    class BasePhaseTask(Task[PhaseConfig, dict]):
        config_type = PhaseConfig
        inputs = {}
        outputs = {"value": int}
        phase = "inherited"

        def execute(self, ctx=None, **inputs):
            return {"value": 1}

    class DerivedTask(BasePhaseTask):
        # Does not override phase, should inherit "inherited"
        def execute(self, ctx=None, **inputs):
            return {"value": 2}

    class OverriddenDerivedTask(BasePhaseTask):
        # Overrides phase
        phase = "overridden"

        def execute(self, ctx=None, **inputs):
            return {"value": 3}

    register_task(BasePhaseTask)
    register_task(DerivedTask)
    register_task(OverriddenDerivedTask)

    # Test inheritance
    derived = DerivedTask(label="derived")
    assert derived.phase == "inherited"

    # Test override
    overridden = OverriddenDerivedTask(label="overridden")
    assert overridden.phase == "overridden"

    # Verify workflow serialization
    workflow = Workflow.from_tasks(tasks=[derived, overridden], links=[])
    assert workflow.task_configs[0].phase == "inherited"
    assert workflow.task_configs[1].phase == "overridden"


# ============================================================================
# MapTask tests (tasks.md 2.x)
# ============================================================================


class SimpleConfig(BaseModel):
    prefix: str = "job"


class SimpleTask(Task[SimpleConfig, dict]):
    config_type = SimpleConfig
    inputs = {}
    outputs = {}

    def execute(self, ctx=None, **inputs):
        return {"result": f"{self.config.prefix}_{inputs['value']}"}


register_task(SimpleTask)


def test_map_task_execute(tmp_path):
    """MapTask.execute() runs base_task for each item."""
    base = SimpleTask(prefix="x")
    map_task = MapTask(base, map_over="items")

    run = _make_run(tmp_path)
    with run.start() as ctx:
        result = map_task.execute(ctx=ctx, items=[{"value": 1}, {"value": 2}])

    assert result["results"][0]["result"] == "x_1"
    assert result["results"][1]["result"] == "x_2"


# ============================================================================
# RunContext.from_run_dir tests (tasks.md 3.x)
# ============================================================================


def test_from_run_dir_reconstructs_hierarchy(tmp_path):
    """from_run_dir should reconstruct the full hierarchy from disk."""
    workspace, project, experiment = _make_hierarchy(tmp_path)
    run = experiment.create_run(parameters={"lr": 0.001})

    # Execute to create run.json with context
    with run.start() as ctx:
        ctx.set_result("test_key", {"data": 42})

    # Now reconstruct from run_dir
    run_dir = (
        workspace.root / "projects" / project.id
        / "experiments" / experiment.id / "runs" / run.id
    )
    restored = RunContext.from_run_dir(run_dir)

    assert restored.run.id == run.id
    assert restored.run.experiment.id == experiment.id
    assert restored.run.experiment.project.id == project.id
    assert restored.run.experiment.project.workspace.root == workspace.root
    assert restored.work_dir == run_dir


def test_from_run_dir_restores_context(tmp_path):
    """from_run_dir should restore context.results from run.json."""
    workspace, project, experiment = _make_hierarchy(tmp_path)
    run = experiment.create_run(parameters={"batch_size": 32})

    with run.start() as ctx:
        ctx.set_result("my_result", {"value": 99})

    run_dir = (
        workspace.root / "projects" / project.id
        / "experiments" / experiment.id / "runs" / run.id
    )
    restored = RunContext.from_run_dir(run_dir)

    assert restored.context.results.get("my_result") == {"value": 99}
    assert restored.run.parameters == {"batch_size": 32}


def test_from_run_dir_supports_context_manager(tmp_path):
    """from_run_dir result should work as context manager."""
    workspace, project, experiment = _make_hierarchy(tmp_path)
    run = experiment.create_run(parameters={})

    with run.start() as ctx:
        pass

    run_dir = (
        workspace.root / "projects" / project.id
        / "experiments" / experiment.id / "runs" / run.id
    )
    restored = RunContext.from_run_dir(run_dir)

    with restored as ctx:
        ctx.set_result("new_key", "new_value")

    # Verify run.json was updated
    run_json = run_dir / "run.json"
    data = json.loads(run_json.read_text())
    assert "new_key" in data["context"]["results"]


def test_from_run_dir_missing_run_json(tmp_path):
    """from_run_dir should raise FileNotFoundError if run.json is missing."""
    with pytest.raises(FileNotFoundError, match="run.json"):
        RunContext.from_run_dir(tmp_path)


# ============================================================================
# RunContext.get_asset / find_asset tests (tasks.md 4.x)
# ============================================================================


def test_get_asset_by_scope(tmp_path):
    """get_asset delegates to the correct scope's AssetLibrary."""
    workspace, project, experiment = _make_hierarchy(tmp_path)

    # Create a file to use as asset source
    asset_src = tmp_path / "data.txt"
    asset_src.write_text("test data")

    # Register asset at project level
    project.assets.import_asset("my_dataset", asset_src)

    run = experiment.create_run(parameters={})
    with run.start() as ctx:
        asset = ctx.get_asset("my_dataset", scope="project")
        assert asset is not None
        assert asset.name == "my_dataset"

        # Asset not at experiment level
        assert ctx.get_asset("my_dataset", scope="experiment") is None


def test_find_asset_fallback(tmp_path):
    """find_asset searches experiment → project → workspace."""
    workspace, project, experiment = _make_hierarchy(tmp_path)

    asset_src = tmp_path / "global.txt"
    asset_src.write_text("global")

    # Register at workspace level only
    workspace.assets.import_asset("shared_model", asset_src)

    run = experiment.create_run(parameters={})
    with run.start() as ctx:
        # find_asset should find it at workspace level
        asset = ctx.find_asset("shared_model")
        assert asset is not None
        assert asset.name == "shared_model"

        # Non-existent asset
        assert ctx.find_asset("nonexistent") is None


def test_get_asset_invalid_scope(tmp_path):
    """get_asset should raise ValueError for unknown scope."""
    run = _make_run(tmp_path)
    with run.start() as ctx:
        with pytest.raises(ValueError, match="Unknown scope"):
            ctx.get_asset("anything", scope="galaxy")
