"""Tests for workflow persistence, replay, and strict mappings."""

from __future__ import annotations

from pydantic import BaseModel
import pytest

from molexp.workflow.engine import WorkflowEngine
from molexp.workflow.link import Link
from molexp.workflow.registry import register_task
from molexp.workflow.task import Task
from molexp.workflow.workflow import Workflow
from molexp.workspace import Workspace


def _make_run(tmp_path):
    workspace = Workspace(root=tmp_path / "workspace", name="Test")
    workspace.materialize()
    project = workspace.create_project(name="Project")
    experiment = project.create_experiment(name="Experiment")
    return experiment.create_run(parameters={})


class AddConfig(BaseModel):
    value: int = 1


class AddTask(Task[AddConfig, dict]):
    config_type = AddConfig
    inputs = {"x": int}
    outputs = {"result": int}

    def execute(self, ctx=None, **inputs):
        return {"result": inputs["x"] + self.config.value}


class MultiplyConfig(BaseModel):
    factor: int = 2


class MultiplyTask(Task[MultiplyConfig, dict]):
    config_type = MultiplyConfig
    inputs = {"value": int}
    outputs = {"result": int}

    def execute(self, ctx=None, **inputs):
        return {"result": inputs["value"] * self.config.factor}


def test_workflow_roundtrip(tmp_path):
    register_task(AddTask)
    register_task(MultiplyTask)

    add = AddTask(value=2)
    mul = MultiplyTask(factor=3)

    link = Link(
        source=add.task_id,
        target=mul.task_id,
        mapping={"result": "value"},
    )
    workflow = Workflow.from_tasks([add, mul], links=[link])
    path = tmp_path / "workflow.json"
    workflow.save(path)

    loaded = Workflow.load(path)
    run = _make_run(tmp_path)

    with run.context() as ctx:
        engine = WorkflowEngine(loaded)
        results = engine.execute(ctx, x=4)

    assert results[add.task_id]["result"] == 6
    assert results[mul.task_id]["result"] == 18


def test_missing_mapping_rejected():
    register_task(AddTask)
    register_task(MultiplyTask)

    add = AddTask(value=2)
    mul = MultiplyTask(factor=3)

    link = Link(source=add.task_id, target=mul.task_id, mapping={})
    workflow = Workflow.from_tasks([add, mul], links=[link])

    with pytest.raises(ValueError, match="must declare a mapping"):
        WorkflowEngine(workflow)


def test_missing_required_input_fails(tmp_path):
    register_task(AddTask)

    add = AddTask(value=2)
    workflow = Workflow.from_tasks([add], links=[])
    run = _make_run(tmp_path)

    with run.context() as ctx:
        engine = WorkflowEngine(workflow)
        with pytest.raises(ValueError, match="missing required inputs"):
            engine.execute(ctx)
